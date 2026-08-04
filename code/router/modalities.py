"""Modality-specific branches: voice prosody, image VLM, text LLM.

Every branch degrades gracefully. If a model or library is unavailable the
branch falls back to the heuristic scorer and records that in `backend`, so
the pipeline always produces a complete output.csv. On this machine that
matters: Windows ARM64 has no wheels for llvmlite (librosa), ctranslate2
(faster-whisper), or torch (openai-whisper).

The prosody implementation is therefore written directly against numpy rather
than librosa. Decoding is handled by soundfile, whose bundled libsndfile 1.2.2
reads MP3 natively. RMS energy, autocorrelation pitch, and pause segmentation
are ~80 lines and remove the dependency that would not build.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import config
from .cache import Cache, file_fingerprint
from .content import ContentResult, score_heuristic
from .dataset import Context, Message


# --- Ollama client ----------------------------------------------------------

def ollama_generate(
    model: str,
    prompt: str,
    images: list[str] | None = None,
    timeout: int = config.LLM_TIMEOUT_SECONDS,
    as_json: bool = True,
) -> str:
    """Single deterministic completion. temperature=0 plus a fixed seed.

    top_p is pinned to 1 as well: with temperature 0 the sampler is greedy, but
    leaving nucleus filtering at its default is a silent source of drift across
    Ollama versions.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "top_p": 1, "seed": config.SEED},
    }
    if as_json:
        payload["format"] = "json"
    if images:
        payload["images"] = images

    request = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8")).get("response", "")


def ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/tags", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def safe_generate(cache: Cache, namespace: str, key: dict, compute) -> str:
    """Run a cached model call, returning "" if the model is unavailable.

    A model that has not been pulled yet answers 404, and a half-loaded server
    can time out. Neither should abort a 110-message run: the caller falls back
    to the heuristic result and records that in `backend`, so a partial model
    set still produces a complete output.csv.
    """
    if config.CACHE_ONLY:
        # Deadline mode: use whatever inference is already cached and fall back
        # to the heuristic for the rest, rather than blocking on new calls.
        from .cache import fingerprint

        return cache.get(f"{namespace}:{fingerprint(namespace, key)}") or ""
    try:
        return cache.resolve(namespace, key, compute)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
        return ""


def _parse_json_response(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


# --- Prosody ----------------------------------------------------------------

MIN_AUDIO_SECONDS = 0.4


def _decode_audio(path: Path):
    """Decode an MP3, salvaging what is readable if the stream is damaged.

    Two of the supplied voice notes have corrupt frames partway through and
    abort libsndfile's whole-file read. Block reading keeps the audio decoded
    before the fault instead of discarding the message, which matters because
    prosody only needs a few seconds of speech.

    libmpg123 writes resync warnings straight to fd 2, so stderr is redirected
    around the decode to keep pipeline output readable.
    """
    import numpy as np
    import soundfile as sf

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        try:
            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        except sf.LibsndfileError:
            blocks = []
            with sf.SoundFile(str(path)) as handle:
                sample_rate = handle.samplerate
                try:
                    while True:
                        block = handle.read(frames=8192, dtype="float32", always_2d=False)
                        if not len(block):
                            break
                        blocks.append(block)
                except sf.LibsndfileError:
                    pass
            if not blocks:
                raise
            audio = np.concatenate(blocks)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def prosody_features(path: Path) -> dict[str, float]:
    """Acoustic correlates of urgency, computed without librosa.

    Transcription alone cannot separate "it's fine, don't worry" spoken calmly
    from the same words spoken through tears. These features measure that
    difference explicitly instead of asking a model to infer it.
    """
    import numpy as np

    audio, sample_rate = _decode_audio(path)
    # Two supplied notes are ~0.1s of silence; prosody over them is noise.
    if audio.size == 0 or len(audio) / sample_rate < MIN_AUDIO_SECONDS:
        return {}

    duration = len(audio) / sample_rate
    frame = int(0.025 * sample_rate)          # 25 ms
    hop = int(0.010 * sample_rate)            # 10 ms
    if frame <= 0 or len(audio) < frame:
        return {"duration_s": round(duration, 3)}

    frames = np.lib.stride_tricks.sliding_window_view(audio, frame)[::hop]

    # Energy.
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    peak = float(np.max(rms)) or 1e-12
    normalized = rms / peak
    voiced = normalized > 0.15

    # Pauses: runs of low-energy frames.
    pauses: list[int] = []
    run = 0
    for is_voiced in voiced:
        if is_voiced:
            if run:
                pauses.append(run)
            run = 0
        else:
            run += 1
    if run:
        pauses.append(run)
    pause_durations = [p * hop / sample_rate for p in pauses]
    long_pauses = [p for p in pause_durations if p >= 0.3]

    # Pitch via autocorrelation on voiced frames (60-400 Hz covers speech).
    min_lag, max_lag = int(sample_rate / 400), int(sample_rate / 60)
    f0_values: list[float] = []
    voiced_indices = np.flatnonzero(voiced)
    # Cap the number of analysed frames; pitch statistics converge quickly and
    # this keeps runtime flat on longer notes.
    for index in voiced_indices[:: max(1, len(voiced_indices) // 200)]:
        window = frames[index] - float(np.mean(frames[index]))
        correlation = np.correlate(window, window, mode="full")[len(window) - 1 :]
        segment = correlation[min_lag : min(max_lag, len(correlation))]
        if segment.size and float(np.max(segment)) > 0:
            lag = int(np.argmax(segment)) + min_lag
            if lag > 0:
                f0_values.append(sample_rate / lag)

    f0 = np.array(f0_values) if f0_values else np.array([0.0])
    zero_crossings = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))

    return {
        "duration_s": round(duration, 3),
        "rms_mean": round(float(np.mean(rms)), 6),
        "rms_max": round(peak, 6),
        "rms_std": round(float(np.std(rms)), 6),
        "dynamic_range": round(float(np.max(rms) / (np.mean(rms) + 1e-12)), 4),
        "voiced_ratio": round(float(np.mean(voiced)), 4),
        "f0_mean": round(float(np.mean(f0)), 2),
        "f0_std": round(float(np.std(f0)), 2),
        "f0_range": round(float(np.max(f0) - np.min(f0)), 2),
        "pause_count": float(len(long_pauses)),
        "longest_pause_s": round(max(pause_durations) if pause_durations else 0.0, 3),
        "speech_rate_proxy": round(float(np.mean(voiced)) / max(duration, 0.1), 4),
        "zero_crossing_rate": round(zero_crossings, 5),
    }


def prosody_urgency(features: dict[str, float]) -> tuple[float, list[str]]:
    """Map acoustic features to an urgency adjustment in roughly [-0.1, 0.35].

    Weights are hand-set rather than learned: with 8 voice notes in
    messages.csv there is no honest way to fit them, and saying so is better
    than implying a model that does not exist.
    """
    if not features:
        return 0.0, []

    delta = 0.0
    signals: list[str] = []

    if features.get("dynamic_range", 0) > 3.5 and features.get("rms_std", 0) > 0.05:
        delta += 0.12
        signals.append("raised, uneven vocal energy")

    if features.get("f0_std", 0) > 45:
        delta += 0.12
        signals.append("high pitch variability consistent with distress")

    if features.get("voiced_ratio", 0) > 0.72 and features.get("longest_pause_s", 1.0) < 0.35:
        delta += 0.10
        signals.append("fast, unbroken speech with few pauses")

    if features.get("pause_count", 0) >= 4 and features.get("duration_s", 0) < 20:
        delta += 0.06
        signals.append("halting delivery")

    if (
        features.get("dynamic_range", 0) < 2.2
        and features.get("f0_std", 0) < 25
        and features.get("voiced_ratio", 0) < 0.6
    ):
        delta -= 0.08
        signals.append("calm, evenly paced delivery")

    return round(delta, 4), signals


# --- Branches ---------------------------------------------------------------

VOICE_PROMPT = """You are triaging a WhatsApp voice note for a notification router.
Transcript: {transcript}

Reply with JSON only:
{{"urgency": <0-1>, "risk": <0-1>, "message_type": "<personal|urgent|event|payment|business_update|promotion|greeting|forward|spam|scam|unknown>", "reasoning": "<one short sentence>"}}"""

# Two-stage: the VLM describes, the text model classifies.
#
# A 7B vision model cannot reliably emit an 11-category taxonomy as structured
# JSON, but it describes layout and reads poster text well. Splitting the work
# lets each model do what it is good at, and the classification half reuses the
# already-calibrated TEXT_PROMPT instead of maintaining a second taxonomy.
IMAGE_PROMPT = """Describe this image for someone who cannot see it.

State plainly:
1. What kind of image it is - a marketing or promotional poster, a screenshot of
   a chat conversation, a personal photo, a document or official notice, or a
   meme or forwarded graphic.
2. Any text visible in the image, quoted as accurately as you can.
3. Its visual style - bright gradients, price badges, discount callouts and
   logos suggest advertising; a plain chat layout suggests a forwarded
   screenshot.

Answer in under 120 words."""

TEXT_PROMPT = """You are triaging a WhatsApp message for a notification router.

Conversation type: {conversation_type}
Recipient user id: {user_id}
Forwarded count: {forwarded_count}
Message:
\"\"\"{text}\"\"\"

Score two things INDEPENDENTLY.

urgency (0-1): how time-critical this is for the recipient.
  0.8-1.0  needs action within hours: same-day schedule changes, deadlines
           today, emergencies, someone waiting on a reply now
  0.5-0.7  useful soon but not immediate: upcoming events, order updates
  0.2-0.4  informational, read whenever
  0.0-0.1  pleasantries, chain forwards, generic marketing
  Raise urgency when the recipient is named with @{user_id}, or is directly
  asked to call, confirm, send, or join something.

risk (0-1): how likely this is a SCAM. Reserve high risk for messages that
  request credentials, OTPs, card details, or payment to an unexpected place,
  or impersonate a brand or authority. Aggressive marketing is NOT a scam:
  discounts, countdowns and hard-sell copy from a real business score risk
  below 0.2. A scam that sounds urgent gets HIGH risk and LOW urgency.

message_type - pick the single best fit:
  urgent          time-critical, needs attention now
  event           something scheduled: meetings, trips, functions, bookings,
                  school or society activities with a date or time
  personal        one-to-one conversation, or a direct ask aimed at this user
  payment         a bill, invoice, due amount, or payment confirmation
  business_update ONLY from a business about the user's own order, delivery,
                  booking, policy or account. Never use this for messages sent
                  by a person in a group chat, even operational ones.
  promotion       marketing from a business the user may know
  greeting        good morning, festival wishes, congratulations
  forward         a chain message circulated widely, not written to this group
  spam            unsolicited bulk marketing from an unknown sender
  scam            fraud, phishing, impersonation
  unknown         genuinely unclear

Reply with JSON only:
{{"urgency": <0-1>, "risk": <0-1>, "message_type": "<one value from the list>", "reasoning": "<one short sentence>"}}"""


def _merge_llm(base: ContentResult, parsed: dict[str, Any], backend: str) -> ContentResult:
    """Trust the model's semantics, keep the heuristic as a floor on risk.

    Risk uses max() rather than replacement because the lexical scam markers
    are high precision; a model that misses one should not be able to lower the
    score below what the deterministic layer already established.
    """
    if not parsed:
        return base
    try:
        urgency = float(parsed.get("urgency", base.urgency))
        risk = float(parsed.get("risk", base.risk))
    except (TypeError, ValueError):
        return base

    message_type = str(parsed.get("message_type", base.message_type)).strip()
    if message_type not in config.MESSAGE_TYPES:
        message_type = base.message_type

    reasoning = str(parsed.get("reasoning", base.reasoning)).strip() or base.reasoning
    return ContentResult(
        urgency=round(max(0.0, min(1.0, urgency)), 4),
        risk=round(max(0.0, min(1.0, max(risk, base.risk))), 4),
        message_type=message_type,
        reasoning=reasoning,
        signals=base.signals + [f"{backend} assessment"],
        backend=backend,
        transcript=base.transcript,
    )


def score_text_llm(
    message: Message,
    context: Context,
    cache: Cache,
    text_override: str | None = None,
    base: ContentResult | None = None,
) -> ContentResult:
    """Classify message text. `text_override` lets the image branch reuse this
    with a VLM-generated description standing in for the message body."""
    base = base or score_heuristic(message, context)
    if not ollama_available():
        return base

    prompt = TEXT_PROMPT.format(
        conversation_type=message.conversation_type,
        user_id=message.user_id,
        forwarded_count=message.forwarded_count,
        text=(text_override if text_override is not None else message.message_text)[:4000],
    )
    raw = safe_generate(
        cache,
        "text_llm",
        {"model": config.TEXT_MODEL, "prompt": prompt},
        lambda: ollama_generate(config.TEXT_MODEL, prompt),
    )
    return _merge_llm(base, _parse_json_response(raw), config.TEXT_MODEL)


def score_image(message: Message, context: Context, cache: Cache) -> ContentResult:
    base = score_heuristic(message, context)
    path = context.media_path(message)
    if not path or not ollama_available():
        return base

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    raw = safe_generate(
        cache,
        "image_vlm",
        {"model": config.VISION_MODEL, "file": file_fingerprint(path), "prompt": IMAGE_PROMPT},
        lambda: ollama_generate(config.VISION_MODEL, IMAGE_PROMPT, images=[encoded]),
    )
    description = raw.strip()
    if not description:
        return base

    # Stage two: hand the description to the calibrated text classifier.
    caption = message.message_text.strip()
    combined = f"[Image received] {description}"
    if caption:
        combined = f"{combined}\n\n[Caption sent with the image] {caption}"

    result = score_text_llm(message, context, cache, text_override=combined, base=base)
    result.backend = f"{config.VISION_MODEL}+{config.TEXT_MODEL}"
    result.transcript = description[:500]
    result.signals = base.signals + ["image described by vision model"]
    return result


def score_voice(message: Message, context: Context, cache: Cache) -> ContentResult:
    """Prosody-first. Transcription is layered on only when a model exists.

    On Windows ARM64 no local speech-to-text installs via pip, so this branch
    normally runs on acoustic features plus conversation metadata alone.
    """
    base = score_heuristic(message, context)
    path = context.media_path(message)
    if not path:
        return base

    try:
        features = cache.resolve(
            "prosody",
            {"file": file_fingerprint(path), "version": 1},
            lambda: prosody_features(path),
        )
    except (ImportError, RuntimeError, OSError) as exc:
        base.signals.append(f"prosody unavailable ({type(exc).__name__})")
        return base

    delta, prosody_signals = prosody_urgency(features)
    urgency = max(0.0, min(1.0, base.urgency + delta))

    message_type = base.message_type
    if message_type == "unknown":
        message_type = "urgent" if urgency >= config.HIGH_CONTENT_OVERRIDE else "personal"

    signals = base.signals + prosody_signals
    return ContentResult(
        urgency=round(urgency, 4),
        risk=base.risk,
        message_type=message_type,
        reasoning="; ".join(signals) if signals else "voice note assessed on acoustic features",
        signals=signals,
        backend="prosody",
    )
