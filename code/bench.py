"""Timing check for the local models. Run this before writing pipeline code.

    python code/bench.py

Answers the only question that matters before committing to model sizes: how
long does one text message and one image take on this machine? Multiply by 87
text and 15 image messages to get the full-run cost.

If image scoring exceeds ~30s, drop to a smaller vision model and re-run:

    VISION_MODEL=qwen2.5vl:7b python code/bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router import config
from router.cache import Cache
from router.dataset import Context, load_messages
from router.modalities import ollama_available, score_image, score_text_llm
from router.content import score_heuristic


def timed(label: str, fn):
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - a benchmark should report, not crash
        print(f"  {label:<26} FAILED  {type(exc).__name__}: {str(exc)[:70]}")
        return None, 0.0
    elapsed = time.perf_counter() - start
    print(f"  {label:<26} {elapsed:6.1f}s")
    return result, elapsed


def main() -> int:
    print(f"Ollama host: {config.OLLAMA_HOST}")
    if not ollama_available():
        print("\nOllama is NOT reachable. Start it with:  ollama serve")
        print("Then re-run this script.")
        return 1
    print("Ollama is reachable.\n")

    context = Context()
    messages = load_messages()
    cache = Cache(enabled=False)  # always time the real call, never a cache hit

    text_msg = next(m for m in messages if m.modality == "text" and len(m.message_text) > 200)
    image_msg = next(m for m in messages if m.modality == "image")
    voice_msg = next(m for m in messages if m.modality == "voice")

    print(f"Text model:   {config.TEXT_MODEL}")
    print(f"Vision model: {config.VISION_MODEL}\n")

    print("Warm-up (model load into RAM is a one-off cost)")
    timed("text warm-up", lambda: score_text_llm(text_msg, context, cache))

    print("\nMeasured")
    _, text_time = timed(f"text  ({text_msg.message_id})", lambda: score_text_llm(text_msg, context, cache))
    _, image_time = timed(f"image ({image_msg.message_id})", lambda: score_image(image_msg, context, cache))

    from router.content import score_content

    _, voice_time = timed(f"voice ({voice_msg.message_id})", lambda: score_content(voice_msg, context, cache, "ollama"))
    timed("heuristic (reference)", lambda: score_heuristic(text_msg, context))

    counts = {"text": 87, "image": 15, "voice": 8}
    projected = text_time * counts["text"] + image_time * counts["image"] + voice_time * counts["voice"]
    print(f"\nProjected full run over 110 messages: {projected / 60:.1f} min (uncached)")
    if image_time > 30:
        print("Image scoring is slow. Try a smaller vision model:")
        print("  set VISION_MODEL=qwen2.5vl:7b   (Windows)")
    print("Cached re-runs are near-instant; only changed branches recompute.")

    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
