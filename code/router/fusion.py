"""Fusion: combine the content and pattern scores into a routing decision.

    blended = ALPHA * content_urgency + (1 - ALPHA) * pattern_score

then a short chain of guardrail overrides that the blend alone cannot express.
Each override exists because of a specific failure the blend produces:

  risk_veto        A scam reads as urgent by design. Without a veto, the most
                   dangerous messages score highest and get promoted.
  broadcast        Bulk senders use urgency-bait ("LAST CHANCE"), which the
                   content layer cannot distinguish from real urgency.
  high_content     A stranger's genuine emergency scores content ~0.95 /
                   pattern ~0.10, blending to ~0.65 - digest, which is wrong.
  muted_group      An explicit user preference should outrank an inferred one,
                   except when the user is named directly.
  dnd_degrade      notify vs digest is partly a timing question; the blend has
                   no notion of the clock.

Every decision records which override fired, so the reason string and the UI
badges are generated from the same source of truth rather than restated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .content import ContentResult
from .dataset import Context, Message
from .features import PatternResult

RISK_VETO_THRESHOLD = 0.45


@dataclass
class Decision:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str
    content_score: float
    pattern_score: float
    blended_score: float
    risk_score: float
    overrides: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    backend: str = "heuristic"

    def as_row(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": f"{self.confidence:.2f}",
            "evidence_message_ids": self.evidence_message_ids,
        }


def _base_action(blended: float) -> str:
    if blended >= config.NOTIFY_THRESHOLD:
        return "notify"
    if blended >= config.DIGEST_THRESHOLD:
        return "digest"
    return "mute"


def _confidence(action: str, blended: float, overrides: list[str]) -> float:
    """Map decision margin into the per-action band observed in the samples.

    Distance from the nearer threshold is the natural margin: a message sitting
    on a boundary is genuinely less certain than one deep inside a band. An
    override is a high-precision rule, so it lands near the top of the band.
    """
    low, high = config.CONFIDENCE_BANDS[action]

    if action == "notify":
        margin = (blended - config.NOTIFY_THRESHOLD) / (1.0 - config.NOTIFY_THRESHOLD)
    elif action == "mute":
        margin = (config.DIGEST_THRESHOLD - blended) / config.DIGEST_THRESHOLD
    else:
        midpoint = (config.NOTIFY_THRESHOLD + config.DIGEST_THRESHOLD) / 2
        half_width = (config.NOTIFY_THRESHOLD - config.DIGEST_THRESHOLD) / 2
        margin = 1.0 - abs(blended - midpoint) / half_width

    margin = max(0.0, min(1.0, margin))
    if overrides:
        margin = max(margin, 0.75)

    return round(low + (high - low) * margin, 2)


def _reason(
    action: str,
    message_type: str,
    message: Message,
    content: ContentResult,
    pattern: PatternResult,
    overrides: list[str],
) -> str:
    """One-sentence explanation, phrased like dataset/sample_messages.csv.

    Generated from the decision path rather than from the model, so the
    explanation always matches the arithmetic that produced the action.
    """
    if "risk_veto" in overrides:
        if "domain_mismatch" in pattern.flags:
            return (
                "The sender is using a domain that does not match the official "
                "brand domain, which indicates an impersonation attempt."
            )
        return (
            "The message asks for credentials, payment, or verification in a way "
            "that matches known scam patterns."
        )

    if "broadcast_override" in overrides:
        return (
            "The sender behaves like a bulk broadcaster and this user has no "
            "active relationship with the account."
        )

    if "high_content_override" in overrides:
        return (
            "The content is time-critical enough to interrupt the user even though "
            "the sender is not a frequent contact."
        )

    if "muted_group" in overrides:
        return (
            "The user has muted this group and the message does not address them "
            "directly, so it is held for later."
        )

    if "dnd_degrade" in overrides:
        return (
            "The message is useful but arrived inside the user's quiet hours, so it "
            "is batched instead of interrupting."
        )

    trusted = pattern.score >= 0.6
    if action == "notify":
        if "direct_mention" in pattern.flags or "the user is addressed directly" in content.signals:
            return "The sender directly asks this user for a response or action."
        if "sender_is_admin" in pattern.flags:
            return "A trusted group admin sent a time-sensitive update that should interrupt the user."
        if message.conversation_type == "business":
            return "A verified business is sending an update that matches the user's recent activity."
        return "The message is time-sensitive and comes from a sender this user engages with."

    if action == "digest":
        if message_type == "promotion":
            return "The message is promotional but matches a business the user has engaged with."
        if message_type in {"event", "business_update"}:
            return "The message is useful information, but it is not urgent enough to interrupt the user."
        if trusted:
            return "The message is from a familiar sender but does not need immediate attention."
        return "The message is safe and potentially useful, but low priority for this user."

    if message_type in {"spam", "scam"}:
        return "The message is unsolicited and shows characteristics this user has reported before."
    if message_type == "greeting":
        return "The message is a routine social greeting with no action required."
    if message_type == "forward":
        return "The message is a widely forwarded chain message with low personal relevance."
    return "The user consistently ignores messages of this kind from this sender."


def decide(
    message: Message,
    context: Context,
    content: ContentResult,
    pattern: PatternResult,
    evidence_ids: str,
) -> Decision:
    blended = config.ALPHA * content.urgency + (1 - config.ALPHA) * pattern.score
    action = _base_action(blended)
    message_type = content.message_type
    overrides: list[str] = []

    # 1. Risk veto - highest precedence. A scam must never be promoted.
    # Exception: a verified business the user transacts with, sending from its
    # own domain, is exempt from a content-only risk score. Language models
    # reliably score hard-sell marketing as fraud ("act now", "limited slots"),
    # and without this guard legitimate promotions get muted as scams.
    risky_flags = {"domain_mismatch", "widely_reported"} & pattern.flags
    content_risk = content.risk
    if "trusted_business" in pattern.flags and not risky_flags:
        content_risk = min(content_risk, RISK_VETO_THRESHOLD - 0.01)

    if content_risk >= RISK_VETO_THRESHOLD or risky_flags:
        action = "mute"
        message_type = "scam" if (content_risk >= RISK_VETO_THRESHOLD or "domain_mismatch" in pattern.flags) else "spam"
        overrides.append("risk_veto")

    # 2. Known bulk broadcaster.
    elif "broadcast_like" in pattern.flags and pattern.score < config.BROADCAST_OVERRIDE + 0.2:
        action = "mute"
        if message_type not in {"promotion", "spam"}:
            message_type = "spam"
        overrides.append("broadcast_override")

    # 3. Genuine high-urgency content outranks an unfamiliar sender.
    elif content.urgency >= config.HIGH_CONTENT_OVERRIDE:
        if action != "notify":
            overrides.append("high_content_override")
        action = "notify"
        if message_type in {"unknown", "personal"}:
            message_type = "urgent"

    # 4. Explicit user preference: muted group, not addressed directly.
    elif "group_muted" in pattern.flags and "direct_mention" not in pattern.flags and action == "notify":
        action = "digest"
        overrides.append("muted_group")

    # 5. Quiet hours degrade an interrupt into a digest entry.
    elif "in_dnd" in pattern.flags and action == "notify" and content.urgency < config.HIGH_CONTENT_OVERRIDE:
        action = "digest"
        overrides.append("dnd_degrade")

    if message_type not in config.MESSAGE_TYPES:
        message_type = "unknown"

    return Decision(
        message_id=message.message_id,
        action=action,
        message_type=message_type,
        reason=_reason(action, message_type, message, content, pattern, overrides),
        confidence=_confidence(action, blended, overrides),
        evidence_message_ids=evidence_ids,
        content_score=round(content.urgency, 4),
        pattern_score=round(pattern.score, 4),
        blended_score=round(blended, 4),
        risk_score=round(content.risk, 4),
        overrides=overrides,
        signals=content.signals,
        backend=content.backend,
    )
