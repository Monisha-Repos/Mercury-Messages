"""Content layer: what does this message actually say?

Returns two independent scores, which is a deliberate departure from a single
"urgency" number:

  urgency - how time-critical the content is
  risk    - how scam-like or unsafe it is

They must stay separate because scams are engineered to read as urgent
("your account is blocked, verify now"). Collapsing them into one score means
the most dangerous messages score highest and get promoted to notify, which is
exactly backwards. fusion.py routes on urgency but lets risk veto.

The heuristic backend below is deterministic, dependency-free, and serves as
the ablation floor. The `ollama` backend swaps in an LLM per modality behind
the same interface; both are cached identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .cache import Cache
from .dataset import Context, Message


@dataclass
class ContentResult:
    urgency: float
    risk: float
    message_type: str
    reasoning: str
    signals: list[str] = field(default_factory=list)
    backend: str = "heuristic"
    transcript: str = ""


def _phrases(*terms: str) -> list[re.Pattern[str]]:
    return [re.compile(rf"\b{t}\b" if " " not in t else re.escape(t)) for t in terms]


# Lexical families. Ordered by specificity - the most diagnostic first.
# "click the link" is deliberately absent: legitimate businesses use it for
# feedback surveys and tracking. Link presence is handled below, but only as a
# multiplier on an existing credential or payment request.
SCAM_TERMS = _phrases(
    "otp", "one time password", "kyc", "verify your account", "account blocked",
    "account suspended", "claim now", "lottery", "winner",
    "prize", "reattempt fee", "unclaimed", "refund pending", "share the code",
    "pin", "password", "lucky draw", "processing fee", "penalty", "seized",
    "arrest", "legal action", "limited slots left", "act now",
)
PAYMENT_TERMS = _phrases(
    "payment", "invoice", "due", "bill", "amount", "upi", "transfer", "paid",
    "outstanding", "installment", "emi", "fees", "fee", "recharge", "wallet",
)
URGENT_TERMS = _phrases(
    "urgent", "urgently", "asap", "immediately", "emergency", "hospital",
    "accident", "ambulance", "right now", "call me", "please call", "need you",
    "critical", "serious", "help", "stuck", "cannot wait", "can not wait",
    "last chance", "deadline", "expires today", "final reminder",
)
TIME_TERMS = _phrases(
    "today", "tonight", "this evening", "in an hour", "within", "before",
    "by 5", "by 6", "by 7", "tomorrow morning", "shortly", "now", "minutes",
    "mins", "starting", "reaching", "reached",
)
# "trip" and "pickup" are omitted: they collide with travel promotions and
# neighbourhood classifieds respectively, which are not events.
EVENT_TERMS = _phrases(
    "meeting", "event", "rsvp", "invite", "invitation", "venue", "schedule",
    "rehearsal", "practice", "match", "exam", "class", "function", "ceremony",
    "celebration", "gathering", "form", "register", "registration", "slot",
    "appointment", "booking", "picnic", "session",
)
BUSINESS_TERMS = _phrases(
    "order", "delivery", "delivered", "shipped", "dispatched", "tracking",
    "statement", "ticket", "reservation", "confirmed", "prescription",
    "policy", "renewal", "subscription", "account update", "service",
)
# "flat" is omitted: in Indian English it means apartment ("flat no", "Tower B
# flats") far more often than "flat 50% off", and it was the single largest
# source of false promotion labels on residential group messages.
PROMO_TERMS = _phrases(
    "off", "sale", "discount", "offer", "deal", "coupon", "shop now",
    "tap below", "limited time", "book now", "cashback",
    "lowest price", "hurry", "exclusive", "new arrival",
    "per person", "starting at", "upgrade",
)
GREETING_TERMS = _phrases(
    "good morning", "good night", "happy birthday", "many happy returns",
    "congratulations", "happy anniversary", "happy new year", "diwali",
    "eid mubarak", "merry christmas", "blessings", "god bless", "stay blessed",
    "have a great day", "namaste", "greetings",
)


def _count(patterns: list[re.Pattern[str]], text: str) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_heuristic(message: Message, context: Context, extra_text: str = "") -> ContentResult:
    """Deterministic lexical baseline. No model, no network, no dependencies."""
    text = f"{message.message_text} {extra_text}".lower().strip()
    signals: list[str] = []

    scam_hits = _count(SCAM_TERMS, text)
    urgent_hits = _count(URGENT_TERMS, text)
    time_hits = _count(TIME_TERMS, text)
    event_hits = _count(EVENT_TERMS, text)
    business_hits = _count(BUSINESS_TERMS, text)
    promo_hits = _count(PROMO_TERMS, text)
    greeting_hits = _count(GREETING_TERMS, text)
    payment_hits = _count(PAYMENT_TERMS, text)

    has_link = bool(re.search(r"https?://|www\.|\b[a-z0-9-]+\.(?:in|com|net|org|co)\b", text))
    direct_mention = f"@{message.user_id}" in (message.message_text or "")
    asks_user = bool(re.search(r"\b(can you|could you|please (?:call|send|confirm|reply)|let me know|need your)\b", text))

    # --- risk ---------------------------------------------------------------
    risk = 0.0
    if scam_hits:
        risk += 0.30 + 0.12 * min(scam_hits - 1, 3)
        signals.append(f"scam-associated wording ({scam_hits} markers)")
    if has_link and (scam_hits or payment_hits):
        risk += 0.20
        signals.append("payment or credential request combined with a link")
    if message.forwarded_count >= 4:
        risk += 0.10
        signals.append("forwarded many times")
    risk = _clamp(risk)

    # --- urgency ------------------------------------------------------------
    urgency = 0.25
    if urgent_hits:
        urgency += 0.16 * min(urgent_hits, 3)
        signals.append("explicit urgency wording")
    if time_hits:
        urgency += 0.07 * min(time_hits, 3)
        signals.append("same-day time reference")
    if direct_mention:
        urgency += 0.20
        signals.append("the user is addressed directly")
    if asks_user:
        urgency += 0.14
        signals.append("a direct request for action")
    if message.conversation_type == "personal":
        urgency += 0.08
    if greeting_hits and not urgent_hits:
        urgency -= 0.18
        signals.append("social pleasantry")
    if promo_hits >= 2 and not direct_mention:
        urgency -= 0.14
        signals.append("promotional phrasing")
    if message.forwarded_count >= 2:
        urgency -= 0.08
    urgency = _clamp(urgency)

    # --- type ---------------------------------------------------------------
    scores = {
        "scam": risk * 2.2,
        "promotion": promo_hits * 0.9 + (0.5 if message.conversation_type == "business" else 0.0),
        "event": event_hits * 0.85,
        "payment": payment_hits * 0.8,
        "business_update": business_hits * 0.8 + (0.4 if message.conversation_type == "business" else 0.0),
        "greeting": greeting_hits * 1.4,
        "urgent": urgent_hits * 1.1 + (0.6 if urgency > 0.75 else 0.0),
        "personal": (1.2 if message.conversation_type == "personal" else 0.0)
        + (0.7 if direct_mention or asks_user else 0.0),
        "forward": 0.9 if message.forwarded_count >= 3 else 0.0,
    }
    message_type = max(scores, key=lambda key: (scores[key], key))
    if scores[message_type] <= 0.0:
        message_type = "unknown"

    # Bulk promotional content with no relationship reads as spam, not promotion.
    if message_type == "promotion" and message.forwarded_count >= 3:
        message_type = "spam"

    reasoning = "; ".join(signals) if signals else "no strong content signal detected"
    return ContentResult(
        urgency=round(urgency, 4),
        risk=round(risk, 4),
        message_type=message_type,
        reasoning=reasoning,
        signals=signals,
        backend="heuristic",
    )


def score_content(
    message: Message,
    context: Context,
    cache: Cache,
    backend: str = "heuristic",
) -> ContentResult:
    """Dispatch to the right modality branch, with caching around model calls."""
    # Voice runs the prosody branch in every mode: it needs no model, only
    # numpy and soundfile, so gating it behind the LLM backend would throw away
    # the acoustic signal on a default run.
    if message.modality == "voice":
        from .modalities import score_voice

        return score_voice(message, context, cache)

    if backend == "heuristic":
        return score_heuristic(message, context)

    if message.modality == "image":
        from .modalities import score_image

        return score_image(message, context, cache)

    from .modalities import score_text_llm

    return score_text_llm(message, context, cache)
