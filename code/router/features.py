"""Pattern layer: how much does THIS user care about THIS sender?

Computed purely from metadata and historical engagement - it never looks at
what the current message says. That separation is the point of the
architecture: `pattern_score` is a prior about the sender, `content_score` is
evidence about the message, and fusion.py combines them. Keeping them
independent is what makes the ablation in code/evaluation/main.py meaningful.

Also emits `flags`, a set of deterministic risk markers. The most valuable is
`domain_mismatch`: business_accounts.csv carries both `official_domain` and
`domain_used_by_sender`, so a sender using a lookalike domain is detectable
exactly rather than probabilistically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import Context, Message, _int


@dataclass
class PatternResult:
    score: float
    features: dict[str, float] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


def _rate(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _user_baseline(context: Context, user_id: str) -> dict[str, float]:
    row = context.users.get(user_id, {})
    opened = _int(row.get("messages_opened_30d"))
    replied = _int(row.get("messages_replied_30d"))
    dismissed = _int(row.get("notifications_dismissed_30d"))
    reported = _int(row.get("messages_reported_30d"))
    total = opened + dismissed or 1
    return {
        "user_open_share": _rate(opened, total, 0.5),
        "user_reply_ratio": _rate(replied, max(opened, 1)),
        "user_dismiss_share": _rate(dismissed, total),
        "user_reported_30d": float(reported),
    }


def _conversation_engagement(context: Context, message: Message) -> dict[str, float]:
    """Engagement the user showed toward past messages in this same conversation."""
    opened = replied = dismissed = reported = muted = 0
    counted = 0

    for past in context.history_for_user(message.user_id):
        same_conversation = (
            (message.group_id and past.group_id == message.group_id)
            or (message.business_id and past.business_id == message.business_id)
            or (message.sender_user_id and past.sender_user_id == message.sender_user_id)
        )
        if not same_conversation:
            continue
        event = context.event_for(message.user_id, past.message_id)
        if not event:
            continue
        counted += 1
        opened += _int(event.get("message_opened"))
        replied += _int(event.get("message_replied"))
        dismissed += _int(event.get("notification_dismissed"))
        reported += _int(event.get("message_reported"))
        muted += _int(event.get("muted_after_message"))

    return {
        "conv_history_n": float(counted),
        "conv_open_rate": _rate(opened, counted, 0.5),
        "conv_reply_rate": _rate(replied, counted),
        "conv_dismiss_rate": _rate(dismissed, counted),
        "conv_report_rate": _rate(reported, counted),
        "conv_mute_rate": _rate(muted, counted),
    }


def _group_features(context: Context, message: Message) -> tuple[dict[str, float], set[str], list[str]]:
    features: dict[str, float] = {}
    flags: set[str] = set()
    notes: list[str] = []

    group = context.groups.get(message.group_id, {})
    membership = context.group_members.get((message.group_id, message.user_id), {})

    muted = _int(membership.get("group_muted_by_user"))
    features["group_muted"] = float(muted)
    if muted:
        flags.add("group_muted")
        notes.append("the user has muted this group")

    read = _int(membership.get("messages_read_30d"))
    replies = _int(membership.get("replies_sent_30d"))
    dismissals = _int(membership.get("notifications_dismissed_30d"))
    group_volume = max(_int(group.get("messages_30d")), 1)

    features["group_read_rate"] = _clamp(_rate(read, group_volume))
    features["group_reply_rate"] = _clamp(_rate(replies, group_volume))
    features["group_dismiss_pressure"] = _clamp(_rate(dismissals, max(read + dismissals, 1)))
    features["group_size"] = float(_int(group.get("member_count")))
    features["group_is_large"] = 1.0 if _int(group.get("member_count")) > 50 else 0.0

    # A message the user is directly addressed in outranks group-level apathy.
    if f"@{message.user_id}" in (message.message_text or ""):
        flags.add("direct_mention")
        features["direct_mention"] = 1.0
        notes.append("the user is mentioned directly")
    else:
        features["direct_mention"] = 0.0

    if message.sender_user_id and context.is_admin(message.group_id, message.sender_user_id):
        flags.add("sender_is_admin")
        features["sender_is_admin"] = 1.0
        notes.append("the sender is a group admin")
    else:
        features["sender_is_admin"] = 0.0

    group_type = (group.get("group_type") or "").strip()
    trusted_types = {"family", "school", "work", "close_friends"}
    features["group_type_trusted"] = 1.0 if group_type in trusted_types else 0.0

    return features, flags, notes


def _business_features(context: Context, message: Message) -> tuple[dict[str, float], set[str], list[str]]:
    features: dict[str, float] = {}
    flags: set[str] = set()
    notes: list[str] = []

    business = context.businesses.get(message.business_id, {})
    relationship = context.user_business.get((message.user_id, message.business_id), {})

    verified = _int(business.get("verified"))
    features["business_verified"] = float(verified)
    if verified:
        notes.append("the sender is a verified business")

    official = (business.get("official_domain") or "").strip().lower()
    used = (business.get("domain_used_by_sender") or "").strip().lower()
    mismatch = bool(official and used and official != used)

    # A mismatch only indicates impersonation when the account is unverified.
    # Verified brands legitimately send from marketing subdomains and link
    # shorteners: in this dataset both verified mismatches are long-established
    # domains with single-digit reports, while all 21 unverified mismatches are
    # lookalikes (hdfcbank-kyc.in, chase-secure-alert.com) with 10-77 reports.
    impersonation = mismatch and not verified
    features["domain_mismatch"] = 1.0 if impersonation else 0.0
    if impersonation:
        flags.add("domain_mismatch")
        notes.append(f"the sender uses {used} instead of the official domain {official}")
    elif mismatch:
        flags.add("benign_sending_domain")

    domain_age = _int(business.get("domain_used_by_sender_age_days"), -1)
    features["sender_domain_age_days"] = float(max(domain_age, 0))
    if 0 <= domain_age < 90:
        flags.add("young_domain")
        notes.append("the sending domain was registered recently")

    reports = _int(business.get("user_reports_30d"))
    features["business_reports_30d"] = float(reports)
    if reports >= 25:
        flags.add("widely_reported")
        notes.append("other users have reported this sender")

    features["business_account_age_days"] = float(_int(business.get("account_age_days")))
    features["business_volume_30d"] = float(_int(business.get("messages_sent_30d")))

    has_relationship = bool(relationship)
    features["has_business_relationship"] = 1.0 if has_relationship else 0.0
    features["business_activity_180d"] = float(_int(relationship.get("activity_count_180d")))

    allows_promotions = _int(relationship.get("allows_promotions")) if has_relationship else 0
    features["allows_promotions"] = float(allows_promotions)
    if has_relationship and not allows_promotions:
        flags.add("promotions_not_allowed")
    if relationship.get("promotions_opted_out_at"):
        flags.add("opted_out_of_promotions")
        notes.append("the user opted out of promotions from this business")

    opened = _int(relationship.get("messages_opened_30d"))
    dismissed = _int(relationship.get("messages_dismissed_30d"))
    replied = _int(relationship.get("messages_replied_30d"))
    seen = max(opened + dismissed, 1)
    features["business_open_rate"] = _rate(opened, seen, 0.5)
    features["business_dismiss_rate"] = _rate(dismissed, seen)
    features["business_reply_rate"] = _rate(replied, seen)

    # A verified brand the user actually transacts with, sending from its own
    # domain, is not a scam however aggressive the copy reads. This guards the
    # risk veto against a content model that scores marketing language as
    # fraud - it cannot see that the account is legitimate, but this can.
    if verified and has_relationship and not impersonation and reports < 25:
        flags.add("trusted_business")

    if has_relationship and relationship.get("why_user_knows_account"):
        notes.append(
            "the user has recent activity with this business "
            f"({relationship['why_user_knows_account'].replace('_', ' ')})"
        )

    # Bulk broadcaster with no relationship and heavy dismissal -> override input.
    if (
        not has_relationship
        and _int(business.get("messages_sent_30d")) > 800
    ) or features["business_dismiss_rate"] > 0.7:
        flags.add("broadcast_like")

    return features, flags, notes


def compute_pattern(context: Context, message: Message) -> PatternResult:
    """Return a 0-1 prior for how much this user values this sender."""
    features: dict[str, float] = {}
    flags: set[str] = set()
    notes: list[str] = []

    features.update(_user_baseline(context, message.user_id))
    features.update(_conversation_engagement(context, message))

    if message.conversation_type == "group":
        group_features, group_flags, group_notes = _group_features(context, message)
        features.update(group_features)
        flags |= group_flags
        notes += group_notes
    elif message.conversation_type == "business":
        business_features, business_flags, business_notes = _business_features(context, message)
        features.update(business_features)
        flags |= business_flags
        notes += business_notes

    features["forwarded_count"] = float(message.forwarded_count)
    if message.forwarded_count >= 3:
        flags.add("heavily_forwarded")
        notes.append("the message has been forwarded many times")

    features["in_dnd_window"] = 1.0 if context.in_dnd(message) else 0.0
    if features["in_dnd_window"]:
        flags.add("in_dnd")

    stamp = message.timestamp
    if stamp:
        load = context.notification_load.get((message.user_id, stamp.strftime("%Y-%m-%d")), {})
        sent = _int(load.get("notifications_sent"))
        dismissed = _int(load.get("notifications_dismissed"))
        features["day_notifications_sent"] = float(sent)
        features["day_dismiss_rate"] = _rate(dismissed, max(sent, 1))
        if sent >= 8 and features["day_dismiss_rate"] > 0.6:
            flags.add("notification_fatigue")

    score = _heuristic_score(message, features, flags)
    return PatternResult(score=score, features=features, flags=flags, notes=notes)


def _heuristic_score(message: Message, features: dict[str, float], flags: set[str]) -> float:
    """Interpretable additive prior, used as the baseline pattern model.

    The learned alternative (model.py) replaces this with logistic regression
    fitted on message_events.csv. Both are kept so the ablation can show what
    the learned layer actually adds.
    """
    score = 0.5

    if message.conversation_type == "group":
        score += 0.22 * features.get("group_reply_rate", 0.0)
        score += 0.14 * features.get("group_read_rate", 0.0)
        score -= 0.18 * features.get("group_dismiss_pressure", 0.0)
        score += 0.10 * features.get("group_type_trusted", 0.0)
        score += 0.08 * features.get("sender_is_admin", 0.0)
        score += 0.25 * features.get("direct_mention", 0.0)
        score -= 0.28 * features.get("group_muted", 0.0)
        score -= 0.05 * features.get("group_is_large", 0.0)

    elif message.conversation_type == "business":
        score += 0.12 * features.get("business_verified", 0.0)
        score += 0.18 * features.get("has_business_relationship", 0.0)
        score += 0.16 * features.get("business_open_rate", 0.0)
        score += 0.10 * features.get("business_reply_rate", 0.0)
        score -= 0.24 * features.get("business_dismiss_rate", 0.0)
        score -= 0.30 * features.get("domain_mismatch", 0.0)
        if "opted_out_of_promotions" in flags:
            score -= 0.18
        if "widely_reported" in flags:
            score -= 0.15

    else:  # personal
        score += 0.20 * features.get("conv_reply_rate", 0.0)
        score += 0.12 * features.get("conv_open_rate", 0.0)
        score -= 0.15 * features.get("conv_dismiss_rate", 0.0)
        score += 0.10  # 1:1 conversations carry an inherent priority

    score -= 0.10 * features.get("conv_report_rate", 0.0)
    score -= 0.08 * features.get("conv_mute_rate", 0.0)
    if "heavily_forwarded" in flags:
        score -= 0.08
    if "notification_fatigue" in flags:
        score -= 0.05

    return round(_clamp(score), 4)
