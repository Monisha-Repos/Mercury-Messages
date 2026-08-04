"""Dataset loading and joins.

Deliberately stdlib-only (`csv`). The dataset is small (110 messages, 412
history rows) so pandas buys nothing, and this build targets Windows ARM64
where third-party wheels are not guaranteed to install. Keeping the critical
path dependency-free means `output.csv` can always be produced.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


@dataclass
class Message:
    """One row of messages.csv (or sample_messages.csv)."""

    message_id: str
    user_id: str
    conversation_type: str
    group_id: str
    business_id: str
    sender_user_id: str
    created_at: str
    message_text: str
    media_type: str
    media_id: str
    forwarded_count: int
    # Present only in sample_messages.csv; used by the evaluation harness.
    gold: dict[str, str] = field(default_factory=dict)

    @property
    def timestamp(self) -> datetime | None:
        return _parse_ts(self.created_at)

    @property
    def modality(self) -> str:
        if self.media_type == "image":
            return "image"
        if self.media_type == "voice":
            return "voice"
        return "text"

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Message":
        gold = {
            key: row[key]
            for key in ("action", "message_type", "reason", "confidence", "evidence_message_ids")
            if key in row
        }
        return cls(
            message_id=row["message_id"].strip(),
            user_id=row.get("user_id", "").strip(),
            conversation_type=row.get("conversation_type", "").strip(),
            group_id=row.get("group_id", "").strip(),
            business_id=row.get("business_id", "").strip(),
            sender_user_id=row.get("sender_user_id", "").strip(),
            created_at=row.get("created_at", "").strip(),
            message_text=row.get("message_text", "") or "",
            media_type=row.get("media_type", "").strip(),
            media_id=row.get("media_id", "").strip(),
            forwarded_count=_int(row.get("forwarded_count")),
            gold=gold,
        )


class Context:
    """All supporting tables, indexed for O(1) lookup during scoring."""

    def __init__(self, dataset_dir: Path | None = None) -> None:
        base = Path(dataset_dir or config.DATASET_DIR)
        self.dataset_dir = base

        self.users = {r["user_id"]: r for r in _read(base / "users.csv")}
        self.groups = {r["group_id"]: r for r in _read(base / "groups.csv")}
        self.businesses = {r["business_id"]: r for r in _read(base / "business_accounts.csv")}

        self.group_members = {
            (r["group_id"], r["user_id"]): r for r in _read(base / "group_members.csv")
        }
        self.user_business = {
            (r["user_id"], r["business_id"]): r for r in _read(base / "user_business_history.csv")
        }

        self.history = [Message.from_row(r) for r in _read(base / "message_history.csv")]
        self.history_by_id = {m.message_id: m for m in self.history}

        self.events = {
            (r["user_id"], r["message_id"]): r for r in _read(base / "message_events.csv")
        }

        self.images = {r["image_id"]: r["file_path"] for r in _read(base / "images.csv")}
        self.voice_notes = {
            r["voice_note_id"]: r["file_path"] for r in _read(base / "voice_notes.csv")
        }

        self.notification_load = {
            (r["user_id"], r["date"]): r for r in _read(base / "daily_notification_summary.csv")
        }

        # Derived indexes.
        self._history_by_user: dict[str, list[Message]] = {}
        for msg in self.history:
            self._history_by_user.setdefault(msg.user_id, []).append(msg)

        self._admins_by_group: dict[str, set[str]] = {}
        for (group_id, user_id), row in self.group_members.items():
            if row.get("role", "").strip() == "admin":
                self._admins_by_group.setdefault(group_id, set()).add(user_id)

    # --- lookups ------------------------------------------------------------

    def history_for_user(self, user_id: str) -> list[Message]:
        return self._history_by_user.get(user_id, [])

    def event_for(self, user_id: str, message_id: str) -> dict[str, str] | None:
        return self.events.get((user_id, message_id))

    def is_admin(self, group_id: str, user_id: str) -> bool:
        return user_id in self._admins_by_group.get(group_id, set())

    def media_path(self, message: Message) -> Path | None:
        if message.modality == "image":
            rel = self.images.get(message.media_id)
        elif message.modality == "voice":
            rel = self.voice_notes.get(message.media_id)
        else:
            return None
        if not rel:
            return None
        path = self.dataset_dir / rel
        return path if path.exists() else None

    def dnd_window(self, user_id: str) -> tuple[int, int] | None:
        """Return (start_hour, end_hour) for the user's do-not-disturb window."""
        row = self.users.get(user_id)
        if not row:
            return None
        raw = (row.get("do_not_disturb_window") or "").strip()
        if "-" not in raw:
            return None
        start, _, end = raw.partition("-")
        try:
            return int(start.split(":")[0]), int(end.split(":")[0])
        except ValueError:
            return None

    def in_dnd(self, message: Message) -> bool:
        window = self.dnd_window(message.user_id)
        stamp = message.timestamp
        if not window or not stamp:
            return False
        start, end = window
        hour = stamp.hour
        # Windows normally wrap midnight (e.g. 22:00-07:00).
        return hour >= start or hour < end if start > end else start <= hour < end


def load_messages(path: Path | None = None) -> list[Message]:
    return [Message.from_row(r) for r in _read(Path(path or config.DATASET_DIR / "messages.csv"))]


def load_samples(path: Path | None = None) -> list[Message]:
    return [
        Message.from_row(r)
        for r in _read(Path(path or config.DATASET_DIR / "sample_messages.csv"))
    ]


def summarize(messages: list[Message]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for msg in messages:
        counts[msg.modality] = counts.get(msg.modality, 0) + 1
    return {"total": len(messages), "by_modality": counts}
