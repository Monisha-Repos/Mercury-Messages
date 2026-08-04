"""Content-addressed cache for model inference.

Every expensive call (LLM, VLM, transcription, prosody) is keyed on a hash of
the exact inputs that produced it, so re-running the pipeline after editing an
unrelated prompt only recomputes what actually changed.

This serves three purposes at once:
  1. Speed - the tuning loop stays interactive despite CPU-only inference.
  2. Determinism - a cache hit is byte-identical to the first run.
  3. Demo safety - the demo set is warm, so nothing runs live on stage.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from . import config


def fingerprint(*parts: Any) -> str:
    """Stable hash over arbitrary inputs. sort_keys makes dict order irrelevant."""
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Cache:
    def __init__(self, path: Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path or config.CACHE_PATH)
        self.hits = 0
        self.misses = 0
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                "  key TEXT PRIMARY KEY,"
                "  namespace TEXT NOT NULL,"
                "  value TEXT NOT NULL,"
                "  created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            self._conn.commit()

    def get(self, key: str) -> Any | None:
        if not self._conn:
            return None
        row = self._conn.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, namespace: str, value: Any) -> None:
        if not self._conn:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO entries (key, namespace, value) VALUES (?, ?, ?)",
            (key, namespace, json.dumps(value, ensure_ascii=False)),
        )
        self._conn.commit()

    def resolve(self, namespace: str, key_parts: Any, compute: Callable[[], Any]) -> Any:
        """Return the cached value for `key_parts`, computing it on a miss."""
        key = f"{namespace}:{fingerprint(namespace, key_parts)}"
        cached = self.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        value = compute()
        self.set(key, namespace, value)
        return value

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def clear(self, namespace: str | None = None) -> None:
        if not self._conn:
            return
        if namespace:
            self._conn.execute("DELETE FROM entries WHERE namespace = ?", (namespace,))
        else:
            self._conn.execute("DELETE FROM entries")
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
