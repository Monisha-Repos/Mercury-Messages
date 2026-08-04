"""End-to-end orchestration: messages.csv in, decisions out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cache import Cache
from .content import score_content
from .dataset import Context, Message, load_messages
from .evidence import EvidenceIndex
from .features import compute_pattern
from .fusion import Decision, decide


@dataclass
class RunResult:
    decisions: list[Decision]
    messages: list[Message]
    cache_stats: dict[str, int]


def run(
    messages: list[Message] | None = None,
    context: Context | None = None,
    backend: str = "heuristic",
    cache_enabled: bool = True,
    dataset_dir: Path | None = None,
    progress: bool = False,
) -> RunResult:
    context = context or Context(dataset_dir)
    messages = messages if messages is not None else load_messages(
        (dataset_dir / "messages.csv") if dataset_dir else None
    )

    index = EvidenceIndex(context)
    cache = Cache(enabled=cache_enabled)
    decisions: list[Decision] = []

    try:
        for position, message in enumerate(messages, start=1):
            if progress:
                print(
                    f"  [{position:>3}/{len(messages)}] {message.message_id} ({message.modality})",
                    flush=True,
                )

            content = score_content(message, context, cache, backend=backend)
            pattern = compute_pattern(context, message)
            retrieved = index.retrieve(message)
            decisions.append(
                decide(message, context, content, pattern, index.format_ids(retrieved))
            )
        stats = cache.stats()
    finally:
        cache.close()

    return RunResult(decisions=decisions, messages=messages, cache_stats=stats)
