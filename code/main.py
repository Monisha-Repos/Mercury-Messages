"""Entry point: python code/main.py -> output.csv

    python code/main.py                      heuristic backend, cached
    python code/main.py --backend ollama     add LLM/VLM scoring
    python code/main.py --no-cache           force full recompute
    python code/main.py --limit 10 --verbose smoke test
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router import config, pipeline, writer
from router.dataset import Context, load_messages, summarize


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WhatsApp message notification router")
    parser.add_argument(
        "--backend",
        choices=["heuristic", "ollama"],
        default="heuristic",
        help="content scoring backend (default: heuristic, no models required)",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="dataset directory")
    parser.add_argument(
        "--out",
        type=Path,
        nargs="*",
        default=None,
        help="output path(s); defaults to ./output.csv and dataset/output.csv",
    )
    parser.add_argument("--trace", type=Path, default=None, help="also write a score-breakdown CSV")
    parser.add_argument("--no-cache", action="store_true", help="bypass the inference cache")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N messages")
    parser.add_argument("--verbose", action="store_true", help="print per-message progress")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    dataset_dir = args.dataset or config.DATASET_DIR
    context = Context(dataset_dir)
    messages = load_messages(dataset_dir / "messages.csv")
    if args.limit:
        messages = messages[: args.limit]

    stats = summarize(messages)
    print(f"Loaded {stats['total']} messages: {stats['by_modality']}")
    print(f"Backend: {args.backend} | cache: {'off' if args.no_cache else 'on'}")

    result = pipeline.run(
        messages=messages,
        context=context,
        backend=args.backend,
        cache_enabled=not args.no_cache,
        progress=args.verbose,
    )

    # Validate before writing: a malformed file scores zero however good the
    # reasoning was.
    writer.validate(result.decisions, messages)

    paths = writer.write(result.decisions, args.out)
    for path in paths:
        print(f"Wrote {path}")

    if args.trace:
        print(f"Wrote {writer.write_trace(result.decisions, args.trace)}")

    actions = Counter(d.action for d in result.decisions)
    types = Counter(d.message_type for d in result.decisions)
    overrides = Counter(o for d in result.decisions for o in d.overrides)
    evidence_hits = sum(1 for d in result.decisions if d.evidence_message_ids != "none")

    print(f"\nActions:   {dict(actions)}")
    print(f"Types:     {dict(types.most_common())}")
    print(f"Overrides: {dict(overrides) or 'none fired'}")
    print(f"Evidence:  {evidence_hits}/{len(result.decisions)} rows cite history")
    print(f"Cache:     {result.cache_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
