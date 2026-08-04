"""Evaluation harness: scoring, ablation, and threshold calibration.

Run:
    python code/evaluation/main.py                 score + ablation
    python code/evaluation/main.py --calibrate     grid-search ALPHA/thresholds
    python code/evaluation/main.py --errors        show every misclassification

Ground truth is dataset/sample_messages.csv (30 labeled rows). That is a small
set, so the calibration path reports leave-one-out accuracy alongside the
fitted number: tuning two parameters on 30 examples and quoting the tuned
accuracy would overstate performance, and the gap between the two is itself
worth knowing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import config
from router.cache import Cache
from router.content import score_content
from router.dataset import Context, Message, load_samples
from router.evidence import EvidenceIndex
from router.features import compute_pattern
from router.fusion import decide


def score_components(
    samples: list[Message], context: Context, backend: str = "heuristic"
) -> list[dict]:
    """Compute content/pattern/evidence once; ablations reuse these."""
    index = EvidenceIndex(context)
    cache = Cache()
    rows = []
    try:
        for message in samples:
            content = score_content(message, context, cache, backend=backend)
            pattern = compute_pattern(context, message)
            retrieved = index.retrieve(message)
            rows.append({
                "message": message,
                "content": content,
                "pattern": pattern,
                "evidence": index.format_ids(retrieved),
            })
    finally:
        cache.close()
    return rows


def _action_from(score: float, notify: float, digest: float) -> str:
    if score >= notify:
        return "notify"
    if score >= digest:
        return "digest"
    return "mute"


def predict(
    row: dict,
    mode: str = "full",
    alpha: float | None = None,
    notify: float | None = None,
    digest: float | None = None,
) -> tuple[str, str]:
    """Return (action, message_type) for one ablation mode."""
    alpha = config.ALPHA if alpha is None else alpha
    notify = config.NOTIFY_THRESHOLD if notify is None else notify
    digest = config.DIGEST_THRESHOLD if digest is None else digest

    content, pattern = row["content"], row["pattern"]

    if mode == "content_only":
        return _action_from(content.urgency, notify, digest), content.message_type
    if mode == "pattern_only":
        return _action_from(pattern.score, notify, digest), content.message_type
    if mode == "fused":
        blended = alpha * content.urgency + (1 - alpha) * pattern.score
        return _action_from(blended, notify, digest), content.message_type

    # full: fusion plus guardrail overrides
    saved = (config.ALPHA, config.NOTIFY_THRESHOLD, config.DIGEST_THRESHOLD)
    config.ALPHA, config.NOTIFY_THRESHOLD, config.DIGEST_THRESHOLD = alpha, notify, digest
    try:
        decision = decide(row["message"], None, content, pattern, row["evidence"])
    finally:
        config.ALPHA, config.NOTIFY_THRESHOLD, config.DIGEST_THRESHOLD = saved
    return decision.action, decision.message_type


def evaluate(rows: list[dict], mode: str = "full", **params) -> dict:
    action_hits = type_hits = both_hits = 0
    evidence_hits = evidence_total = 0
    confusion: Counter[tuple[str, str]] = Counter()

    for row in rows:
        gold = row["message"].gold
        action, message_type = predict(row, mode=mode, **params)

        if action == gold.get("action"):
            action_hits += 1
        if message_type == gold.get("message_type"):
            type_hits += 1
        if action == gold.get("action") and message_type == gold.get("message_type"):
            both_hits += 1
        confusion[(gold.get("action", "?"), action)] += 1

        gold_evidence = {e for e in (gold.get("evidence_message_ids") or "").split(";") if e and e != "none"}
        if gold_evidence:
            evidence_total += 1
            if gold_evidence & {e for e in row["evidence"].split(";") if e != "none"}:
                evidence_hits += 1

    total = len(rows) or 1
    return {
        "n": len(rows),
        "action_acc": action_hits / total,
        "type_acc": type_hits / total,
        "joint_acc": both_hits / total,
        "evidence_recall": evidence_hits / evidence_total if evidence_total else 0.0,
        "confusion": confusion,
    }


def print_ablation(rows: list[dict]) -> None:
    print("\nAblation - what each layer contributes")
    print(f"{'mode':<16}{'action':>9}{'type':>8}{'joint':>8}")
    print("-" * 41)
    for mode, label in [
        ("content_only", "content only"),
        ("pattern_only", "pattern only"),
        ("fused", "fused"),
        ("full", "fused+overrides"),
    ]:
        result = evaluate(rows, mode=mode)
        print(
            f"{label:<16}{result['action_acc']:>8.1%}"
            f"{result['type_acc']:>8.1%}{result['joint_acc']:>8.1%}"
        )


def print_confusion(result: dict) -> None:
    actions = ["notify", "digest", "mute"]
    print("\nConfusion (rows = gold, cols = predicted)")
    print(f"{'':<9}" + "".join(f"{a:>9}" for a in actions))
    for gold in actions:
        cells = "".join(f"{result['confusion'].get((gold, pred), 0):>9}" for pred in actions)
        print(f"{gold:<9}{cells}")


def calibrate(rows: list[dict]) -> dict:
    """Grid-search ALPHA and thresholds, then report leave-one-out accuracy.

    LOO refits on 29 examples and predicts the held-out one, 30 times. It is
    the honest number to quote, because the fitted accuracy has seen every row
    it is scored on.
    """
    alphas = [round(0.4 + 0.05 * i, 2) for i in range(9)]          # 0.40 - 0.80
    notifies = [round(0.40 + 0.025 * i, 3) for i in range(13)]     # 0.40 - 0.70
    digests = [round(0.15 + 0.025 * i, 3) for i in range(11)]      # 0.15 - 0.40

    def best_params(subset: list[dict]) -> tuple[float, float, float, float]:
        best = (-1.0, config.ALPHA, config.NOTIFY_THRESHOLD, config.DIGEST_THRESHOLD)
        for alpha in alphas:
            for notify in notifies:
                for digest in digests:
                    if digest >= notify:
                        continue
                    accuracy = evaluate(
                        subset, mode="full", alpha=alpha, notify=notify, digest=digest
                    )["action_acc"]
                    # Tie-break toward the current defaults for stability.
                    if accuracy > best[0]:
                        best = (accuracy, alpha, notify, digest)
        return best

    fitted_acc, alpha, notify, digest = best_params(rows)

    loo_hits = 0
    for index in range(len(rows)):
        train = rows[:index] + rows[index + 1 :]
        _, a, n, d = best_params(train)
        held_out = rows[index]
        predicted, _ = predict(held_out, mode="full", alpha=a, notify=n, digest=d)
        if predicted == held_out["message"].gold.get("action"):
            loo_hits += 1

    return {
        "alpha": alpha,
        "notify": notify,
        "digest": digest,
        "fitted_acc": fitted_acc,
        "loo_acc": loo_hits / len(rows),
        "baseline_acc": evaluate(rows, mode="full")["action_acc"],
    }


def print_errors(rows: list[dict]) -> None:
    print("\nMisclassifications")
    for row in rows:
        gold = row["message"].gold
        action, message_type = predict(row, mode="full")
        if action == gold.get("action") and message_type == gold.get("message_type"):
            continue
        message = row["message"]
        print(f"\n  {message.message_id} ({message.conversation_type}/{message.modality})")
        print(f"    gold: {gold.get('action')}/{gold.get('message_type')}"
              f"  predicted: {action}/{message_type}")
        print(f"    content={row['content'].urgency:.2f} risk={row['content'].risk:.2f} "
              f"pattern={row['pattern'].score:.2f}")
        print(f"    text: {(message.message_text or '')[:110].strip()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the notification router")
    parser.add_argument("--backend", choices=["heuristic", "ollama"], default="heuristic")
    parser.add_argument("--calibrate", action="store_true", help="grid-search parameters")
    parser.add_argument("--errors", action="store_true", help="list misclassifications")
    args = parser.parse_args(argv)

    context = Context()
    samples = load_samples()
    rows = score_components(samples, context, backend=args.backend)

    result = evaluate(rows, mode="full")
    print(f"Evaluating {result['n']} labeled samples (backend: {args.backend})")
    print(f"  action accuracy     {result['action_acc']:.1%}")
    print(f"  type accuracy       {result['type_acc']:.1%}")
    print(f"  joint accuracy      {result['joint_acc']:.1%}")
    print(f"  evidence recall     {result['evidence_recall']:.1%}")

    print_confusion(result)
    print_ablation(rows)

    if args.calibrate:
        print("\nCalibration (grid search + leave-one-out)")
        found = calibrate(rows)
        print(f"  current defaults    {found['baseline_acc']:.1%}")
        print(f"  best fitted         {found['fitted_acc']:.1%}  "
              f"(alpha={found['alpha']}, notify={found['notify']}, digest={found['digest']})")
        print(f"  leave-one-out       {found['loo_acc']:.1%}  <- honest estimate")

    if args.errors:
        print_errors(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
