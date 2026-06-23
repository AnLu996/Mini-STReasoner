"""Classify each case into a text-vs-ECG dominance class.

Decision tree over the per-case ``text_score`` (sensitivity to question edits)
and ``ecg_score`` (sensitivity to signal perturbations), both in ``[0, 1]``:

1. both below ``inactive``            -> UNCLEAR   (nothing moved the answer)
2. both above ``high``                -> UNSTABLE  (flips on almost everything)
3. text_score - ecg_score >= margin   -> TEXT_DOMINANT
4. ecg_score - text_score >= margin   -> ECG_DOMINANT
5. otherwise                          -> BALANCED

Thresholds live in ``counterfactual.DOMINANCE_THRESHOLDS``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual import DOMINANCE_THRESHOLDS  # noqa: E402


def classify(text_score: float, ecg_score: float, thresholds: dict[str, float] | None = None) -> tuple[str, str]:
    """Return (dominance_class, human-readable reason)."""
    t = thresholds or DOMINANCE_THRESHOLDS
    inactive, high, margin = t["inactive"], t["high"], t["margin"]
    gap = text_score - ecg_score
    if text_score < inactive and ecg_score < inactive:
        return "UNCLEAR", f"neither modality moved the answer (text={text_score:.2f}, ecg={ecg_score:.2f})"
    if text_score > high and ecg_score > high:
        return "UNSTABLE", f"answer flips under nearly every perturbation (text={text_score:.2f}, ecg={ecg_score:.2f})"
    if gap >= margin:
        return "TEXT_DOMINANT", f"text edits dominate (text={text_score:.2f} > ecg={ecg_score:.2f})"
    if -gap >= margin:
        return "ECG_DOMINANT", f"ECG perturbations dominate (ecg={ecg_score:.2f} > text={text_score:.2f})"
    return "BALANCED", f"both modalities matter comparably (text={text_score:.2f}, ecg={ecg_score:.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify per-case modal dominance")
    parser.add_argument("--per-case", type=Path, default=PROJECT_ROOT / "outputs/metrics/per_case_metrics.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/metrics/dominance.jsonl")
    parser.add_argument("--summary-output", type=Path, default=PROJECT_ROOT / "outputs/metrics/dominance_summary.json")
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    with args.per_case.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            label, reason = classify(case.get("text_score", 0.0), case.get("ecg_score", 0.0))
            counts[label] += 1
            rows.append({
                "id": case["id"],
                "question_type": case.get("question_type", ""),
                "attribute_type": case.get("attribute_type", ""),
                "text_score": case.get("text_score", 0.0),
                "ecg_score": case.get("ecg_score", 0.0),
                "textual_dominance": case.get("textual_dominance", 0.0),
                "dominance_class": label,
                "reason": reason,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    summary = {
        "count": total,
        "thresholds": DOMINANCE_THRESHOLDS,
        "class_counts": dict(counts),
        "class_fractions": {k: v / total for k, v in counts.items()} if total else {},
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
