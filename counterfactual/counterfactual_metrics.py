"""Counterfactual metrics for text-vs-ECG dominance.

Reads the per-variant predictions and produces, per case and in aggregate:

- ``QCFR``  Question/Text Counterfactual Flip Rate: how often a
  *meaning-preserving* question rewrite changes the answer. High QCFR means the
  model reacts to surface text form.
- ``ECFR``  ECG Counterfactual Flip Rate: average flip rate across the ECG
  signal perturbations. High ECFR means the model reacts to the signal.
- ``textual_dominance = QCFR - ECFR``.
- ``conflict_follow_question_rate`` / ``conflict_follow_ecg_rate``: on conflict
  probes (text claim contradicts the ECG), does the answer follow the text
  claim or stay with the evidence-based original answer?
- ``neutral_question_flip_rate``: flip rate when the clinical cue is removed.

A *flip* is a change of the normalized prediction relative to the ``original``
variant. Flips are measured against the model's own original prediction, not the
gold answer, so the metric is self-relative and independent of raw accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual import CONFLICT_VARIANTS, ECG_VARIANTS, NEUTRAL_VARIANT  # noqa: E402


def normalize(text: str) -> str:
    """Lowercase, whitespace-collapsed token form for flip comparison.

    Kept local (instead of importing ``inference.evaluate_tasks``) so the metric
    and downstream stages stay free of the torch dependency.
    """
    return " ".join(re.findall(r"\w+", str(text).lower(), flags=re.UNICODE))


def polarity(text: str) -> str | None:
    """Map an answer to clinical polarity: positive=abnormal, negative=normal."""
    norm = normalize(text)
    tokens = set(norm.split())
    positive = {"yes", "abnormal", "present", "elevation", "fibrillation", "block", "tachycardia"}
    negative = {"no", "none", "normal", "sinus"}
    has_pos = bool(tokens & positive)
    has_neg = bool(tokens & negative)
    if has_pos and not has_neg:
        return "positive"
    if has_neg and not has_pos:
        return "negative"
    return None


_CLAIM_POLARITY = {"normal": "negative", "abnormal": "positive"}


def per_case_metrics(row: dict[str, Any]) -> dict[str, Any]:
    predictions = row["predictions"]
    meta = row.get("variants_meta", {})
    base = normalize(predictions.get("original", ""))

    def flip(name: str) -> int:
        return int(normalize(predictions.get(name, "")) != base)

    qcfr = flip("question_cf")
    neutral_flip = flip(NEUTRAL_VARIANT)
    ecg_flips = {name: flip(name) for name in ECG_VARIANTS}
    ecg_flip_frac = sum(ecg_flips.values()) / max(len(ecg_flips), 1)

    # Conflict probes: only count those that actually contradict the ECG.
    conflict_obs: list[dict[str, Any]] = []
    for name, claim in CONFLICT_VARIANTS.items():
        variant_meta = meta.get(name, {}).get("meta", {})
        if not variant_meta.get("contradicts_ecg", True):
            continue
        pred_pol = polarity(predictions.get(name, ""))
        follow_question = int(pred_pol is not None and pred_pol == _CLAIM_POLARITY[claim])
        follow_ecg = int(normalize(predictions.get(name, "")) == base)
        conflict_obs.append({
            "variant": name,
            "claim": claim,
            "follow_question": follow_question,
            "follow_ecg": follow_ecg,
            "decidable": pred_pol is not None,
        })

    follow_q = [o["follow_question"] for o in conflict_obs]
    follow_e = [o["follow_ecg"] for o in conflict_obs]
    conflict_follow_question = sum(follow_q) / len(follow_q) if follow_q else None
    conflict_follow_ecg = sum(follow_e) / len(follow_e) if follow_e else None

    # Aggregate text sensitivity for dominance classification.
    text_signals = [qcfr, neutral_flip]
    if conflict_follow_question is not None:
        text_signals.append(conflict_follow_question)
    text_score = sum(text_signals) / len(text_signals)
    ecg_score = ecg_flip_frac

    return {
        "id": row["id"],
        "question_type": row.get("question_type", ""),
        "attribute_type": row.get("attribute_type", ""),
        "qcfr": qcfr,
        "ecfr": ecg_flip_frac,
        "ecg_flips": ecg_flips,
        "neutral_flip": neutral_flip,
        "conflict_follow_question_rate": conflict_follow_question,
        "conflict_follow_ecg_rate": conflict_follow_ecg,
        "conflict_observations": conflict_obs,
        "text_score": text_score,
        "ecg_score": ecg_score,
        "textual_dominance": qcfr - ecg_flip_frac,
    }


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    qcfr = _mean([c["qcfr"] for c in cases])
    ecfr = _mean([c["ecfr"] for c in cases])
    return {
        "count": len(cases),
        "qcfr": qcfr,
        "ecfr": ecfr,
        "textual_dominance": qcfr - ecfr,
        "conflict_follow_question_rate": _mean([c["conflict_follow_question_rate"] for c in cases]),
        "conflict_follow_ecg_rate": _mean([c["conflict_follow_ecg_rate"] for c in cases]),
        "neutral_question_flip_rate": _mean([c["neutral_flip"] for c in cases]),
    }


def grouped(cases: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        buckets[case.get(key) or "unknown"].append(case)
    return {name: aggregate(items) for name, items in sorted(buckets.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute counterfactual flip metrics")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "outputs/predictions/counterfactual_predictions.jsonl")
    parser.add_argument("--per-case-output", type=Path, default=PROJECT_ROOT / "outputs/metrics/per_case_metrics.jsonl")
    parser.add_argument("--summary-output", type=Path, default=PROJECT_ROOT / "outputs/metrics/counterfactual_metrics.json")
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(per_case_metrics(json.loads(line)))

    args.per_case_output.parent.mkdir(parents=True, exist_ok=True)
    with args.per_case_output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    summary = {
        "global": aggregate(cases),
        "by_question_type": grouped(cases, "question_type"),
        "by_attribute_type": grouped(cases, "attribute_type"),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["global"], indent=2))


if __name__ == "__main__":
    main()
