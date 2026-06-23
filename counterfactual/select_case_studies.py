"""Select representative case studies for a future visualizer.

Picks the clearest examples of each dominance class and writes them to
``outputs/reports/selected_cases.jsonl``. Each line carries everything a future
viewer needs: the original sample, every counterfactual (its edit, prediction
and whether it flipped) and a short reason.

Schema per line::

    {
      "id": "...",
      "dominance_class": "...",
      "original": {...},
      "counterfactuals": {...},
      "reason": "..."
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual import DOMINANCE_CLASSES  # noqa: E402
from counterfactual.counterfactual_metrics import normalize  # noqa: E402


def _index_jsonl(path: Path, key: str = "id") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row[key]] = row
    return out


def clarity(case: dict[str, Any]) -> float:
    """How clear-cut the dominance signal is (used to rank within a class)."""
    label = case["dominance_class"]
    text, ecg = case.get("text_score", 0.0), case.get("ecg_score", 0.0)
    if label == "TEXT_DOMINANT":
        return text - ecg
    if label == "ECG_DOMINANT":
        return ecg - text
    if label == "UNSTABLE":
        return min(text, ecg)
    if label == "UNCLEAR":
        return -(text + ecg)
    return -abs(text - ecg)  # BALANCED: smallest gap is clearest


def build_counterfactuals(
    cf_record: dict[str, Any],
    pred_record: dict[str, Any],
) -> dict[str, Any]:
    predictions = pred_record.get("predictions", {})
    base = normalize(predictions.get("original", ""))
    variants = cf_record.get("variants", {})
    out: dict[str, Any] = {}
    for name, variant in variants.items():
        if name == "original":
            continue
        prediction = predictions.get(name, "")
        entry: dict[str, Any] = {
            "type": variant.get("type"),
            "prediction": prediction,
            "flipped": normalize(prediction) != base,
        }
        if variant.get("type") == "ecg":
            entry["transform"] = variant.get("transform")
        else:
            entry["question"] = variant.get("question")
            if "claim_polarity" in variant:
                entry["claim_polarity"] = variant["claim_polarity"]
        out[name] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative counterfactual cases")
    parser.add_argument("--dominance", type=Path, default=PROJECT_ROOT / "outputs/metrics/dominance.jsonl")
    parser.add_argument("--counterfactuals", type=Path, default=PROJECT_ROOT / "outputs/counterfactuals/counterfactuals.jsonl")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "outputs/predictions/counterfactual_predictions.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/reports/selected_cases.jsonl")
    parser.add_argument("--per-class", type=int, default=3, help="Cases to keep per dominance class")
    args = parser.parse_args()

    dominance = list(_index_jsonl(args.dominance).values())
    cf_index = _index_jsonl(args.counterfactuals)
    pred_index = _index_jsonl(args.predictions)

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in dominance:
        by_class[case["dominance_class"]].append(case)

    selected: list[dict[str, Any]] = []
    for label in DOMINANCE_CLASSES:
        ranked = sorted(by_class.get(label, []), key=clarity, reverse=True)
        for case in ranked[: args.per_class]:
            cf_record = cf_index.get(case["id"], {})
            pred_record = pred_index.get(case["id"], {})
            predictions = pred_record.get("predictions", {})
            selected.append({
                "id": case["id"],
                "dominance_class": label,
                "original": {
                    "question": cf_record.get("original_question", ""),
                    "answer": cf_record.get("answer", ""),
                    "question_type": case.get("question_type", ""),
                    "attribute_type": case.get("attribute_type", ""),
                    "ecg_abnormal_hint": cf_record.get("ecg_abnormal_hint"),
                    "prediction": predictions.get("original", ""),
                },
                "counterfactuals": build_counterfactuals(cf_record, pred_record),
                "scores": {
                    "text_score": case.get("text_score", 0.0),
                    "ecg_score": case.get("ecg_score", 0.0),
                    "textual_dominance": case.get("textual_dominance", 0.0),
                },
                "reason": case.get("reason", ""),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in selected:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({"selected": len(selected), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
