from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.evaluate_tasks import token_f1  # noqa: E402
from training.dataset_loader import iter_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute text-vs-series dominance scores")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "outputs/xai/modal_ablation_results.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/xai/dominance_metrics.json")
    args = parser.parse_args()
    rows = []
    for row in iter_jsonl([args.input]):
        answer = row["answer"]
        predictions = row["predictions"]
        full = token_f1(predictions["full"], answer)
        no_text = token_f1(predictions["no_text"], answer)
        no_series = token_f1(predictions["no_series"], answer)
        text_impact = full - no_text
        series_impact = full - no_series
        rows.append({"task": row.get("task"), "text_impact": text_impact, "series_impact": series_impact, "text_dominance_score": text_impact - series_impact, "metadata": row.get("metadata", {})})
    mean = sum(item["text_dominance_score"] for item in rows) / max(len(rows), 1)
    payload = {"count": len(rows), "mean_text_dominance_score": mean, "examples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"count": len(rows), "mean_text_dominance_score": mean}, indent=2))


if __name__ == "__main__":
    main()
