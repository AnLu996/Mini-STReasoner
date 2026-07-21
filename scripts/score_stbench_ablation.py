"""Score the modal ablation of ST-Bench with the paper's answer protocol.

``xai/modal_ablation.py`` regenerates each sample under four conditions
(``full``, ``no_text``, ``no_series``, ``conflict_text``). This scorer turns
those generations into the modal-contribution numbers already used for the
ECG-QA runs::

    text_contribution   = score(full) - score(no_text)
    series_contribution = score(full) - score(no_series)
    textual_dominance   = text_contribution - series_contribution

Dominance is only reported for the multiple-choice tasks, where the score is an
accuracy and the three quantities share a scale. Forecasting is scored with MAE
(lower is better), so its per-condition values are reported without deriving a
dominance figure from them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.score_stbench import (  # noqa: E402
    FORECASTING_TASK,
    MULTIPLE_CHOICE_TASKS,
    score_forecasting,
    score_multiple_choice,
)
from training.dataset_loader import iter_jsonl  # noqa: E402


CONDITIONS = ("full", "no_text", "no_series", "conflict_text")


def score_condition(rows: list[dict[str, Any]], task: str, condition: str) -> dict[str, Any]:
    flattened = [
        {"prediction": row.get("predictions", {}).get(condition, ""), "answer": row.get("answer", "")}
        for row in rows
    ]
    if task == FORECASTING_TASK:
        return score_forecasting(flattened, task)
    return score_multiple_choice(flattened, task)


def score_ablation_file(path: Path, task: str) -> dict[str, Any]:
    rows = list(iter_jsonl([path]))
    metric = "mae" if task == FORECASTING_TASK else "accuracy"
    # A run may regenerate only some conditions to afford a larger token budget.
    present = [
        condition
        for condition in CONDITIONS
        if any(condition in row.get("predictions", {}) for row in rows)
    ]
    by_condition = {
        condition: score_condition(rows, task, condition) for condition in present
    }
    result: dict[str, Any] = {
        "task": task,
        "metric": metric,
        "samples": len(rows),
        "by_condition": {
            condition: {
                metric: scores.get(metric),
                **(
                    {"parseable_choice_rate": scores.get("parseable_choice_rate")}
                    if metric == "accuracy"
                    else {"coverage": scores.get("coverage")}
                ),
            }
            for condition, scores in by_condition.items()
        },
    }

    if metric == "accuracy":
        full = by_condition.get("full", {}).get("accuracy")
        no_text = by_condition.get("no_text", {}).get("accuracy")
        no_series = by_condition.get("no_series", {}).get("accuracy")
        if full is not None and no_text is not None:
            result["text_contribution"] = full - no_text
        if full is not None and no_series is not None:
            result["series_contribution"] = full - no_series
        if "text_contribution" in result and "series_contribution" in result:
            result["textual_dominance"] = (
                result["text_contribution"] - result["series_contribution"]
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ST-Bench modal ablation")
    parser.add_argument(
        "--ablation-dir", type=Path, default=PROJECT_ROOT / "outputs/stbench_small"
    )
    parser.add_argument(
        "--tasks", nargs="*", default=list(MULTIPLE_CHOICE_TASKS) + [FORECASTING_TASK]
    )
    parser.add_argument("--prefix", default="ablation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results: dict[str, Any] = {}
    for task in args.tasks:
        path = args.ablation_dir / f"{args.prefix}_{task}.jsonl"
        if path.exists():
            results[task] = score_ablation_file(path, task)
        else:
            print(f"[skip] missing {path}")

    dominances = [
        result["textual_dominance"]
        for result in results.values()
        if "textual_dominance" in result
    ]
    if dominances:
        results["_summary"] = {
            "mean_textual_dominance_multiple_choice": sum(dominances) / len(dominances),
            "tasks_scored": len(dominances),
        }

    output = args.output or args.ablation_dir / "stbench_ablation_scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved ablation scores to {output}")


if __name__ == "__main__":
    main()
