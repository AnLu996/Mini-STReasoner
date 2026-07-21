"""Score ST-Bench predictions with the protocol of the original STReasoner paper.

``inference/evaluate_tasks.py`` compares normalised strings, which is stricter
than the reference implementation and therefore not comparable against the
numbers the paper reports. This module reproduces
``STReasoner/evaluation/evaluate_qa.py``: multiple-choice answers are read from
the ``<answer>...</answer>`` tag and reduced to a single A-D letter, while
forecasting answers are parsed as numeric series and scored with MAE/MAPE.

It also reports the answer distribution so a collapse towards one option is
visible, the same diagnostic used for the ECG-QA runs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_loader import iter_jsonl  # noqa: E402


MULTIPLE_CHOICE_TASKS = (
    "reasoning_correlation",
    "reasoning_entity",
    "reasoning_etiological",
)
FORECASTING_TASK = "reasoning_forecasting"

CHOICES = frozenset("ABCD")

# Accuracy of guessing uniformly among the four options.
RANDOM_BASELINE = 0.25


def extract_tag_content(text: str, tag: str = "answer") -> str:
    if not text:
        return ""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).replace("```", "").strip()
    return text.strip()


def normalize_choice(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if not value:
        return ""
    value = extract_tag_content(value, "answer")
    match = re.match(r"\s*([A-Da-d])[\.\)\s-]*", value)
    return match.group(1).upper() if match else value.lower()


def parse_series(text: Any) -> list[float]:
    if text is None:
        return []
    if isinstance(text, list):
        return [float(value) for value in text]
    if isinstance(text, (int, float)):
        return [float(text)]
    value = str(text).strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
        if isinstance(parsed, (int, float)):
            return [float(parsed)]
    return [float(number) for number in re.findall(r"-?\d+\.?\d*", value)]


def shannon_entropy(counts: Counter) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {"distinct": 0, "entropy_bits": 0.0, "normalized_entropy": 0.0}
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in counts.values() if count
    )
    maximum = math.log2(len(counts)) if len(counts) > 1 else 0.0
    return {
        "distinct": len(counts),
        "entropy_bits": entropy,
        "normalized_entropy": entropy / maximum if maximum else 0.0,
    }


def score_multiple_choice(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    correct = 0
    predicted_counts: Counter = Counter()
    target_counts: Counter = Counter()
    for row in rows:
        prediction = normalize_choice(row.get("prediction"))
        target = normalize_choice(row.get("answer"))
        predicted_counts[prediction or "<empty>"] += 1
        target_counts[target or "<empty>"] += 1
        correct += prediction == target
    total = len(rows)
    accuracy = correct / total if total else None
    parsed = sum(
        count for choice, count in predicted_counts.items() if choice in CHOICES
    )
    return {
        "task": task,
        "metric": "accuracy",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "random_baseline": RANDOM_BASELINE,
        "above_random": None if accuracy is None else accuracy - RANDOM_BASELINE,
        "parseable_choice_rate": parsed / total if total else 0.0,
        "prediction_distribution": dict(predicted_counts.most_common()),
        "target_distribution": dict(target_counts.most_common()),
        "diversity": shannon_entropy(predicted_counts),
    }


def score_forecasting(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    evaluated = 0
    mae_sum = 0.0
    mape_sum = 0.0
    mape_count = 0
    target_values: list[float] = []
    for row in rows:
        # The tag content must be isolated first: a chain-of-thought prefix also
        # contains digits, and parsing the raw text would score those instead.
        target = parse_series(extract_tag_content(str(row.get("answer", ""))))
        predicted = parse_series(extract_tag_content(str(row.get("prediction", ""))))
        if not target or not predicted:
            continue
        if len(predicted) < len(target):
            predicted = predicted + [predicted[-1]] * (len(target) - len(predicted))
        else:
            predicted = predicted[: len(target)]
        errors = [abs(predicted[i] - target[i]) for i in range(len(target))]
        mae_sum += sum(errors) / len(errors)
        percentage = [
            abs(predicted[i] - target[i]) / abs(target[i])
            for i in range(len(target))
            if abs(target[i]) > 1e-8
        ]
        if percentage:
            mape_sum += sum(percentage) / len(percentage) * 100
            mape_count += 1
        target_values.extend(target)
        evaluated += 1
    total = len(rows)
    return {
        "task": task,
        "metric": "mae",
        "total_samples": total,
        "evaluated_samples": evaluated,
        "coverage": evaluated / total if total else 0.0,
        "mae": mae_sum / evaluated if evaluated else None,
        "mape": mape_sum / mape_count if mape_count else None,
        "target_abs_mean": (
            sum(abs(value) for value in target_values) / len(target_values)
            if target_values
            else None
        ),
    }


def score_task(path: Path, task: str) -> dict[str, Any]:
    rows = list(iter_jsonl([path]))
    if task == FORECASTING_TASK:
        return score_forecasting(rows, task)
    return score_multiple_choice(rows, task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ST-Bench predictions (paper protocol)")
    parser.add_argument("--predictions-dir", type=Path, default=PROJECT_ROOT / "outputs/stbench_small")
    parser.add_argument(
        "--tasks", nargs="*", default=list(MULTIPLE_CHOICE_TASKS) + [FORECASTING_TASK]
    )
    parser.add_argument(
        "--prefix",
        default="predictions",
        help="file stem before the task name, e.g. baseline_predictions",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results: dict[str, Any] = {}
    for task in args.tasks:
        path = args.predictions_dir / f"{args.prefix}_{task}.jsonl"
        if path.exists():
            results[task] = score_task(path, task)
        else:
            print(f"[skip] missing {path}")

    accuracies = [
        result["accuracy"]
        for result in results.values()
        if result.get("metric") == "accuracy" and result.get("accuracy") is not None
    ]
    if accuracies:
        results["_summary"] = {
            "mean_accuracy_multiple_choice": sum(accuracies) / len(accuracies),
            "random_baseline": RANDOM_BASELINE,
            "tasks_above_random": sum(value > RANDOM_BASELINE for value in accuracies),
            "tasks_scored": len(accuracies),
        }

    output = args.output or args.predictions_dir / "stbench_scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved scores to {output}")


if __name__ == "__main__":
    main()
