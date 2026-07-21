"""Confidence intervals for the ST-Bench validation run.

The validation uses a 60-sample-per-task test subset, so point estimates alone
invite over-reading: a gap of a few points against the published numbers is well
inside sampling noise. This reports Wilson intervals for each accuracy and
bootstrap intervals for the modal contributions, which are differences between
two accuracies measured on the *same* samples and therefore have to be
resampled jointly rather than compared interval-to-interval.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.score_stbench import (  # noqa: E402
    MULTIPLE_CHOICE_TASKS,
    RANDOM_BASELINE,
    normalize_choice,
)
from training.dataset_loader import iter_jsonl  # noqa: E402


# Published accuracies of STReasoner-8B, for reference in the same table.
PUBLISHED_ACCURACY = {
    "reasoning_correlation": 0.8712311557788944,
    "reasoning_entity": 0.7571189279731994,
    "reasoning_etiological": 0.9565217391304348,
}


def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    if not total:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * ((proportion * (1 - proportion) / total + z**2 / (4 * total**2)) ** 0.5)
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_interval(
    samples: list[Any], statistic, resamples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(samples)
    if not size:
        return (float("nan"), float("nan"))
    estimates = []
    for _ in range(resamples):
        draw = [samples[rng.randrange(size)] for _ in range(size)]
        estimates.append(statistic(draw))
    return (percentile(estimates, 0.025), percentile(estimates, 0.975))


def accuracy_report(path: Path, task: str) -> dict[str, Any]:
    rows = list(iter_jsonl([path]))
    hits = [
        normalize_choice(row.get("prediction")) == normalize_choice(row.get("answer"))
        for row in rows
    ]
    total = len(hits)
    successes = sum(hits)
    low, high = wilson_interval(successes, total)
    published = PUBLISHED_ACCURACY.get(task)
    return {
        "task": task,
        "n": total,
        "accuracy": successes / total if total else None,
        "wilson_95": [low, high],
        "random_baseline": RANDOM_BASELINE,
        "above_random": low > RANDOM_BASELINE,
        "published_8b": published,
        "published_within_interval": (
            None if published is None else low <= published <= high
        ),
    }


def ablation_report(path: Path, task: str, resamples: int, seed: int) -> dict[str, Any]:
    rows = list(iter_jsonl([path]))
    paired = [
        {
            condition: normalize_choice(row.get("predictions", {}).get(condition))
            == normalize_choice(row.get("answer"))
            for condition in ("full", "no_text", "no_series", "conflict_text")
        }
        for row in rows
    ]

    def mean(draw: list[dict[str, bool]], condition: str) -> float:
        return sum(item[condition] for item in draw) / len(draw) if draw else 0.0

    statistics = {
        "text_contribution": lambda draw: mean(draw, "full") - mean(draw, "no_text"),
        "series_contribution": lambda draw: mean(draw, "full") - mean(draw, "no_series"),
        "textual_dominance": lambda draw: (mean(draw, "full") - mean(draw, "no_text"))
        - (mean(draw, "full") - mean(draw, "no_series")),
    }
    report: dict[str, Any] = {"task": task, "n": len(paired)}
    for condition in ("full", "no_text", "no_series", "conflict_text"):
        report[f"accuracy_{condition}"] = mean(paired, condition)
    for name, statistic in statistics.items():
        low, high = bootstrap_interval(paired, statistic, resamples, seed)
        report[name] = statistic(paired)
        report[f"{name}_95"] = [low, high]
        # An interval straddling zero means the run cannot tell the modality
        # apart from no contribution at all.
        report[f"{name}_significant"] = low > 0 or high < 0
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence intervals for ST-Bench validation")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "outputs/stbench_small")
    parser.add_argument("--tasks", nargs="*", default=list(MULTIPLE_CHOICE_TASKS))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results: dict[str, Any] = {"accuracy": {}, "ablation": {}}
    for task in args.tasks:
        predictions = args.results_dir / f"predictions_{task}.jsonl"
        if predictions.exists():
            results["accuracy"][task] = accuracy_report(predictions, task)
        ablation = args.results_dir / f"ablation_{task}.jsonl"
        if ablation.exists():
            results["ablation"][task] = ablation_report(
                ablation, task, args.resamples, args.seed
            )

    output = args.output or args.results_dir / "stbench_confidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved confidence intervals to {output}")


if __name__ == "__main__":
    main()
