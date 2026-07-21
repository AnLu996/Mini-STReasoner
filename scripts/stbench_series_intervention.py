"""Ask whether the *content* of the time series changes the answer on ST-Bench.

The ``no_series`` ablation removes the temporal tokens altogether. On this
checkpoint that also removes the cue that keeps the model in the fine-tuned
answer format: it falls back to the base model's ``<think>`` behaviour and never
emits an ``<answer>`` tag, so its accuracy measures a formatting collapse rather
than the value of the modality.

These interventions keep the four temporal tokens in place and only change what
they encode, so the prompt structure the model was trained on is preserved:

    original  the sample's own series
    swapped   another sample's series, same question
    zeroed    an all-zero series, same question

If accuracy survives ``swapped`` and ``zeroed``, the answer does not depend on
the temporal evidence — the textual-dominance reading — and the drop under
``no_series`` was an artefact of the intervention, not evidence of grounding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.runtime import load_checkpoint, predict  # noqa: E402
from scripts.score_stbench import extract_tag_content, normalize_choice  # noqa: E402
from training.dataset_loader import iter_jsonl  # noqa: E402


CONDITIONS = ("original", "swapped", "zeroed")

# Relative tolerance when both sides of a comparison parse as numbers, so that
# "0.2618" and "0.26180" count as the same alignment answer.
NUMERIC_TOLERANCE = 0.01


def _as_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def match_text(prediction: Any, target: Any) -> bool:
    """Compare free-form answers, as used by the alignment task."""
    left = extract_tag_content(str(prediction or "")).strip().lower()
    right = extract_tag_content(str(target or "")).strip().lower()
    if left == right:
        return True
    left_number, right_number = _as_number(left), _as_number(right)
    if left_number is None or right_number is None:
        return False
    scale = max(abs(right_number), 1e-8)
    return abs(left_number - right_number) / scale <= NUMERIC_TOLERANCE


MATCHERS = {
    "choice": lambda prediction, target: normalize_choice(prediction) == normalize_choice(target),
    "text": match_text,
}


def zeroed_series(series: list[list[float]]) -> list[list[float]]:
    return [[0.0] * len(row) for row in series]


def build_variants(examples: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    variants = []
    for index, example in enumerate(examples):
        donor = examples[(index + 1) % len(examples)]
        variants.append(
            {
                "original": example,
                "swapped": {**example, "time_series": donor["time_series"]},
                "zeroed": {**example, "time_series": zeroed_series(example["time_series"])},
            }
        )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Series-content interventions on ST-Bench")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/stbench_small/test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--match",
        choices=sorted(MATCHERS),
        default="choice",
        help="'choice' for the A-D reasoning tasks, 'text' for alignment answers",
    )
    args = parser.parse_args()
    matches = MATCHERS[args.match]

    examples = list(iter_jsonl([args.data_dir / f"{args.task}.jsonl"]))
    if args.limit:
        examples = examples[: args.limit]
    if len(examples) < 2:
        raise SystemExit("need at least two samples so a series can be swapped in")

    tokenizer, model, config = load_checkpoint(args.model_path)
    output = args.output or PROJECT_ROOT / "outputs/stbench_small" / f"series_intervention_{args.task}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    hits = {condition: 0 for condition in CONDITIONS}
    flips = {condition: 0 for condition in CONDITIONS if condition != "original"}
    with output.open("w", encoding="utf-8") as handle:
        for index, variants in enumerate(build_variants(examples)):
            gold = examples[index].get("answer")
            predictions = {}
            for condition in CONDITIONS:
                text, _ = predict(
                    tokenizer, model, config, variants[condition], "full", args.max_new_tokens
                )
                predictions[condition] = text
                hits[condition] += matches(text, gold)
            for condition in flips:
                flips[condition] += not matches(
                    predictions[condition], predictions["original"]
                )
            handle.write(
                json.dumps(
                    {
                        "task": args.task,
                        "answer": examples[index].get("answer", ""),
                        "predictions": predictions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"[{index + 1}/{len(examples)}] " + " ".join(
                f"{condition}={predictions[condition][:20]!r}" for condition in CONDITIONS
            ), flush=True)

    total = len(examples)
    summary = {
        "task": args.task,
        "n": total,
        "accuracy": {condition: hits[condition] / total for condition in CONDITIONS},
        # Share of samples whose answer changes when only the series content does.
        "flip_rate": {condition: flips[condition] / total for condition in flips},
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output} and {summary_path}")


if __name__ == "__main__":
    main()
