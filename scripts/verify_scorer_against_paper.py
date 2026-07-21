"""Check ``scripts/score_stbench.py`` against the numbers published by STReasoner.

The original repository ships both the raw generations of STReasoner-8B and the
metrics computed from them. Re-scoring those generations with our own scorer and
matching the published values is what makes any later comparison meaningful: it
shows a gap against the paper comes from the scaled-down model, not from a
different evaluation protocol.

Usage::

    python scripts/verify_scorer_against_paper.py \
        --reference-dir ../Tesis/STReasoner/exp_STReasoner-8B \
        --dataset-dir data/stbench_small/raw/ST-Test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.score_stbench import score_task  # noqa: E402


# Metrics reported in exp_STReasoner-8B/*/evaluation_metrics.json.
PUBLISHED = {
    "reasoning_correlation": ("accuracy", 0.8712311557788944, "correlation_test.jsonl"),
    "reasoning_entity": ("accuracy", 0.7571189279731994, "entity_test.jsonl"),
    "reasoning_etiological": ("accuracy", 0.9565217391304348, "etiological_test.jsonl"),
    "reasoning_forecasting": ("mae", 65.5934736377407, "forecasting_test.jsonl"),
}

TOLERANCE = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the ST-Bench scorer against the paper")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "Tesis/STReasoner/exp_STReasoner-8B",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "data/stbench_small/raw/ST-Test"
    )
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp())
    failures = 0
    print(f"{'task':30s} {'ours':>16s} {'published':>16s}  match")
    for task, (metric, expected, dataset_name) in PUBLISHED.items():
        generations = json.loads(
            (args.reference_dir / f"{task}-STReasoner-8B" / "generated_answer.json").read_text()
        )
        predictions = {entry["idx"]: entry.get("response", "") for entry in generations}
        dataset = [
            json.loads(line)
            for line in (args.dataset_dir / dataset_name).read_text().splitlines()
            if line.strip()
        ]

        rebuilt = workspace / f"predictions_{task}.jsonl"
        with rebuilt.open("w", encoding="utf-8") as handle:
            for index, sample in enumerate(dataset):
                row = {"prediction": predictions.get(index, ""), "answer": sample["output"]}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        observed = score_task(rebuilt, task)[metric]
        matches = observed is not None and abs(observed - expected) < TOLERANCE
        failures += not matches
        print(f"{task:30s} {observed:16.9f} {expected:16.9f}  {'ok' if matches else 'MISMATCH'}")

    if failures:
        raise SystemExit(f"{failures} metric(s) did not reproduce the published values")
    print("\nAll published metrics reproduced.")


if __name__ == "__main__":
    main()
