"""Confidence intervals and hypothesis tests for the ECG-QA runs.

Every headline number of Corridas A and B is a point estimate over a few hundred
samples, and several conclusions hang on comparisons against a threshold or
between two runs. Two of them cannot be settled by eye:

* ``cont_ECG = 0.0364`` against the operational threshold of 0.05 (n=300);
* ``QCFR`` against ``ECFR``, which is what the textual-dominance index subtracts.

The contributions are differences between conditions measured on the *same*
samples, so the resampling is paired: each bootstrap draw takes whole samples
and recomputes both terms on that draw. The same applies when comparing two
runs, because both were evaluated on the identical sample set in the same order
(verified before use).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual.counterfactual_metrics import normalize  # noqa: E402
from scripts.stbench_confidence import (  # noqa: E402
    bootstrap_interval,
    percentile,
    wilson_interval,
)

CONDITIONS = ("full", "no_text", "no_series", "conflict_text")
ECG_INTERVENTIONS = ("ecg_time_mask", "ecg_lead_mask", "ecg_noise", "ecg_spike")

# Threshold the improvement plan set for claiming the signal is actually in play.
CONT_ECG_THRESHOLD = 0.05


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def estimate(
    rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float],
    resamples: int, seed: int,
) -> dict[str, Any]:
    low, high = bootstrap_interval(rows, statistic, resamples, seed)
    return {"value": statistic(rows), "ci95": [low, high], "n": len(rows)}


def mean_of(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / len(values) if values else 0.0


def condition_f1(rows: list[dict[str, Any]], condition: str) -> float:
    values = [float(row["configs"][condition]["token_f1"]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def binomial_two_sided(successes: int, total: int, probability: float = 0.5) -> float:
    from scipy.stats import binomtest

    return float(binomtest(successes, total, probability).pvalue)


def paired_permutation(
    rows: list[dict[str, Any]],
    left: Callable[[dict[str, Any]], float],
    right: Callable[[dict[str, Any]], float],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Test whether two per-sample measurements differ, swapping labels within samples.

    Used for QCFR against ECFR: both are measured on the same case, so the null
    is that the two interventions are exchangeable for that case.
    """
    rng = random.Random(seed)
    pairs = [(left(row), right(row)) for row in rows]
    observed = sum(a - b for a, b in pairs) / len(pairs)
    extreme = 0
    for _ in range(resamples):
        total = 0.0
        for a, b in pairs:
            total += (a - b) if rng.random() < 0.5 else (b - a)
        if abs(total / len(pairs)) >= abs(observed):
            extreme += 1
    return {
        "difference": observed,
        "p_value": (extreme + 1) / (resamples + 1),
        "resamples": resamples,
    }


def evaluation_report(path: Path, resamples: int, seed: int) -> dict[str, Any]:
    rows = read(path)
    report = {
        "exact_match": estimate(rows, lambda r: mean_of(r, "exact_match"), resamples, seed),
        "token_f1": estimate(rows, lambda r: mean_of(r, "token_f1"), resamples, seed),
    }
    yesno = [row for row in rows if str(row.get("is_yesno")).lower() == "true"]
    if yesno:
        correct = sum(round(float(row["yesno_correct"])) for row in yesno)
        low, high = wilson_interval(correct, len(yesno))
        report["yesno"] = {
            "n": len(yesno),
            "accuracy": correct / len(yesno),
            "wilson_95": [low, high],
            "p_value_vs_chance": binomial_two_sided(correct, len(yesno)),
            # Beating chance is the minimum bar for a binary clinical question.
            "above_chance": low > 0.5,
        }
    return report


def ablation_report(path: Path, resamples: int, seed: int) -> dict[str, Any]:
    rows = read(path)
    report: dict[str, Any] = {
        "per_condition": {
            condition: estimate(
                rows, lambda r, c=condition: condition_f1(r, c), resamples, seed
            )
            for condition in CONDITIONS
        }
    }
    contributions = {
        "text_contribution": lambda r: condition_f1(r, "full") - condition_f1(r, "no_text"),
        "ecg_contribution": lambda r: condition_f1(r, "full") - condition_f1(r, "no_series"),
        "textual_dominance": lambda r: (condition_f1(r, "full") - condition_f1(r, "no_text"))
        - (condition_f1(r, "full") - condition_f1(r, "no_series")),
    }
    for name, statistic in contributions.items():
        report[name] = estimate(rows, statistic, resamples, seed)
        low, high = report[name]["ci95"]
        report[name]["significant"] = low > 0 or high < 0

    ecg = report["ecg_contribution"]
    ecg["threshold"] = CONT_ECG_THRESHOLD
    ecg["threshold_inside_ci"] = ecg["ci95"][0] <= CONT_ECG_THRESHOLD <= ecg["ci95"][1]
    # Only a CI entirely below the threshold licenses saying it was not reached.
    ecg["below_threshold_significantly"] = ecg["ci95"][1] < CONT_ECG_THRESHOLD
    return report


def counterfactual_report(
    path: Path, interventions: tuple[str, ...], resamples: int, seed: int
) -> dict[str, Any]:
    rows = read(path)

    def flip(row: dict[str, Any], name: str) -> float:
        base = normalize(row["predictions"].get("original", ""))
        return float(normalize(row["predictions"].get(name, "")) != base)

    def qcfr(sample: list[dict[str, Any]]) -> float:
        return sum(flip(row, "question_cf") for row in sample) / len(sample)

    def ecfr(sample: list[dict[str, Any]]) -> float:
        return sum(
            sum(flip(row, name) for name in interventions) / len(interventions)
            for row in sample
        ) / len(sample)

    report = {
        "interventions": list(interventions),
        "QCFR": estimate(rows, qcfr, resamples, seed),
        "ECFR": estimate(rows, ecfr, resamples, seed),
        "textual_dominance": estimate(rows, lambda r: qcfr(r) - ecfr(r), resamples, seed),
        "per_intervention": {
            name: estimate(
                rows, lambda r, n=name: sum(flip(row, n) for row in r) / len(r), resamples, seed
            )
            for name in ECG_INTERVENTIONS
        },
        "permutation_qcfr_vs_ecfr": paired_permutation(
            rows,
            lambda row: flip(row, "question_cf"),
            lambda row: sum(flip(row, name) for name in interventions) / len(interventions),
            resamples,
            seed,
        ),
    }
    low, high = report["textual_dominance"]["ci95"]
    report["textual_dominance"]["significant"] = low > 0 or high < 0
    return report


def paired_run_difference(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the B-minus-A difference resampling sample *indices* jointly."""
    if len(rows_a) != len(rows_b):
        raise ValueError("runs must be evaluated on the same samples")
    rng = random.Random(seed)
    size = len(rows_a)
    observed = statistic(rows_b) - statistic(rows_a)
    estimates = []
    for _ in range(resamples):
        index = [rng.randrange(size) for _ in range(size)]
        estimates.append(
            statistic([rows_b[i] for i in index]) - statistic([rows_a[i] for i in index])
        )
    low, high = percentile(estimates, 0.025), percentile(estimates, 0.975)
    return {
        "difference": observed,
        "ci95": [low, high],
        "significant": low > 0 or high < 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence intervals for the ECG-QA runs")
    parser.add_argument("--run-a", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_5k_control")
    parser.add_argument("--run-b", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_5k_encoder_b")
    parser.add_argument(
        "--ecg-interventions",
        nargs="*",
        default=["ecg_time_mask"],
        help="which ECG counterfactuals make up ECFR; the default is the calibrated one",
    )
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    interventions = tuple(args.ecg_interventions)
    results: dict[str, Any] = {"resamples": args.resamples, "seed": args.seed}
    for label, root in (("A", args.run_a), ("B", args.run_b)):
        results[label] = {
            "evaluation": evaluation_report(root / "evaluation.jsonl", args.resamples, args.seed),
            "ablation": ablation_report(root / "ablation.jsonl", args.resamples, args.seed),
            "counterfactual": counterfactual_report(
                root / "counterfactual_results.jsonl", interventions, args.resamples, args.seed
            ),
        }

    ablation_a = read(args.run_a / "ablation.jsonl")
    ablation_b = read(args.run_b / "ablation.jsonl")
    results["B_minus_A"] = {
        name: paired_run_difference(ablation_a, ablation_b, statistic, args.resamples, args.seed)
        for name, statistic in (
            ("ecg_contribution", lambda r: condition_f1(r, "full") - condition_f1(r, "no_series")),
            ("text_contribution", lambda r: condition_f1(r, "full") - condition_f1(r, "no_text")),
            ("full_token_f1", lambda r: condition_f1(r, "full")),
        )
    }

    output = args.output or PROJECT_ROOT / "outputs/ecgqa_confidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    def show(value: dict[str, Any]) -> str:
        return f"{value['value']:7.4f}  [{value['ci95'][0]:7.4f}, {value['ci95'][1]:7.4f}]"

    print(f"=== DESEMPENO (bootstrap {args.resamples}) ===")
    print(f"{'metrica':20s} {'Corrida A':>28s} {'Corrida B':>28s}")
    for key in ("exact_match", "token_f1"):
        print(f"{key:20s} {show(results['A']['evaluation'][key]):>28s} {show(results['B']['evaluation'][key]):>28s}")
    for label in ("A", "B"):
        yn = results[label]["evaluation"].get("yesno")
        if yn:
            print(
                f"  yes/no {label}: {yn['accuracy']:.4f} "
                f"[{yn['wilson_95'][0]:.4f}, {yn['wilson_95'][1]:.4f}] "
                f"n={yn['n']}  p vs azar = {yn['p_value_vs_chance']:.4f}  "
                f"{'supera el azar' if yn['above_chance'] else 'NO supera el azar'}"
            )

    print(f"\n=== ABLACION MODAL ===")
    print(f"{'metrica':20s} {'Corrida A':>28s} {'Corrida B':>28s}")
    for key in ("text_contribution", "ecg_contribution", "textual_dominance"):
        print(f"{key:20s} {show(results['A']['ablation'][key]):>28s} {show(results['B']['ablation'][key]):>28s}")
    for label in ("A", "B"):
        ecg = results[label]["ablation"]["ecg_contribution"]
        verdict = (
            "por debajo del umbral con significancia"
            if ecg["below_threshold_significantly"]
            else "NO se distingue del umbral"
            if ecg["threshold_inside_ci"]
            else "por encima del umbral"
        )
        print(f"  cont_ECG {label} frente a {CONT_ECG_THRESHOLD}: {verdict}")

    print(f"\n=== CONTRAFACTUAL (ECFR = {', '.join(interventions)}) ===")
    print(f"{'metrica':20s} {'Corrida A':>28s} {'Corrida B':>28s}")
    for key in ("QCFR", "ECFR", "textual_dominance"):
        print(f"{key:20s} {show(results['A']['counterfactual'][key]):>28s} {show(results['B']['counterfactual'][key]):>28s}")
    for label in ("A", "B"):
        perm = results[label]["counterfactual"]["permutation_qcfr_vs_ecfr"]
        print(f"  permutacion QCFR vs ECFR {label}: diferencia {perm['difference']:.4f}  p = {perm['p_value']:.4f}")

    print(f"\n=== DIFERENCIA B - A (bootstrap pareado) ===")
    for name, value in results["B_minus_A"].items():
        mark = "significativa" if value["significant"] else "no significativa"
        print(
            f"{name:20s} {value['difference']:+7.4f}  "
            f"[{value['ci95'][0]:+7.4f}, {value['ci95'][1]:+7.4f}]  {mark}"
        )

    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
