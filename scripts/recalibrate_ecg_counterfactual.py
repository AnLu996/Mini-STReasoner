"""Recompute ECFR and textual dominance choosing which ECG intervention counts.

``run_ecgqa_counterfactual_small.py`` reports ECFR as the mean flip rate over
four ECG counterfactuals (noise, lead mask, time mask, spike). Averaging them
hides that they are not comparable interventions: zero-mean Gaussian noise is
cancelled by the encoder's attention pooling, so it barely moves the
representation, while occluding a temporal window moves it three to four orders
of magnitude more (see section 9 of RESULTADOS_CORRIDA_A.txt). Including the
weak ones deflates ECFR and inflates the textual-dominance index by the same
amount.

This script re-derives the metrics per intervention and for chosen subsets, from
the predictions already stored by a finished run, so no inference is repeated.
The point is not to pick the flattering number but to report how much the
conclusion depends on the calibration of the intervention.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual.counterfactual_metrics import normalize  # noqa: E402

ALL_INTERVENTIONS = ("ecg_time_mask", "ecg_lead_mask", "ecg_noise", "ecg_spike")

# Interventions that survive the encoder's temporal pooling because they remove
# or displace structure instead of adding zero-mean noise.
STRUCTURED = ("ecg_time_mask", "ecg_lead_mask")

SUBSETS: dict[str, tuple[str, ...]] = {
    "todas (como en Corrida A)": ALL_INTERVENTIONS,
    "estructuradas": STRUCTURED,
    "solo oclusion temporal": ("ecg_time_mask",),
}


def flip(predictions: dict[str, str], name: str, base: str) -> int:
    return int(normalize(predictions.get(name, "")) != base)


def summarise(rows: list[dict[str, Any]], interventions: Iterable[str]) -> dict[str, Any]:
    interventions = tuple(interventions)
    qcfr_total = 0.0
    ecfr_total = 0.0
    for row in rows:
        predictions = row.get("predictions", {})
        base = normalize(predictions.get("original", ""))
        qcfr_total += flip(predictions, "question_cf", base)
        ecfr_total += sum(flip(predictions, name, base) for name in interventions) / len(
            interventions
        )
    count = max(len(rows), 1)
    qcfr = qcfr_total / count
    ecfr = ecfr_total / count
    return {
        "n": len(rows),
        "interventions": list(interventions),
        "QCFR": qcfr,
        "ECFR": ecfr,
        "textual_dominance": qcfr - ecfr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibrate ECFR by ECG intervention")
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "outputs/ecgqa_5k_control/counterfactual_results.jsonl",
    )
    parser.add_argument("--group-by", default="question_type")
    parser.add_argument("--min-group", type=int, default=10, help="skip smaller groups")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]

    per_intervention = {
        name: summarise(rows, (name,)) for name in ALL_INTERVENTIONS
    }
    per_subset = {label: summarise(rows, names) for label, names in SUBSETS.items()}

    print(f"=== Tasa de cambio por intervencion (n={len(rows)}) ===")
    print(f"{'intervencion':24s} {'ECFR':>8s}")
    for name, values in sorted(
        per_intervention.items(), key=lambda item: -item[1]["ECFR"]
    ):
        print(f"{name:24s} {values['ECFR']:8.4f}")

    print(f"\n=== QCFR, ECFR y dominancia segun el conjunto elegido ===")
    print(f"{'conjunto':28s} {'QCFR':>8s} {'ECFR':>8s} {'D_texto':>9s}")
    for label, values in per_subset.items():
        print(
            f"{label:28s} {values['QCFR']:8.4f} {values['ECFR']:8.4f} "
            f"{values['textual_dominance']:9.4f}"
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get(args.group_by) or "unknown"].append(row)
    by_group = {
        name: {label: summarise(items, names) for label, names in SUBSETS.items()}
        for name, items in groups.items()
        if len(items) >= args.min_group
    }

    print(f"\n=== D_texto por {args.group_by} (grupos con n>={args.min_group}) ===")
    header = f"{'grupo':32s} {'n':>4s}" + "".join(f"{label[:16]:>18s}" for label in SUBSETS)
    print(header)
    for name, subsets in sorted(
        by_group.items(), key=lambda item: -item[1]["solo oclusion temporal"]["textual_dominance"]
    ):
        row = f"{name:32s} {subsets['todas (como en Corrida A)']['n']:4d}"
        row += "".join(f"{subsets[label]['textual_dominance']:18.4f}" for label in SUBSETS)
        print(row)

    results = {
        "source": str(args.results),
        "per_intervention": per_intervention,
        "per_subset": per_subset,
        f"by_{args.group_by}": by_group,
    }
    output = args.output or args.results.with_name("counterfactual_recalibrated.json")
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
