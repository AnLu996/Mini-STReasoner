"""Aggregate per-case metrics into the final summary table and report.

Produces:
- ``outputs/tables/counterfactual_summary.csv`` with one row per
  ``(question_type, attribute_type)`` group plus an overall ``ALL`` row::

      question_type,attribute_type,qcfr,ecfr,textual_dominance,
      conflict_follow_question_rate,conflict_follow_ecg_rate,neutral_question_flip_rate

- ``outputs/reports/counterfactual_report.md`` summarising the global metrics
  and the dominance-class distribution for quick human reading.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual.counterfactual_metrics import aggregate  # noqa: E402

CSV_FIELDS = [
    "question_type",
    "attribute_type",
    "qcfr",
    "ecfr",
    "textual_dominance",
    "conflict_follow_question_rate",
    "conflict_follow_ecg_rate",
    "neutral_question_flip_rate",
]


def _row(question_type: str, attribute_type: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_type": question_type,
        "attribute_type": attribute_type,
        "qcfr": round(stats["qcfr"], 4),
        "ecfr": round(stats["ecfr"], 4),
        "textual_dominance": round(stats["textual_dominance"], 4),
        "conflict_follow_question_rate": round(stats["conflict_follow_question_rate"], 4),
        "conflict_follow_ecg_rate": round(stats["conflict_follow_ecg_rate"], 4),
        "neutral_question_flip_rate": round(stats["neutral_question_flip_rate"], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate counterfactual results into a table")
    parser.add_argument("--per-case", type=Path, default=PROJECT_ROOT / "outputs/metrics/per_case_metrics.jsonl")
    parser.add_argument("--dominance-summary", type=Path, default=PROJECT_ROOT / "outputs/metrics/dominance_summary.json")
    parser.add_argument("--table-output", type=Path, default=PROJECT_ROOT / "outputs/tables/counterfactual_summary.csv")
    parser.add_argument("--report-output", type=Path, default=PROJECT_ROOT / "outputs/reports/counterfactual_report.md")
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with args.per_case.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        key = (case.get("question_type") or "unknown", case.get("attribute_type") or "unknown")
        groups[key].append(case)

    overall = aggregate(cases)
    rows = [_row("ALL", "ALL", overall)]
    for (question_type, attribute_type), items in sorted(groups.items()):
        rows.append(_row(question_type, attribute_type, aggregate(items)))

    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    with args.table_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    dominance = {}
    if args.dominance_summary.exists():
        dominance = json.loads(args.dominance_summary.read_text())

    lines = [
        "# Counterfactual dominance report",
        "",
        f"Cases analysed: **{overall['count']}**",
        "",
        "## Global metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| QCFR (text flip rate) | {overall['qcfr']:.4f} |",
        f"| ECFR (ECG flip rate) | {overall['ecfr']:.4f} |",
        f"| textual_dominance (QCFR - ECFR) | {overall['textual_dominance']:.4f} |",
        f"| conflict_follow_question_rate | {overall['conflict_follow_question_rate']:.4f} |",
        f"| conflict_follow_ecg_rate | {overall['conflict_follow_ecg_rate']:.4f} |",
        f"| neutral_question_flip_rate | {overall['neutral_question_flip_rate']:.4f} |",
        "",
    ]
    if dominance:
        lines += ["## Dominance class distribution", "", "| class | count | fraction |", "| --- | --- | --- |"]
        counts = dominance.get("class_counts", {})
        fractions = dominance.get("class_fractions", {})
        for label, count in sorted(counts.items()):
            lines.append(f"| {label} | {count} | {fractions.get(label, 0):.4f} |")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "- `textual_dominance > 0`: the model leans on the **question text** more than the **ECG signal**.",
        "- `textual_dominance < 0`: the model leans on the **ECG signal**.",
        "- High `conflict_follow_question_rate` with abnormal/normal claims that contradict the ECG",
        "  indicates the text overrides clinical evidence — a clinically risky behaviour.",
        f"- Per-group breakdown is in `{args.table_output.name}`.",
        "",
    ]
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"groups": len(rows), "table": str(args.table_output), "report": str(args.report_output)}, indent=2))


if __name__ == "__main__":
    main()
