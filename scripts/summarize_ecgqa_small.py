"""Final stage -- print a paper-ready summary of the small ECG-QA run.

Reads the artefacts produced by the earlier stages (no model needed) and prints
the consolidated summary block, also saving it to ``run_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def unique_ecgs(paths: list[Path]) -> int:
    seen: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    for eid in json.loads(line).get("ecg_id", []):
                        seen.add(int(eid))
    return len(seen)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise the small ECG-QA run")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data/ecgqa_small")
    parser.add_argument("--outputs_dir", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_small")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir, out_dir = args.data_dir, args.outputs_dir

    train_n = count_lines(data_dir / "processed_train.jsonl")
    valid_n = count_lines(data_dir / "processed_valid.jsonl")
    test_n = count_lines(data_dir / "processed_test.jsonl")
    all_n = count_lines(data_dir / "processed.jsonl")
    n_ecgs = unique_ecgs([
        data_dir / "processed_train.jsonl",
        data_dir / "processed_valid.jsonl",
        data_dir / "processed_test.jsonl",
        data_dir / "processed.jsonl",
    ])

    prepare = load_json(data_dir / "prepare_summary.json")
    ecg_shape = prepare.get("ecg_shape", [1000, 12])

    evaluation = load_json(out_dir / "evaluation_summary.json").get("global", {})
    cf = load_json(out_dir / "counterfactual_summary.json").get("global", {})
    abl = load_json(out_dir / "ablation_summary.json").get("global", {})
    abl_per_config = abl.get("per_config", {})
    abl_dom = abl.get("dominance", {})
    selected_path = out_dir / "selected_cases.jsonl"

    summary = {
        "train_samples": train_n,
        "valid_samples": valid_n,
        "test_samples": test_n,
        "all_samples": all_n,
        "unique_ecgs": n_ecgs,
        "ecg_shape": ecg_shape,
        "evaluation": evaluation,
        "counterfactual": cf,
        "ablation": abl,
        "selected_cases": str(selected_path),
    }
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "",
        "Mini-STReasoner ECG-QA Small Run Summary",
        "",
        f"Train samples: {train_n}",
        f"Valid samples: {valid_n}",
        f"Test samples: {test_n}",
        f"Unique ECGs: {n_ecgs}",
        f"ECG shape: {ecg_shape}",
        "",
        "Evaluation:",
        f"Exact Match: {fmt(evaluation.get('exact_match'))}",
        f"Token F1: {fmt(evaluation.get('token_f1'))}",
        f"Yes/No Accuracy: {fmt(evaluation.get('yesno_accuracy'))}",
        "",
        "Counterfactual:",
        f"QCFR: {fmt(cf.get('QCFR'))}",
        f"ECFR: {fmt(cf.get('ECFR'))}",
        f"Textual dominance: {fmt(cf.get('textual_dominance'))}",
        f"Conflict follows question: {fmt(cf.get('conflict_follow_question_rate'))}",
        f"Conflict follows ECG: {fmt(cf.get('conflict_follow_ecg_rate'))}",
        "",
        "Ablation (Token F1 por configuración):",
        *([f"  {cfg}: {fmt(m.get('token_f1'))}" for cfg, m in abl_per_config.items()] or ["  (no calculada)"]),
        f"Text contribution (full−no_text): {fmt(abl_dom.get('text_contribution'))}",
        f"ECG contribution (full−no_series): {fmt(abl_dom.get('ecg_contribution'))}",
        f"Modal textual dominance: {fmt(abl_dom.get('textual_dominance'))}",
        "",
        "Selected cases saved to:",
        str(selected_path),
        "",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
