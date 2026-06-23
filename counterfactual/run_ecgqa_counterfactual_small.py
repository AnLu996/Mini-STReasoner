"""Stage 6 -- small counterfactual probe of text-vs-ECG dominance.

For each of a handful of test cases the trained model is run on a fixed set of
counterfactual variants and the answer flips are turned into dominance metrics:

Variants
    original                                the untouched question + ECG
    question_cf                             meaning-preserving question rewrite
    neutral_question                        clinical cue removed
    ecg_noise / ecg_lead_mask /
    ecg_time_mask / ecg_spike               pure signal perturbations
    conflict_question_normal_ecg_abnormal   question asserts "normal"
    conflict_question_abnormal_ecg_normal   question asserts "abnormal"

Metrics (a *flip* = answer differs from the ``original`` answer)
    QCFR   question_cf flip rate                       (text surface sensitivity)
    ECFR   mean flip rate over the 4 ECG perturbations (signal sensitivity)
    textual_dominance = QCFR - ECFR
    conflict_follow_question_rate / conflict_follow_ecg_rate
    neutral_question_flip_rate

Example::

    python counterfactual/run_ecgqa_counterfactual_small.py \\
      --model_path checkpoints/ecgqa_small_lora \\
      --data data/ecgqa_small/processed_test.jsonl \\
      --max_samples 50 \\
      --output outputs/ecgqa_small/counterfactual_results.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual.counterfactual_metrics import normalize, polarity  # noqa: E402
from counterfactual.transformations_ecg import apply_ecg_transform  # noqa: E402
from counterfactual.transformations_text import apply_text_transform  # noqa: E402

# Map the paper-facing variant names to the canonical ECG transforms.
ECG_VARIANTS = {
    "ecg_noise": "ecg_cf_noise",
    "ecg_lead_mask": "ecg_cf_lead_mask",
    "ecg_time_mask": "ecg_cf_time_mask",
    "ecg_spike": "ecg_cf_spike",
}
TEXT_VARIANTS = ("question_cf", "neutral_question")
CONFLICT_VARIANTS = {
    "conflict_question_normal_ecg_abnormal": "normal",
    "conflict_question_abnormal_ecg_normal": "abnormal",
}
_CLAIM_POLARITY = {"normal": "negative", "abnormal": "positive"}


def build_variants(question: str, signal: np.ndarray, attribute_type: str, seed: int) -> dict[str, dict[str, Any]]:
    """Return {variant_name: {"question": str, "signal": ndarray}}."""
    variants: dict[str, dict[str, Any]] = {"original": {"question": question, "signal": signal}}
    for name in TEXT_VARIANTS:
        new_q, _ = apply_text_transform(question, name, attribute_type=attribute_type, seed=seed)
        variants[name] = {"question": new_q, "signal": signal}
    for name in CONFLICT_VARIANTS:
        # The conflict variant names match the text-transform keys directly.
        new_q, _ = apply_text_transform(question, name, attribute_type=attribute_type, seed=seed)
        variants[name] = {"question": new_q, "signal": signal}
    for name, transform in ECG_VARIANTS.items():
        perturbed = apply_ecg_transform([signal.tolist()], transform, seed=seed)[0]
        variants[name] = {"question": question, "signal": np.asarray(perturbed, dtype=np.float32)}
    return variants


def case_metrics(predictions: dict[str, str]) -> dict[str, Any]:
    """Compute flip-based dominance metrics for a single case."""
    base = normalize(predictions.get("original", ""))

    def flip(name: str) -> int:
        return int(normalize(predictions.get(name, "")) != base)

    qcfr = flip("question_cf")
    neutral_flip = flip("neutral_question")
    ecg_flips = {name: flip(name) for name in ECG_VARIANTS}
    ecfr = sum(ecg_flips.values()) / max(len(ecg_flips), 1)

    follow_q: list[int] = []
    follow_e: list[int] = []
    for name, claim in CONFLICT_VARIANTS.items():
        pred_pol = polarity(predictions.get(name, ""))
        if pred_pol is None:
            continue  # undecidable answer, skip this conflict observation
        follow_q.append(int(pred_pol == _CLAIM_POLARITY[claim]))
        follow_e.append(int(normalize(predictions.get(name, "")) == base))

    return {
        "qcfr": qcfr,
        "ecfr": ecfr,
        "ecg_flips": ecg_flips,
        "neutral_flip": neutral_flip,
        "conflict_follow_question_rate": (sum(follow_q) / len(follow_q)) if follow_q else None,
        "conflict_follow_ecg_rate": (sum(follow_e) / len(follow_e)) if follow_e else None,
        "textual_dominance": qcfr - ecfr,
    }


def _mean(values: list[Any]) -> float:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    qcfr = _mean([c["metrics"]["qcfr"] for c in cases])
    ecfr = _mean([c["metrics"]["ecfr"] for c in cases])
    return {
        "count": len(cases),
        "QCFR": qcfr,
        "ECFR": ecfr,
        "textual_dominance": qcfr - ecfr,
        "conflict_follow_question_rate": _mean([c["metrics"]["conflict_follow_question_rate"] for c in cases]),
        "conflict_follow_ecg_rate": _mean([c["metrics"]["conflict_follow_ecg_rate"] for c in cases]),
        "neutral_question_flip_rate": _mean([c["metrics"]["neutral_flip"] for c in cases]),
    }


def grouped(cases: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        buckets[case.get(key) or "unknown"].append(case)
    return {name: aggregate(items) for name, items in sorted(buckets.items())}


def select_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Pick illustrative cases: clear text-driven or ECG-driven behaviour first."""
    def interest(case: dict[str, Any]) -> float:
        m = case["metrics"]
        signals = [m["qcfr"], m["ecfr"], m["neutral_flip"]]
        signals += [v for v in (m["conflict_follow_question_rate"], m["conflict_follow_ecg_rate"]) if v is not None]
        return abs(m["textual_dominance"]) + sum(signals)
    ranked = sorted(cases, key=interest, reverse=True)
    return ranked[:limit] if limit else ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small ECG-QA counterfactual dominance probe")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed_test.jsonl")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_small/counterfactual_results.jsonl")
    parser.add_argument("--selected_limit", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="cpu = no GPU power draw (safe but slow)")
    parser.add_argument("--no_quantization", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from inference.runtime import load_checkpoint, predict_ecg  # noqa: E402

    tokenizer, model, config = load_checkpoint(
        args.model_path, not args.no_quantization, device=args.device
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    count = 0
    with args.output.open("w", encoding="utf-8") as handle, args.data.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if args.max_samples and count >= args.max_samples:
                break
            row = json.loads(line)
            signal = np.load(row["ecg_signal_path"]).astype(np.float32)
            variants = build_variants(row["question"], signal, row.get("attribute_type", ""), args.seed)

            predictions: dict[str, str] = {}
            for name, variant in variants.items():
                example = {"question": variant["question"], "ecg_signal": [variant["signal"].tolist()]}
                predictions[name] = predict_ecg(tokenizer, model, config, example, max_new_tokens=args.max_new_tokens)

            metrics = case_metrics(predictions)
            case = {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row.get("question_type", ""),
                "attribute_type": row.get("attribute_type", ""),
                "predictions": predictions,
                "metrics": metrics,
            }
            cases.append(case)
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            print(f"[{count}] {row['id']} dom={metrics['textual_dominance']:.2f}", flush=True)

    out_dir = args.output.parent
    summary = {
        "global": aggregate(cases),
        "by_question_type": grouped(cases, "question_type"),
    }
    (out_dir / "counterfactual_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (out_dir / "counterfactual_by_question_type.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_type", "count", "QCFR", "ECFR", "textual_dominance",
                         "conflict_follow_question_rate", "conflict_follow_ecg_rate", "neutral_question_flip_rate"])
        for name, m in summary["by_question_type"].items():
            writer.writerow([
                name, m["count"], f"{m['QCFR']:.4f}", f"{m['ECFR']:.4f}", f"{m['textual_dominance']:.4f}",
                f"{m['conflict_follow_question_rate']:.4f}", f"{m['conflict_follow_ecg_rate']:.4f}",
                f"{m['neutral_question_flip_rate']:.4f}",
            ])

    selected = select_cases(cases, args.selected_limit)
    with (out_dir / "selected_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in selected:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    g = summary["global"]
    print("\n".join([
        "",
        f"Cases: {g['count']}",
        f"QCFR: {g['QCFR']:.4f}",
        f"ECFR: {g['ECFR']:.4f}",
        f"Textual dominance: {g['textual_dominance']:.4f}",
        f"Conflict follows question: {g['conflict_follow_question_rate']:.4f}",
        f"Conflict follows ECG: {g['conflict_follow_ecg_rate']:.4f}",
        f"Selected cases: {len(selected)} -> {out_dir / 'selected_cases.jsonl'}",
    ]))


if __name__ == "__main__":
    main()
