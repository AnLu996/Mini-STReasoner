"""Modal-ablation evaluation for the ECG-QA small run.

Runs the trained checkpoint over the test subset under four configurations and
measures how much each modality contributes:

    full           text + ECG (reference)
    no_text        ECG only      (text tokens dropped)
    no_series      text only     (temporal tokens dropped)
    conflict_text  text + ECG, with a misleading note pushed against the ECG

Per configuration it reports Exact Match, Token F1 and yes/no accuracy. The
modal dominance follows the project definition (token-F1 based, less noisy than
EM):

    text_contribution = full - no_text
    ecg_contribution  = full - no_series
    textual_dominance = text_contribution - ecg_contribution   ( = no_series - no_text )

Positive dominance => the answer leans on the text more than on the ECG.

Outputs (under ``--results_dir`` / next to ``--output``):
    ablation.jsonl                  per-sample predictions for every config
    ablation_summary.json           per-config metrics + dominance (global + by type)
    ablation_by_config.csv
    ablation_by_question_type.csv

Example::

    python scripts/run_ecgqa_ablation_small.py \\
      --model_path checkpoints/ecgqa_small_lora \\
      --test data/ecgqa_small/processed_test.jsonl \\
      --max_samples 100 \\
      --output outputs/ecgqa_small/ablation.jsonl
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

from scripts.ecgqa_metrics import (  # noqa: E402
    answer_to_text,
    exact_match,
    is_yesno,
    token_f1,
    yesno_correct,
)

# config key -> predict_ecg kwargs
CONFIGS = {
    "full": {"use_text": True, "use_series": True},
    "no_text": {"use_text": False, "use_series": True},
    "no_series": {"use_text": True, "use_series": False},
    "conflict_text": {"use_text": True, "use_series": True, "conflict_text": True},
}


def aggregate(records: list[dict[str, Any]], cfg: str) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"count": 0, "exact_match": 0.0, "token_f1": 0.0, "yesno_accuracy": None}
    yesno = [r for r in records if r["is_yesno"]]
    return {
        "count": n,
        "exact_match": sum(r["configs"][cfg]["exact_match"] for r in records) / n,
        "token_f1": sum(r["configs"][cfg]["token_f1"] for r in records) / n,
        "yesno_accuracy": (sum(r["configs"][cfg]["yesno_correct"] for r in yesno) / len(yesno)) if yesno else None,
    }


def dominance(per_config: dict[str, dict[str, Any]]) -> dict[str, float]:
    full = per_config["full"]["token_f1"]
    no_text = per_config["no_text"]["token_f1"]
    no_series = per_config["no_series"]["token_f1"]
    text_contrib = full - no_text
    ecg_contrib = full - no_series
    return {
        "text_contribution": round(text_contrib, 4),
        "ecg_contribution": round(ecg_contrib, 4),
        "textual_dominance": round(text_contrib - ecg_contrib, 4),
    }


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_config = {cfg: aggregate(records, cfg) for cfg in CONFIGS}
    return {"per_config": per_config, "dominance": dominance(per_config)}


def by_question_type(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record.get("question_type") or "unknown"].append(record)
    return {name: summarise(items) for name, items in sorted(buckets.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modal-ablation evaluation for ECG-QA small run")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed_test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_small/ablation.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="cpu = no GPU power draw (safe but slow)")
    parser.add_argument("--no_quantization", action="store_true")
    parser.add_argument("--no_conflict", action="store_true", help="Skip the conflict_text config")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.no_conflict:
        CONFIGS.pop("conflict_text", None)

    from inference.runtime import load_checkpoint, predict_ecg  # noqa: E402

    tokenizer, model, config = load_checkpoint(
        args.model_path, not args.no_quantization, device=args.device
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    count = 0
    with args.output.open("w", encoding="utf-8") as handle, args.test.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if args.max_samples and count >= args.max_samples:
                break
            row = json.loads(line)
            signal = np.load(row["ecg_signal_path"]).astype(np.float32)
            example = {"question": row["question"], "ecg_signal": [signal.tolist()]}
            gold = answer_to_text(row["answer"])

            per_config: dict[str, dict[str, Any]] = {}
            for cfg, kwargs in CONFIGS.items():
                prediction = predict_ecg(tokenizer, model, config, example,
                                         max_new_tokens=args.max_new_tokens, **kwargs)
                per_config[cfg] = {
                    "prediction": prediction,
                    "exact_match": exact_match(prediction, gold),
                    "token_f1": token_f1(prediction, gold),
                    "yesno_correct": yesno_correct(prediction, gold),
                }

            record = {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row.get("question_type", ""),
                "attribute_type": row.get("attribute_type", ""),
                "is_yesno": is_yesno(gold),
                "configs": per_config,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            preds = {c: per_config[c]["prediction"] for c in CONFIGS}
            print(f"[{count}] gold={gold!r} {preds}", flush=True)

    out_dir = args.output.parent
    global_summary = summarise(records)
    summary = {
        "global": global_summary,
        "by_question_type": by_question_type(records),
        "configs": list(CONFIGS),
    }
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (out_dir / "ablation_by_config.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "count", "exact_match", "token_f1", "yesno_accuracy"])
        for cfg, m in global_summary["per_config"].items():
            writer.writerow([
                cfg, m["count"], f"{m['exact_match']:.4f}", f"{m['token_f1']:.4f}",
                "" if m["yesno_accuracy"] is None else f"{m['yesno_accuracy']:.4f}",
            ])

    with (out_dir / "ablation_by_question_type.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_type", "count", "full_f1", "no_text_f1", "no_series_f1",
                         "text_contribution", "ecg_contribution", "textual_dominance"])
        for name, s in summary["by_question_type"].items():
            pc, dom = s["per_config"], s["dominance"]
            writer.writerow([
                name, pc["full"]["count"], f"{pc['full']['token_f1']:.4f}",
                f"{pc['no_text']['token_f1']:.4f}", f"{pc['no_series']['token_f1']:.4f}",
                f"{dom['text_contribution']:.4f}", f"{dom['ecg_contribution']:.4f}",
                f"{dom['textual_dominance']:.4f}",
            ])

    dom = global_summary["dominance"]
    print("\n".join([
        "",
        f"Samples: {len(records)}",
        *[f"{cfg:>14}: EM={m['exact_match']:.3f}  F1={m['token_f1']:.3f}"
          for cfg, m in global_summary["per_config"].items()],
        f"text_contribution (full-no_text):  {dom['text_contribution']:.4f}",
        f"ecg_contribution  (full-no_series): {dom['ecg_contribution']:.4f}",
        f"textual_dominance:                  {dom['textual_dominance']:.4f}",
    ]))


if __name__ == "__main__":
    main()
