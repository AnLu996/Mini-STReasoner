"""Stage 5 -- evaluate the trained checkpoint on the ECG-QA test subset.

Runs the trained Mini-STReasoner over ``processed_test.jsonl`` and reports
Exact Match, Token F1 and yes/no accuracy, both globally and broken down by
``question_type`` and ``attribute_type``. Per-sample predictions go to a JSONL,
the aggregate to JSON, and the breakdowns to two CSV files for the paper tables.

Example::

    python scripts/evaluate_ecgqa_small.py \\
      --model_path checkpoints/ecgqa_small_lora \\
      --test data/ecgqa_small/processed_test.jsonl \\
      --max_samples 100 \\
      --output outputs/ecgqa_small/evaluation.jsonl
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
    is_valid_prediction,
    is_yesno,
    token_f1,
    yesno_correct,
)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean metrics over a list of per-sample evaluation records."""
    n = len(records)
    if n == 0:
        return {"count": 0, "exact_match": 0.0, "token_f1": 0.0, "yesno_accuracy": None, "yesno_count": 0}
    yesno = [r for r in records if r["is_yesno"]]
    return {
        "count": n,
        "exact_match": sum(r["exact_match"] for r in records) / n,
        "token_f1": sum(r["token_f1"] for r in records) / n,
        "yesno_accuracy": (sum(r["yesno_correct"] for r in yesno) / len(yesno)) if yesno else None,
        "yesno_count": len(yesno),
    }


def grouped(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record.get(key) or "unknown"].append(record)
    return {name: aggregate(items) for name, items in sorted(buckets.items())}


def write_breakdown_csv(path: Path, breakdown: dict[str, dict[str, Any]], key_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([key_name, "count", "exact_match", "token_f1", "yesno_accuracy", "yesno_count"])
        for name, metrics in breakdown.items():
            writer.writerow([
                name, metrics["count"],
                f"{metrics['exact_match']:.4f}", f"{metrics['token_f1']:.4f}",
                "" if metrics["yesno_accuracy"] is None else f"{metrics['yesno_accuracy']:.4f}",
                metrics["yesno_count"],
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Mini-STReasoner on ECG-QA test subset")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed_test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_small/evaluation.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=64)
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
            prediction = predict_ecg(tokenizer, model, config, example, max_new_tokens=args.max_new_tokens)
            gold = answer_to_text(row["answer"])
            record = {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "prediction": prediction,
                "question_type": row.get("question_type", ""),
                "attribute_type": row.get("attribute_type", ""),
                "exact_match": exact_match(prediction, gold),
                "token_f1": token_f1(prediction, gold),
                "is_yesno": is_yesno(gold),
                "yesno_correct": yesno_correct(prediction, gold),
                "valid": is_valid_prediction(prediction),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            print(f"[{count}] gold={gold!r} pred={prediction!r}", flush=True)

    global_metrics = aggregate(records)
    by_qtype = grouped(records, "question_type")
    by_attr = grouped(records, "attribute_type")

    summary = {"global": global_metrics, "by_question_type": by_qtype, "by_attribute_type": by_attr}
    out_dir = args.output.parent
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_breakdown_csv(out_dir / "evaluation_by_question_type.csv", by_qtype, "question_type")
    write_breakdown_csv(out_dir / "evaluation_by_attribute_type.csv", by_attr, "attribute_type")

    yn = global_metrics["yesno_accuracy"]
    print("\n".join([
        "",
        f"Total samples: {global_metrics['count']}",
        f"Exact Match: {global_metrics['exact_match']:.4f}",
        f"Token F1: {global_metrics['token_f1']:.4f}",
        f"Yes/No Accuracy: {'n/a' if yn is None else f'{yn:.4f}'} (n={global_metrics['yesno_count']})",
    ]))


if __name__ == "__main__":
    main()
