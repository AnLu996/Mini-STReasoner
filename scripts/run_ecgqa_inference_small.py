"""Stage 3 -- baseline inference with no training.

Loads base Qwen3-0.6B wrapped in :class:`models.MiniSTReasoner` with a *fresh*
(untrained) ECG encoder + projector, feeds each processed ECG-QA sample
(question + real ECG signal) through ``inputs_embeds`` and records the generated
answer. This establishes the "before training" reference point for the paper.

Example::

    python scripts/run_ecgqa_inference_small.py \\
      --data data/ecgqa_small/processed.jsonl \\
      --max_samples 20 \\
      --output outputs/ecgqa_small/inference_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ecgqa_metrics import (  # noqa: E402
    answer_to_text,
    exact_match,
    is_valid_prediction,
    token_f1,
)


def iter_processed(path: Path, max_samples: int) -> Iterator[dict[str, Any]]:
    """Yield processed rows (with the ECG array loaded from its .npy path)."""
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if max_samples and count >= max_samples:
                break
            row = json.loads(line)
            signal = np.load(row["ecg_signal_path"]).astype(np.float32)
            row["_signal"] = signal
            yield row
            count += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline (untrained) ECG-QA inference")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_small/inference_raw.jsonl")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max_leads", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="cpu = no GPU power draw (safe but slow)")
    parser.add_argument("--quantized", action="store_true", help="4-bit load (GPU only)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from inference.runtime import load_base_model, predict_ecg  # noqa: E402

    tokenizer, model, config = load_base_model(
        base_model=args.base_model,
        input_dim=args.max_leads,
        device=args.device,
        quantized=args.quantized,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = valid = 0
    em_sum = f1_sum = 0.0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in iter_processed(args.data, args.max_samples):
            example = {"question": row["question"], "ecg_signal": [row["_signal"].tolist()]}
            prediction = predict_ecg(tokenizer, model, config, example, max_new_tokens=args.max_new_tokens)
            gold = answer_to_text(row["answer"])
            em = exact_match(prediction, gold)
            f1 = token_f1(prediction, gold)
            total += 1
            valid += int(is_valid_prediction(prediction))
            em_sum += em
            f1_sum += f1
            handle.write(json.dumps({
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "prediction": prediction,
                "question_type": row.get("question_type", ""),
                "attribute_type": row.get("attribute_type", ""),
                "ecg_shape": row.get("ecg_shape", list(row["_signal"].shape)),
                "exact_match": em,
                "token_f1": f1,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{total}] gold={gold!r} pred={prediction!r}", flush=True)

    denom = max(total, 1)
    summary = {
        "total_samples": total,
        "valid_predictions": valid,
        "exact_match": em_sum / denom,
        "token_f1": f1_sum / denom,
        "output": str(args.output),
    }
    (args.output.parent / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n" + "\n".join([
        f"Total samples: {total}",
        f"Valid predictions: {valid}",
        f"Exact match: {summary['exact_match']:.4f}",
        f"Token F1: {summary['token_f1']:.4f}",
    ]))


if __name__ == "__main__":
    main()
