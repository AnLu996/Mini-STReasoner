from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.runtime import load_base_model, load_checkpoint, predict  # noqa: E402
from training.dataset_loader import iter_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mini-STReasoner inference")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--base-model",
        help="Run the untrained baseline with this HF model id instead of a checkpoint",
    )
    parser.add_argument("--input-dim", type=int, default=10, help="only used with --base-model")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--no-quantization", action="store_true")
    args = parser.parse_args()
    if not args.model_path and not args.base_model:
        parser.error("pass either --model-path or --base-model")

    source = args.data_dir / f"{args.task}.jsonl"
    output = args.output or PROJECT_ROOT / "outputs" / f"predictions_{args.task}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.base_model:
        tokenizer, model, config = load_base_model(args.base_model, input_dim=args.input_dim)
    else:
        tokenizer, model, config = load_checkpoint(args.model_path, not args.no_quantization)
    with output.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(iter_jsonl([source])):
            if args.limit and index >= args.limit:
                break
            prediction, _ = predict(tokenizer, model, config, example, max_new_tokens=args.max_new_tokens)
            row = {"task": args.task, "prediction": prediction, "answer": example.get("answer", ""), "metadata": example.get("metadata", {})}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{index + 1}] {prediction}")
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
