from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.runtime import load_checkpoint, predict  # noqa: E402
from training.dataset_loader import iter_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text/time-series modal ablations")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/xai/modal_ablation_results.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    args = parser.parse_args()
    tokenizer, model, config = load_checkpoint(args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.data_dir / f"{args.task}.jsonl"
    with args.output.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(iter_jsonl([source])):
            if args.limit and index >= args.limit:
                break
            predictions = {}
            for mode in ("full", "no_text", "no_series", "conflict_text"):
                predictions[mode], _ = predict(tokenizer, model, config, example, mode, args.max_new_tokens)
            handle.write(json.dumps({"task": args.task, "answer": example.get("answer", ""), "predictions": predictions, "metadata": example.get("metadata", {})}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
