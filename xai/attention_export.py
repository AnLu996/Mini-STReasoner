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
    parser = argparse.ArgumentParser(description="Export temporal-token attention matrices")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/xai/temporal_attention")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tokenizer, model, config = load_checkpoint(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.data_dir / f"{args.task}.jsonl"
    for index, example in enumerate(iter_jsonl([source])):
        if args.limit and index >= args.limit:
            break
        _, attention = predict(tokenizer, model, config, example, max_new_tokens=1)
        payload = {"task": args.task, "index": index, "shape": list(attention[0].shape), "temporal_attention": attention[0].tolist(), "metadata": example.get("metadata", {})}
        (args.output_dir / f"{args.task}_{index:06d}.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
