from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.runtime import build_inputs, load_checkpoint  # noqa: E402
from training.dataset_loader import iter_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient saliency over time-series values")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/xai/temporal_saliency.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tokenizer, model, config = load_checkpoint(args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.data_dir / f"{args.task}.jsonl"
    with args.output.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(iter_jsonl([source])):
            if args.limit and index >= args.limit:
                break
            ids, mask, series, time_mask = build_inputs(tokenizer, example, config["input_dim"])
            series.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            embeds, combined_mask, _ = model.encode_modalities(ids, mask, series, time_mask)
            logits = model.llm(inputs_embeds=embeds, attention_mask=combined_mask).logits[:, -1]
            logits.max(dim=-1).values.sum().backward()
            saliency = series.grad.detach().abs().sum(dim=-1)[0]
            saliency = saliency / saliency.sum().clamp_min(1e-12)
            handle.write(json.dumps({"task": args.task, "index": index, "saliency": saliency.tolist(), "metadata": example.get("metadata", {})}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
