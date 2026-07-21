from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import MiniSTReasoner  # noqa: E402
from training.dataset_loader import SFTCollator, STBenchIterableDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for Mini-STReasoner")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints/mini_streasoner_qwen06")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--temporal-hidden-dim", type=int, default=128)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--num-temporal-tokens", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 processes every record")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument(
        "--init-from",
        type=Path,
        help="continue from a previous checkpoint (its LoRA adapter, encoder and "
        "projector), as the reference pipeline chains alignment into reasoning",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
    if not args.no_qlora:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    llm = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    if not args.no_qlora:
        llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
    else:
        llm.gradient_checkpointing_enable()
        llm.enable_input_require_grads()
    llm.config.use_cache = False
    if args.init_from:
        from peft import PeftModel

        llm = PeftModel.from_pretrained(
            llm, args.init_from / "lora_adapter", is_trainable=True
        )
    else:
        llm = get_peft_model(
            llm,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
    model = MiniSTReasoner(
        llm,
        input_dim=args.input_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_dim=args.temporal_dim,
        num_temporal_tokens=args.num_temporal_tokens,
    )
    if args.init_from:
        model.time_series_encoder.load_state_dict(
            torch.load(args.init_from / "ts_encoder.pt", map_location="cpu")
        )
        model.temporal_projector.load_state_dict(
            torch.load(args.init_from / "temporal_projector.pt", map_location="cpu")
        )
    model.time_series_encoder.to(model.input_device)
    model.temporal_projector.to(model.input_device)
    return tokenizer, model


def save_checkpoint(args: argparse.Namespace, tokenizer, model: MiniSTReasoner) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(args.output_dir / "lora_adapter")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    torch.save(model.time_series_encoder.state_dict(), args.output_dir / "ts_encoder.pt")
    torch.save(model.temporal_projector.state_dict(), args.output_dir / "temporal_projector.pt")
    config = {
        "base_model": args.model_name,
        "input_dim": args.input_dim,
        "temporal_hidden_dim": args.temporal_hidden_dim,
        "temporal_dim": args.temporal_dim,
        "num_temporal_tokens": args.num_temporal_tokens,
        "max_seq_length": args.max_seq_length,
        "qlora": not args.no_qlora,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args)
    dataset = STBenchIterableDataset(args.processed_dir, args.tasks)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=SFTCollator(tokenizer, args.max_seq_length, args.input_dim),
        num_workers=0,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    for epoch in range(args.epochs):
        # Averaged over every micro-batch: the loss of the last one is a single
        # sample, and because the tasks are interleaved it is always drawn from
        # the same task, so on its own it says nothing about convergence.
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch in loader:
            output = model(**batch)
            epoch_loss_sum += float(output.llm_output.loss.item())
            epoch_batches += 1
            loss = output.llm_output.loss / args.gradient_accumulation_steps
            loss.backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    running = epoch_loss_sum / max(epoch_batches, 1)
                    value = loss.item() * args.gradient_accumulation_steps
                    print(
                        f"epoch={epoch + 1} step={global_step} "
                        f"loss={value:.4f} epoch_mean={running:.4f}",
                        flush=True,
                    )
                if args.max_steps and global_step >= args.max_steps:
                    break
        print(
            f"[epoch] epoch={epoch + 1} "
            f"train_loss={epoch_loss_sum / max(epoch_batches, 1):.4f} "
            f"batches={epoch_batches}",
            flush=True,
        )
        if args.max_steps and global_step >= args.max_steps:
            break
    if micro_step % args.gradient_accumulation_steps:
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    save_checkpoint(args, tokenizer, model)
    print(f"Saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
