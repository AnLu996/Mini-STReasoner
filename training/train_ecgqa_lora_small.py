"""Stage 4 -- small LoRA SFT on the ECG-QA subset.

Trains only three things, leaving the rest of Qwen3-0.6B frozen:

* the ECG / time-series encoder (:class:`models.TimeSeriesEncoder`),
* the temporal projector (:class:`models.TemporalProjector`),
* LoRA adapters on the LLM attention projections.

It reads the ``processed_*.jsonl`` files from Stage 2 (question + ECG ``.npy`` +
answer), logs per-step train loss and per-epoch validation metrics, and writes a
checkpoint in the exact layout :func:`inference.runtime.load_checkpoint` expects.

Example::

    python training/train_ecgqa_lora_small.py \\
      --train data/ecgqa_small/processed_train.jsonl \\
      --valid data/ecgqa_small/processed_valid.jsonl \\
      --output_dir checkpoints/ecgqa_small_lora \\
      --epochs 1 --max_samples 300 --batch_size 1 --grad_accum 8 --max_seq_len 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import MiniSTReasoner  # noqa: E402
from scripts.ecgqa_metrics import answer_to_text, exact_match, token_f1  # noqa: E402


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def read_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


class ProcessedECGQADataset(Dataset):
    """Map-style dataset over processed rows; loads each ECG ``.npy`` lazily."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        row["_signal"] = np.load(row["ecg_signal_path"]).astype(np.float32)
        return row


class ECGQACollator:
    """Build LLM inputs + ECG tensors, masking the prompt tokens in the labels."""

    def __init__(self, tokenizer, max_seq_len: int, max_leads: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_leads = max_leads

    def _encode(self, row: dict[str, Any]) -> tuple[list[int], list[int]]:
        question = str(row.get("question", "")).strip()
        answer = answer_to_text(row.get("answer", "")).strip()
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"User: {question}\nAssistant:"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            answer_ids = answer_ids + [eos]
        room = max(1, self.max_seq_len - len(answer_ids))
        prompt_ids = prompt_ids[-room:]
        ids = (prompt_ids + answer_ids)[: self.max_seq_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_seq_len]
        return ids, labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(row) for row in batch]
        max_tokens = max(len(ids) for ids, _ in encoded)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids, attention_mask, labels = [], [], []
        for ids, item_labels in encoded:
            padding = max_tokens - len(ids)
            input_ids.append(ids + [pad_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(item_labels + [-100] * padding)

        signals = [row["_signal"] for row in batch]
        max_steps = max(s.shape[0] for s in signals)
        series = torch.zeros(len(signals), max_steps, self.max_leads, dtype=torch.float32)
        time_mask = torch.zeros(len(signals), max_steps, dtype=torch.bool)
        for index, signal in enumerate(signals):
            tensor = torch.from_numpy(np.ascontiguousarray(signal))
            leads = min(tensor.shape[1], self.max_leads)
            series[index, : tensor.shape[0], :leads] = tensor[:, :leads]
            time_mask[index, : tensor.shape[0]] = True

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "time_series": series,
            "time_mask": time_mask,
        }


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(args: argparse.Namespace, device: str):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_qlora = (not args.no_qlora) and device == "cuda"
    if device == "cpu":
        load_kwargs: dict[str, Any] = {"torch_dtype": torch.float32, "trust_remote_code": True}
    else:
        load_kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
    if use_qlora:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    llm = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if device == "cpu":
        llm = llm.to("cpu")
    if use_qlora:
        llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
    else:
        llm.gradient_checkpointing_enable()
        llm.enable_input_require_grads()
    llm.config.use_cache = False
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
        input_dim=args.max_leads,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_dim=args.temporal_dim,
        num_temporal_tokens=args.num_temporal_tokens,
    )
    model.time_series_encoder.to(model.input_device)
    model.temporal_projector.to(model.input_device)
    return tokenizer, model, use_qlora


def save_checkpoint(args: argparse.Namespace, tokenizer, model: MiniSTReasoner, qlora: bool) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(out / "lora_adapter")
    tokenizer.save_pretrained(out / "tokenizer")
    torch.save(model.time_series_encoder.state_dict(), out / "ts_encoder.pt")
    torch.save(model.temporal_projector.state_dict(), out / "temporal_projector.pt")
    config = {
        "base_model": args.base_model,
        "input_dim": args.max_leads,
        "temporal_hidden_dim": args.temporal_hidden_dim,
        "temporal_dim": args.temporal_dim,
        "num_temporal_tokens": args.num_temporal_tokens,
        "max_seq_length": args.max_seq_len,
        "qlora": qlora,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_valid(model, tokenizer, config, valid_rows, collator, device, max_new_tokens, valid_max):
    """Return (valid_loss, exact_match, token_f1) on a capped validation subset."""
    from inference.runtime import predict_ecg

    rows = valid_rows[:valid_max] if valid_max else valid_rows
    if not rows:
        return None, None, None
    model.eval()

    # Teacher-forced loss (batched), then greedy EM/F1.
    loss_sum = 0.0
    batches = 0
    for start in range(0, len(rows), collator_batch := 4):
        chunk = []
        for row in rows[start : start + collator_batch]:
            item = dict(row)
            item["_signal"] = np.load(row["ecg_signal_path"]).astype(np.float32)
            chunk.append(item)
        batch = collator(chunk)
        out = model(**batch)
        loss_sum += float(out.llm_output.loss.item())
        batches += 1
    valid_loss = loss_sum / max(batches, 1)

    em_sum = f1_sum = 0.0
    for row in rows:
        signal = np.load(row["ecg_signal_path"]).astype(np.float32)
        example = {"question": row["question"], "ecg_signal": [signal.tolist()]}
        prediction = predict_ecg(tokenizer, model, config, example, max_new_tokens=max_new_tokens)
        gold = answer_to_text(row["answer"])
        em_sum += exact_match(prediction, gold)
        f1_sum += token_f1(prediction, gold)
    n = len(rows)
    return valid_loss, em_sum / n, f1_sum / n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small LoRA SFT for Mini-STReasoner on ECG-QA")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed_train.jsonl")
    parser.add_argument("--valid", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed_valid.jsonl")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "checkpoints/ecgqa_small_lora")
    parser.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--valid_max_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_leads", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_temporal_tokens", type=int, default=4)
    parser.add_argument("--temporal_hidden_dim", type=int, default=128)
    parser.add_argument("--temporal_dim", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no_qlora", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_small")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"[train] device={device} qlora={(not args.no_qlora) and device == 'cuda'}")

    tokenizer, model, qlora = load_model(args, device)
    config = {
        "input_dim": args.max_leads,
        "temporal_hidden_dim": args.temporal_hidden_dim,
        "temporal_dim": args.temporal_dim,
        "num_temporal_tokens": args.num_temporal_tokens,
    }

    train_rows = read_rows(args.train, args.max_samples)
    valid_rows = read_rows(args.valid, args.valid_max_samples) if args.valid.exists() else []
    print(f"[train] train={len(train_rows)} valid={len(valid_rows)}")

    collator = ECGQACollator(tokenizer, args.max_seq_len, args.max_leads)
    loader = DataLoader(
        ProcessedECGQADataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    training_log = args.log_dir / "training_log.jsonl"
    log_handle = training_log.open("w", encoding="utf-8")

    global_step = micro_step = 0
    last_loss = float("nan")
    epoch_metrics: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            output = model(**batch)
            loss = output.llm_output.loss / args.grad_accum
            loss.backward()
            micro_step += 1
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                last_loss = float(loss.item() * args.grad_accum)
                if global_step % args.log_every == 0:
                    record = {"epoch": epoch + 1, "step": global_step, "train_loss": last_loss}
                    print(f"epoch={epoch + 1} step={global_step} train_loss={last_loss:.4f}", flush=True)
                    log_handle.write(json.dumps(record) + "\n")
                    log_handle.flush()
        # Flush a trailing partial accumulation window.
        if micro_step % args.grad_accum:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        valid_loss, em, f1 = evaluate_valid(
            model, tokenizer, config, valid_rows, collator, device, args.max_new_tokens, args.valid_max_samples
        )
        metrics = {"epoch": epoch + 1, "valid_loss": valid_loss, "exact_match": em, "token_f1": f1}
        epoch_metrics.append(metrics)
        print(f"[valid] epoch={epoch + 1} valid_loss={valid_loss} exact_match={em} token_f1={f1}", flush=True)
        log_handle.write(json.dumps(metrics) + "\n")
        log_handle.flush()

    log_handle.close()
    save_checkpoint(args, tokenizer, model, qlora)

    summary = {
        "train_samples": len(train_rows),
        "valid_samples": len(valid_rows),
        "steps": global_step,
        "final_train_loss": last_loss,
        "epoch_metrics": epoch_metrics,
        "checkpoint": str(args.output_dir),
        "device": device,
        "qlora": qlora,
    }
    (args.log_dir / "metrics_train.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
