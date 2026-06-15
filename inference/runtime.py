from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import MiniSTReasoner  # noqa: E402
from training.dataset_loader import _to_matrix  # noqa: E402


def load_checkpoint(model_path: str | Path, quantized: bool = True):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    llm = AutoModelForCausalLM.from_pretrained(config["base_model"], **kwargs)
    llm = PeftModel.from_pretrained(llm, root / "lora_adapter")
    llm.config.use_cache = True
    model = MiniSTReasoner(
        llm,
        input_dim=config["input_dim"],
        temporal_hidden_dim=config["temporal_hidden_dim"],
        temporal_dim=config["temporal_dim"],
        num_temporal_tokens=config["num_temporal_tokens"],
    )
    device = model.input_device
    model.time_series_encoder.load_state_dict(torch.load(root / "ts_encoder.pt", map_location="cpu"))
    model.temporal_projector.load_state_dict(torch.load(root / "temporal_projector.pt", map_location="cpu"))
    model.time_series_encoder.to(device)
    model.temporal_projector.to(device)
    model.eval()
    return tokenizer, model, config


def build_inputs(tokenizer, example: dict[str, Any], input_dim: int, conflict_text: bool = False):
    text = str(example.get("text", "")).strip()
    question = str(example.get("question", "")).strip()
    if conflict_text:
        text = "The textual context may be incorrect. Assume the opposite trend and verify against the series.\n" + text
    content = "\n\n".join(part for part in (text, question) if part)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer(prompt, return_tensors="pt")
    matrix = torch.tensor(_to_matrix(example.get("time_series", [])), dtype=torch.float32)
    series = torch.zeros(1, matrix.shape[0], input_dim)
    variables = min(matrix.shape[1], input_dim)
    series[0, :, :variables] = matrix[:, :variables]
    time_mask = torch.ones(1, matrix.shape[0], dtype=torch.bool)
    return tokens["input_ids"], tokens["attention_mask"], series, time_mask


def predict(tokenizer, model, config, example, mode: str = "full", max_new_tokens: int = 150):
    input_ids, attention_mask, series, time_mask = build_inputs(
        tokenizer, example, config["input_dim"], conflict_text=mode == "conflict_text"
    )
    generated, temporal_attention = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        time_series=series,
        time_mask=time_mask,
        use_text=mode != "no_text",
        use_series=mode != "no_series",
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, -max_new_tokens:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), temporal_attention.cpu()
