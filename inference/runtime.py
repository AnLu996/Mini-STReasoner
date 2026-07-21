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
from training.ecgqa_loader import merge_signals  # noqa: E402


def load_checkpoint(model_path: str | Path, quantized: bool = True, device: str = "auto"):
    """Load the checkpoint.

    ``device`` may be ``"auto"``/``"cuda"`` (GPU, optionally 4-bit) or ``"cpu"``.
    The CPU path is slow but draws no GPU power, so it is the safe option on a
    laptop with marginal power delivery — it disables quantization (bitsandbytes
    needs CUDA) and loads the 0.6B model in float32.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", trust_remote_code=True)
    use_cpu = device == "cpu"
    if use_cpu:
        quantized = False
        kwargs: dict[str, Any] = {"torch_dtype": torch.float32, "trust_remote_code": True}
    else:
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    llm = AutoModelForCausalLM.from_pretrained(config["base_model"], **kwargs)
    if use_cpu:
        llm = llm.to("cpu")
    llm = PeftModel.from_pretrained(llm, root / "lora_adapter")
    llm.config.use_cache = True
    model = MiniSTReasoner(
        llm,
        input_dim=config["input_dim"],
        temporal_hidden_dim=config["temporal_hidden_dim"],
        temporal_dim=config["temporal_dim"],
        num_temporal_tokens=config["num_temporal_tokens"],
        temporal_num_layers=config.get("temporal_num_layers", 1),
        match_embedding_scale=config.get("match_embedding_scale", False),
    )
    device = model.input_device
    model.time_series_encoder.load_state_dict(torch.load(root / "ts_encoder.pt", map_location="cpu"))
    model.temporal_projector.load_state_dict(torch.load(root / "temporal_projector.pt", map_location="cpu"))
    model.time_series_encoder.to(device)
    model.temporal_projector.to(device)
    model.eval()
    return tokenizer, model, config


def load_base_model(
    base_model: str = "Qwen/Qwen3-0.6B",
    input_dim: int = 12,
    temporal_hidden_dim: int = 128,
    temporal_dim: int = 256,
    num_temporal_tokens: int = 4,
    device: str = "auto",
    quantized: bool = False,
):
    """Build an *untrained* MiniSTReasoner: base LLM + fresh temporal modules.

    Used by the Stage-3 no-training baseline. The time-series encoder and the
    temporal projector are randomly initialised, so the ECG channel is present
    but uncalibrated; predictions reflect the base Qwen behaviour with an
    untrained ECG path. No LoRA is attached.

    ``device`` follows the same convention as :func:`load_checkpoint`: ``"cpu"``
    draws no GPU power (float32, slow) while ``"auto"``/``"cuda"`` use the GPU and
    may 4-bit quantize when ``quantized`` is set.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_cpu = device == "cpu"
    if use_cpu:
        quantized = False
        kwargs: dict[str, Any] = {"torch_dtype": torch.float32, "trust_remote_code": True}
    else:
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    llm = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if use_cpu:
        llm = llm.to("cpu")
    llm.config.use_cache = True
    model = MiniSTReasoner(
        llm,
        input_dim=input_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        temporal_dim=temporal_dim,
        num_temporal_tokens=num_temporal_tokens,
    )
    model.time_series_encoder.to(model.input_device)
    model.temporal_projector.to(model.input_device)
    model.eval()
    config = {
        "base_model": base_model,
        "input_dim": input_dim,
        "temporal_hidden_dim": temporal_hidden_dim,
        "temporal_dim": temporal_dim,
        "num_temporal_tokens": num_temporal_tokens,
    }
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


def build_ecg_inputs(tokenizer, example: dict[str, Any], input_dim: int):
    """Build LLM inputs for an ECG-QA sample (question + ECG signal).

    ``example`` carries ``question`` and ``ecg_signal`` (a list of 1 or 2 ECGs,
    each ``[time, leads]``). The signals are merged into a single temporal
    tensor; there is no graph or spatial structure involved.
    """
    question = str(example.get("question", "")).strip()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer(prompt, return_tensors="pt")

    merged = merge_signals(example.get("ecg_signal", []))
    matrix = torch.tensor(merged if merged else [[0.0] * input_dim], dtype=torch.float32)
    series = torch.zeros(1, matrix.shape[0], input_dim)
    variables = min(matrix.shape[1], input_dim)
    series[0, :, :variables] = matrix[:, :variables]
    time_mask = torch.ones(1, matrix.shape[0], dtype=torch.bool)
    return tokens["input_ids"], tokens["attention_mask"], series, time_mask


def predict_ecg(
    tokenizer,
    model,
    config,
    example,
    max_new_tokens: int = 64,
    use_text: bool = True,
    use_series: bool = True,
    conflict_text: bool = False,
) -> str:
    """Greedy-decode an answer for one ECG-QA example.

    ``use_text`` / ``use_series`` drop a whole modality for modal-ablation runs
    (only the temporal tokens, or only the text tokens, reach the LLM).
    ``conflict_text`` prepends a misleading note so the question pushes against
    the ECG evidence.
    """
    if conflict_text:
        question = str(example.get("question", ""))
        example = {
            **example,
            "question": "Note: the accompanying clinical note may be incorrect; "
            "rely on the ECG signal itself. " + question,
        }
    input_ids, attention_mask, series, time_mask = build_ecg_inputs(
        tokenizer, example, config["input_dim"]
    )
    generated, _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        time_series=series,
        time_mask=time_mask,
        use_text=use_text,
        use_series=use_series,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, -max_new_tokens:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
