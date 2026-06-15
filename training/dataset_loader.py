from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset


TASKS = (
    "reasoning_forecasting",
    "reasoning_entity",
    "reasoning_etiological",
    "reasoning_correlation",
)


def iter_jsonl(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


class STBenchIterableDataset(IterableDataset):
    def __init__(
        self,
        processed_dir: str | Path,
        tasks: list[str] | None = None,
        exclude_source_patterns: tuple[str, ...] = ("st-test",),
    ) -> None:
        super().__init__()
        root = Path(processed_dir)
        selected = tasks or list(TASKS)
        self.paths = [root / f"{task}.jsonl" for task in selected]
        self.paths = [path for path in self.paths if path.exists()]
        self.exclude_source_patterns = tuple(item.lower() for item in exclude_source_patterns)
        if not self.paths:
            raise FileNotFoundError(f"No processed JSONL files found under {root}")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        paths = self.paths
        if worker is not None:
            paths = paths[worker.id :: worker.num_workers]
        for item in iter_jsonl(paths):
            source = str(item.get("metadata", {}).get("source", "")).lower()
            if any(pattern in source for pattern in self.exclude_source_patterns):
                continue
            yield item


def _to_matrix(value: Any) -> list[list[float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [float(item) for item in value.replace(",", " ").split()]
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor[:, None]
    elif tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor.tolist()


class SFTCollator:
    def __init__(self, tokenizer, max_seq_length: int = 512, input_dim: int = 1) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.input_dim = input_dim

    def _encode(self, example: dict[str, Any]) -> tuple[list[int], list[int]]:
        text = str(example.get("text", "")).strip()
        question = str(example.get("question", "")).strip()
        answer = str(example.get("answer", "")).strip()
        user = "\n\n".join(part for part in (text, question) if part)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"User: {user}\nAssistant:"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            answer_ids = answer_ids + [eos]
        room = max(1, self.max_seq_length - len(answer_ids))
        prompt_ids = prompt_ids[-room:]
        ids = (prompt_ids + answer_ids)[: self.max_seq_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_seq_length]
        return ids, labels

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(item) for item in examples]
        max_tokens = max(len(ids) for ids, _ in encoded)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for ids, item_labels in encoded:
            padding = max_tokens - len(ids)
            input_ids.append(ids + [pad_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(item_labels + [-100] * padding)

        matrices = [_to_matrix(item.get("time_series", [])) for item in examples]
        max_steps = max(len(matrix) for matrix in matrices)
        series = torch.zeros(len(matrices), max_steps, self.input_dim, dtype=torch.float32)
        time_mask = torch.zeros(len(matrices), max_steps, dtype=torch.bool)
        for index, matrix in enumerate(matrices):
            tensor = torch.as_tensor(matrix, dtype=torch.float32)
            variables = min(tensor.shape[1], self.input_dim)
            series[index, : tensor.shape[0], :variables] = tensor[:, :variables]
            time_mask[index, : tensor.shape[0]] = True

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "time_series": series,
            "time_mask": time_mask,
        }
