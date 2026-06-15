from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Iterator


ALIASES = {
    "text": ("text", "context", "description", "instruction", "input", "prompt"),
    "question": ("question", "query"),
    "time_series": ("time_series", "timeseries", "series", "ts", "data"),
    "answer": ("answer", "response", "output", "target", "label"),
    "task": ("task", "task_name", "category", "type"),
}


def first_value(record: dict[str, Any], names: tuple[str, ...], default: Any = "") -> Any:
    lowered = {str(key).lower(): key for key in record}
    for name in names:
        if name in lowered and record[lowered[name]] is not None:
            return record[lowered[name]]
    return default


def infer_task(record: dict[str, Any], source: str) -> str:
    explicit = str(first_value(record, ALIASES["task"], "")).lower()
    combined = f"{explicit} {source}".lower()
    for short in ("forecasting", "entity", "etiological", "correlation"):
        if short in combined:
            return f"reasoning_{short}"
    if "align" in combined:
        return "alignment"
    if "causal" in combined:
        return "causal"
    return "other"


def normalize_time_series(value: Any) -> Any:
    if not isinstance(value, (list, tuple)) or not value:
        return value
    if not isinstance(value[0], (list, tuple)):
        return [[item] for item in value]
    rows = len(value)
    columns = len(value[0]) if value[0] else 0
    rectangular = all(isinstance(row, (list, tuple)) and len(row) == columns for row in value)
    if rectangular and rows <= 64 and columns > rows:
        return [list(step) for step in zip(*value)]
    return value


def normalize(record: dict[str, Any], source: str) -> dict[str, Any] | None:
    time_series = first_value(record, ALIASES["time_series"], None)
    answer = first_value(record, ALIASES["answer"], "")
    if time_series is None or answer in (None, ""):
        return None
    used = {name for names in ALIASES.values() for name in names}
    metadata = {key: value for key, value in record.items() if str(key).lower() not in used}
    metadata["source"] = source
    return {
        "task": infer_task(record, source),
        "text": str(first_value(record, ALIASES["text"], "")),
        "question": str(first_value(record, ALIASES["question"], "")),
        "time_series": normalize_time_series(time_series),
        "answer": str(answer),
        "metadata": metadata,
    }


def iter_local(root: Path) -> Iterator[tuple[dict[str, Any], str]]:
    for path in sorted(root.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line), str(path.relative_to(root))
    for path in sorted(root.rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if isinstance(row, dict):
                yield row, str(path.relative_to(root))


def iter_huggingface(
    dataset_id: str, configs: list[str] | None = None
) -> Iterator[tuple[dict[str, Any], str]]:
    from datasets import get_dataset_config_names, load_dataset

    available = get_dataset_config_names(dataset_id)
    selected = configs or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown dataset configs: {unknown}")
    for config_name in selected:
        dataset = load_dataset(dataset_id, config_name, streaming=True)
        if hasattr(dataset, "items"):
            for split, rows in dataset.items():
                for row in rows:
                    yield dict(row), f"{config_name}/{split}"
        else:
            for row in dataset:
                yield dict(row), config_name


def prepare(rows: Iterable[tuple[dict[str, Any], str]], output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    with ExitStack() as stack:
        handles: dict[str, Any] = {}
        for record, source in rows:
            item = normalize(record, source)
            if item is None:
                continue
            task = re.sub(r"[^a-z0-9_]+", "_", item["task"].lower())
            if task not in handles:
                handles[task] = stack.enter_context(
                    (output_dir / f"{task}.jsonl").open("w", encoding="utf-8")
                )
            handles[task].write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[task] += 1
    (output_dir / "manifest.json").write_text(
        json.dumps(dict(counts), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream ST-Bench into task-specific SFT JSONL files")
    parser.add_argument("--dataset-id", default="Time-HD-Anonymous/ST-Bench")
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--configs", nargs="*", default=None)
    args = parser.parse_args()
    rows = (
        iter_local(args.local_dir)
        if args.local_dir
        else iter_huggingface(args.dataset_id, args.configs)
    )
    counts = prepare(rows, args.output_dir)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
