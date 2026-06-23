"""Stage 2 -- load the real ECG signals referenced by the manifest.

Reads ``manifest.jsonl`` (from :mod:`scripts.download_ecgqa_small`), loads each
PTB-XL WFDB record with ``wfdb``, normalises it to a fixed ``[target_length,
max_leads]`` grid (linear time interpolation + per-lead z-score) and saves it as
a compact ``.npy`` file. The JSONL output stores only the *path* to that array,
never the raw tensor, so ``processed.jsonl`` stays tiny.

Comparison questions (two ECGs) are concatenated along time with a zero marker
row between them before resampling, matching
:func:`training.ecgqa_loader.merge_signals`.

Besides ``processed.jsonl`` (all rows) it also emits ``processed_<split>.jsonl``
for each split present, which Stages 4 and 5 consume directly.

Example::

    python scripts/prepare_ecg_signals.py \\
      --manifest data/ecgqa_small/manifest.jsonl \\
      --output data/ecgqa_small/processed.jsonl \\
      --target_length 1000 --max_leads 12
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_wfdb_array(ecg_path: str, max_leads: int) -> np.ndarray:
    """Load one WFDB record as ``[time, leads]`` (up to ``max_leads`` leads)."""
    import wfdb  # local import: only needed when real signals are present

    record = wfdb.rdrecord(ecg_path)
    signal = np.asarray(record.p_signal, dtype=np.float32)  # [time, channels]
    if signal.ndim == 1:
        signal = signal[:, None]
    return signal[:, :max_leads]


def merge_arrays(arrays: list[np.ndarray], max_leads: int) -> np.ndarray:
    """Concatenate 1-2 ECGs along time with a zero marker row between them."""
    padded = []
    for array in arrays:
        leads = array.shape[1]
        if leads < max_leads:
            array = np.pad(array, ((0, 0), (0, max_leads - leads)))
        padded.append(array[:, :max_leads])
    if len(padded) == 1:
        return padded[0]
    marker = np.zeros((1, max_leads), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, array in enumerate(padded):
        if index > 0:
            pieces.append(marker)
        pieces.append(array)
    return np.concatenate(pieces, axis=0)


def resample_time(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly interpolate each lead onto ``target_length`` time steps."""
    steps = signal.shape[0]
    if steps == target_length:
        return signal.astype(np.float32)
    if steps < 2:
        return np.repeat(signal[:1], target_length, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, steps)
    dst = np.linspace(0.0, 1.0, target_length)
    out = np.empty((target_length, signal.shape[1]), dtype=np.float32)
    for lead in range(signal.shape[1]):
        out[:, lead] = np.interp(dst, src, signal[:, lead])
    return out


def zscore_per_lead(signal: np.ndarray) -> np.ndarray:
    """Z-score each lead over time; leave constant leads untouched."""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return ((signal - mean) / std).astype(np.float32)


def process_row(row: dict[str, Any], target_length: int, max_leads: int) -> np.ndarray | None:
    """Load + normalise the ECG(s) of one manifest row to ``[target_length, max_leads]``."""
    paths = row.get("ecg_path") or []
    if isinstance(paths, str):
        paths = [paths]
    arrays: list[np.ndarray] = []
    for path in paths:
        try:
            arrays.append(load_wfdb_array(str(path), max_leads))
        except Exception as exc:  # noqa: BLE001 - any WFDB/IO error -> skip the row
            print(f"[warn] WFDB load failed for {path} ({exc})", file=sys.stderr)
            return None
    if not arrays:
        return None
    merged = merge_arrays(arrays, max_leads)
    resampled = resample_time(merged, target_length)
    return zscore_per_lead(resampled)


def safe_stem(identifier: str) -> str:
    """Turn an id like ``ptbxl/train/123`` into a filesystem-safe stem."""
    return identifier.replace("/", "_").replace("\\", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and normalise ECG signals for the small run")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/manifest.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed.jsonl")
    parser.add_argument("--target_length", type=int, default=1000)
    parser.add_argument("--max_leads", type=int, default=12)
    parser.add_argument("--signals_dir", type=Path, default=None,
                        help="Where to write <id>.npy (default: <output dir>/signals)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in the output (crash-safe)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    signals_dir = args.signals_dir or (out_dir / "signals")
    signals_dir.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        print(f"[resume] {len(done_ids)} rows already processed, will skip them")

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    written = skipped = 0
    open_mode = "a" if (args.resume and args.output.exists()) else "w"
    with args.manifest.open(encoding="utf-8") as source, args.output.open(open_mode, encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in done_ids:
                continue
            signal = process_row(row, args.target_length, args.max_leads)
            if signal is None:
                skipped += 1
                continue
            stem = safe_stem(row["id"])
            npy_path = signals_dir / f"{stem}.npy"
            np.save(npy_path, signal)
            processed = {
                "id": row["id"],
                "split": row.get("split", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", []),
                "ecg_id": row.get("ecg_id", []),
                "ecg_signal_path": str(npy_path),
                "ecg_shape": list(signal.shape),
                "question_type": row.get("question_type", ""),
                "attribute_type": row.get("attribute_type", ""),
                "metadata": row.get("metadata", {}),
            }
            line_out = json.dumps(processed, ensure_ascii=False)
            sink.write(line_out + "\n")
            sink.flush()
            by_split[processed["split"]].append(processed)
            written += 1
            if written % 25 == 0:
                print(f"[{written}] processed (skipped {skipped})", flush=True)

    # Per-split files for Stages 4/5. Built from the rows written this run; when
    # resuming, re-read the full output so the split files stay complete.
    if args.resume and open_mode == "a":
        by_split.clear()
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    by_split[item.get("split", "")].append(item)

    split_counts: dict[str, int] = {}
    for split, items in by_split.items():
        if not split:
            continue
        split_path = out_dir / f"processed_{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        split_counts[split] = len(items)

    summary = {
        "written": written,
        "skipped": skipped,
        "target_length": args.target_length,
        "max_leads": args.max_leads,
        "ecg_shape": [args.target_length, args.max_leads],
        "by_split": split_counts,
        "output": str(args.output),
        "signals_dir": str(signals_dir),
    }
    (out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
