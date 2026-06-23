"""ECG-QA data access for Mini-STReasoner.

This replaces the ST-Bench graph/spatial loader with a clinical
``question + ECG time series + answer`` loader. There are no graphs, nodes,
edges or adjacency matrices anywhere: the only structured modality is the ECG
signal with shape ``[time, leads]`` (typically 12 leads).

A normalized ECG-QA sample looks like::

    {
      "id": "ptbxl/test/000123",
      "question": "Does this ECG show atrial fibrillation?",
      "answer": "yes",
      "ecg_id": [123],
      "question_type": "single-verify",
      "attribute_type": "rhythm",
      "ecg_signal": [[[...12 leads...], ...T steps...]],   # list of 1 or 2 ECGs
      "metadata": {...}
    }

``ecg_signal`` is always a *list* of signals so comparison questions can carry
two ECGs. :func:`merge_signals` flattens that list into the single
``[time, leads]`` tensor the encoder consumes, inserting a zero marker row
between two ECGs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

# Question types supported by ECG-QA (single + comparison families).
QUESTION_TYPES = (
    "single-verify",
    "single-choose",
    "single-query",
    "comparison_consecutive-verify",
    "comparison_consecutive-query",
    "comparison_irrelevant-verify",
    "comparison_irrelevant-query",
)

DEFAULT_LEADS = 12
DEFAULT_STEPS = 250  # resampled length kept small for a 6 GB laptop


def iter_ecgqa(paths: list[Path]) -> Iterator[dict[str, Any]]:
    """Yield normalized ECG-QA samples from one or more JSONL files."""
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def resample_signal(signal: list[list[float]], steps: int, leads: int) -> list[list[float]]:
    """Truncate/pad a ``[time, leads]`` signal to a fixed ``[steps, leads]`` grid."""
    rows = len(signal)
    if rows == 0:
        return [[0.0] * leads for _ in range(steps)]
    out: list[list[float]] = []
    for t in range(steps):
        source = signal[min(int(t * rows / steps), rows - 1)]
        row = [float(source[v]) if v < len(source) else 0.0 for v in range(leads)]
        out.append(row)
    return out


def synth_ecg(
    seed: int,
    steps: int = DEFAULT_STEPS,
    leads: int = DEFAULT_LEADS,
    abnormal: bool = False,
) -> list[list[float]]:
    """Deterministic ECG-like signal, used when WFDB files are unavailable.

    Each lead is a phase-shifted heartbeat (narrow QRS-like pulse + baseline
    wander). ``abnormal`` injects an ST-like offset and an extra harmonic so
    that "normal" and "abnormal" signals are statistically separable.
    """
    rate = 1.2 + (seed % 5) * 0.1  # beats per window, varies per record
    out: list[list[float]] = []
    for t in range(steps):
        phase = 2 * math.pi * rate * t / steps
        row = []
        for lead in range(leads):
            shift = lead * 0.4
            qrs = math.exp(-((math.sin(phase + shift)) ** 2) * 12.0)
            baseline = 0.05 * math.sin(0.3 * phase + lead)
            value = qrs + baseline
            if abnormal:
                value += 0.25 + 0.15 * math.sin(2 * phase + shift)
            row.append(value)
        out.append(row)
    return out


def load_wfdb_signal(
    ecg_path: str,
    steps: int = DEFAULT_STEPS,
    leads: int = DEFAULT_LEADS,
) -> list[list[float]]:
    """Read a WFDB record (path without extension) into ``[steps, leads]``."""
    import wfdb  # local import: only needed when real signals are present

    record = wfdb.rdrecord(ecg_path)
    signal = record.p_signal  # shape [time, channels]
    as_list = [[float(v) for v in row] for row in signal]
    return resample_signal(as_list, steps, leads)


def merge_signals(
    ecg_signal: list[list[list[float]]],
    add_marker: bool = True,
) -> list[list[float]]:
    """Flatten a list of ECGs into one ``[time, leads]`` matrix.

    For a single ECG this is a no-op. For two ECGs (comparison questions) the
    signals are concatenated along time with an optional zero marker row in
    between so the encoder can tell the boundary.
    """
    if not ecg_signal:
        return []
    if len(ecg_signal) == 1:
        return [list(row) for row in ecg_signal[0]]
    leads = max(len(row) for sig in ecg_signal for row in sig) if ecg_signal else DEFAULT_LEADS
    merged: list[list[float]] = []
    for index, signal in enumerate(ecg_signal):
        if index > 0 and add_marker:
            merged.append([0.0] * leads)
        for row in signal:
            padded = [float(row[v]) if v < len(row) else 0.0 for v in range(leads)]
            merged.append(padded)
    return merged
