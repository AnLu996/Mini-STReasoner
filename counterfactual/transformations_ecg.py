"""ECG signal counterfactual transformations.

Every transformation takes a single ECG signal as a 2D array ``[time, leads]``
and returns a perturbed copy of the same shape. They are deterministic given a
``seed`` so that the whole pipeline is reproducible.

The public entry point is :func:`apply_ecg_transform`, which maps a
counterfactual name (``ecg_cf_noise`` ...) plus a parameter dict to the matching
function and applies it to *every* signal of the sample (a sample may carry two
ECGs for comparison questions).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _as_array(signal: Any) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"ECG signal must be [time, leads], got shape {array.shape}")
    return array


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def add_noise(signal: Any, level: float = 0.05, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise scaled by each lead's standard deviation."""
    array = _as_array(signal).copy()
    per_lead_std = array.std(axis=0, keepdims=True)
    per_lead_std = np.where(per_lead_std > 0, per_lead_std, 1.0)
    noise = _rng(seed).standard_normal(array.shape).astype(np.float32)
    return array + level * per_lead_std * noise


def scale_amplitude(signal: Any, factor: float = 1.5, seed: int = 0) -> np.ndarray:
    """Multiply the whole signal amplitude by ``factor``."""
    return _as_array(signal).copy() * float(factor)


def mask_leads(signal: Any, leads: Any = None, fraction: float = 0.25, seed: int = 0) -> np.ndarray:
    """Zero out one or more leads (derivations)."""
    array = _as_array(signal).copy()
    num_leads = array.shape[1]
    if leads is None:
        count = max(1, int(round(num_leads * fraction)))
        leads = _rng(seed).choice(num_leads, size=min(count, num_leads), replace=False)
    leads = [int(lead) % num_leads for lead in np.atleast_1d(leads)]
    array[:, leads] = 0.0
    return array


def mask_time(signal: Any, fraction: float = 0.25, start: float | None = None, seed: int = 0) -> np.ndarray:
    """Zero out a contiguous temporal window."""
    array = _as_array(signal).copy()
    steps = array.shape[0]
    width = max(1, int(round(steps * fraction)))
    if start is None:
        begin = int(_rng(seed).integers(0, max(1, steps - width + 1)))
    else:
        begin = int(start * steps) if start < 1 else int(start)
    begin = max(0, min(begin, steps - 1))
    end = min(steps, begin + width)
    array[begin:end, :] = 0.0
    return array


def inject_spike(signal: Any, lead: int | None = None, magnitude: float = 8.0, seed: int = 0) -> np.ndarray:
    """Introduce a sharp artificial spike at a random time/lead."""
    array = _as_array(signal).copy()
    steps, num_leads = array.shape
    rng = _rng(seed)
    if lead is None:
        lead = int(rng.integers(0, num_leads))
    position = int(rng.integers(0, steps))
    scale = array[:, lead].std() or 1.0
    array[position, lead] += magnitude * scale
    return array


def shuffle_time(signal: Any, num_segments: int = 8, seed: int = 0) -> np.ndarray:
    """Break the signal into segments and shuffle their temporal order."""
    array = _as_array(signal).copy()
    steps = array.shape[0]
    num_segments = max(2, min(num_segments, steps))
    bounds = np.linspace(0, steps, num_segments + 1, dtype=int)
    segments = [array[bounds[i]:bounds[i + 1]] for i in range(num_segments)]
    order = _rng(seed).permutation(len(segments))
    return np.concatenate([segments[i] for i in order], axis=0)


# name -> (function, default params)
ECG_TRANSFORMS: dict[str, tuple[Callable[..., np.ndarray], dict[str, Any]]] = {
    "ecg_cf_noise": (add_noise, {"level": 0.08}),
    "ecg_cf_scaling": (scale_amplitude, {"factor": 1.6}),
    "ecg_cf_lead_mask": (mask_leads, {"fraction": 0.25}),
    "ecg_cf_time_mask": (mask_time, {"fraction": 0.25}),
    "ecg_cf_spike": (inject_spike, {"magnitude": 8.0}),
    "ecg_cf_shuffle": (shuffle_time, {"num_segments": 8}),
}


def default_params(name: str) -> dict[str, Any]:
    if name not in ECG_TRANSFORMS:
        raise KeyError(f"Unknown ECG counterfactual: {name}")
    return dict(ECG_TRANSFORMS[name][1])


def apply_ecg_transform(
    signals: list[Any],
    name: str,
    params: dict[str, Any] | None = None,
    seed: int = 0,
) -> list[list[list[float]]]:
    """Apply an ECG counterfactual to a list of signals (1 or 2 ECGs).

    The same perturbation (with the same seed) is applied to every ECG so that a
    comparison question is perturbed coherently. Returns plain nested lists so
    the result is JSON-serialisable.
    """
    if name not in ECG_TRANSFORMS:
        raise KeyError(f"Unknown ECG counterfactual: {name}")
    function, defaults = ECG_TRANSFORMS[name]
    merged = {**defaults, **(params or {})}
    out: list[list[list[float]]] = []
    for index, signal in enumerate(signals):
        # Vary the seed per ECG so the two signals are not perturbed identically.
        transformed = function(signal, seed=seed + index, **merged)
        out.append(np.asarray(transformed, dtype=np.float32).tolist())
    return out
