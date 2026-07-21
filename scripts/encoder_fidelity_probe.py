"""Linear probes on the temporal representations: does the encoder keep the series?

This answers, with a number, the question raised in the thesis review: *how do we
know the encoder reflects the input series at all?* Accuracy on the downstream
task cannot answer it, because a model can score well without reading the signal.

For each sample the script extracts two representations and fits a ridge probe
from each to simple statistics of the input series (mean, standard deviation,
range, length, number of variables). The comparison that matters is between
stages:

    encoder    output of the GRU + attention pooling      [tokens x temporal_dim]
    projector  after projection into the LLM space        [tokens x hidden_size]

A high R2 at the encoder and a low one at the projector localises the loss at the
projection. A low R2 at both means the encoder itself discards the signal, which
is an argument for a larger encoder rather than a better projection.

An untrained model is probed alongside as a reference: a randomly initialised
GRU already preserves a surprising amount, so a trained encoder that fails to
beat it has learned nothing useful about the series.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.runtime import build_inputs, load_base_model, load_checkpoint  # noqa: E402
from training.dataset_loader import iter_jsonl  # noqa: E402


# TimeSeriesEncoder z-scores each variable before the GRU, so absolute scale is
# discarded by construction. Probing for it measures the normalisation working as
# designed, not a fidelity failure — the informative targets are the ones that
# survive standardisation.
SCALE_TARGETS = ("mean", "std", "range")
SHAPE_TARGETS = ("dominant_frequency", "autocorrelation_lag1", "trend_slope")
STRUCTURE_TARGETS = ("length", "num_variables")
TARGETS = SCALE_TARGETS + SHAPE_TARGETS + STRUCTURE_TARGETS


def series_statistics(series: list[list[float]]) -> dict[str, float]:
    array = np.asarray(series, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]

    # Shape descriptors are computed on the standardised first variable so they
    # are exactly the kind of information the encoder can still carry.
    first = array[:, 0]
    centred = first - first.mean()
    deviation = first.std()
    standardised = centred / deviation if deviation > 1e-9 else centred

    spectrum = np.abs(np.fft.rfft(standardised))
    dominant = float(np.argmax(spectrum[1:]) + 1) if spectrum.size > 1 else 0.0

    if standardised.size > 1 and np.any(standardised):
        lag1 = float(np.corrcoef(standardised[:-1], standardised[1:])[0, 1])
        steps = np.arange(standardised.size, dtype=np.float64)
        slope = float(np.polyfit(steps, standardised, 1)[0])
    else:
        lag1, slope = 0.0, 0.0

    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "range": float(array.max() - array.min()),
        "dominant_frequency": dominant,
        "autocorrelation_lag1": 0.0 if np.isnan(lag1) else lag1,
        "trend_slope": slope,
        "length": float(array.shape[0]),
        "num_variables": float(array.shape[1]),
    }


@torch.no_grad()
def representations(tokenizer, model, config, example: dict[str, Any]) -> dict[str, np.ndarray]:
    _, _, series, time_mask = build_inputs(tokenizer, example, config["input_dim"])
    device = next(model.time_series_encoder.parameters()).device
    tokens, _ = model.time_series_encoder(series.to(device), time_mask.to(device))
    projected = model.temporal_projector(tokens)
    return {
        "encoder": tokens.float().cpu().numpy().reshape(-1),
        "projector": projected.float().cpu().numpy().reshape(-1),
    }


def ridge_r2(features: np.ndarray, target: np.ndarray, folds: int, seed: int) -> float:
    """Cross-validated R2 of a ridge probe, standardised inside each fold."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if np.allclose(target, target[0]):
        return float("nan")
    predictions = np.zeros_like(target, dtype=np.float64)
    for train_index, test_index in KFold(folds, shuffle=True, random_state=seed).split(features):
        probe = make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 6, 17))
        )
        probe.fit(features[train_index], target[train_index])
        predictions[test_index] = probe.predict(features[test_index])
    residual = float(((target - predictions) ** 2).sum())
    total = float(((target - target.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float("nan")


def collect(tokenizer, model, config, examples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    stages: dict[str, list[np.ndarray]] = {"encoder": [], "projector": []}
    for example in examples:
        extracted = representations(tokenizer, model, config, example)
        for stage, vector in extracted.items():
            stages[stage].append(vector)
    return {stage: np.vstack(vectors) for stage, vectors in stages.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probes on the temporal representations")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="JSONL with time_series")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-untrained", action="store_true", help="do not probe a randomly initialised model"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    examples = list(iter_jsonl([args.data]))[: args.limit]
    statistics = {
        name: np.asarray([series_statistics(item["time_series"])[name] for item in examples])
        for name in TARGETS
    }

    tokenizer, model, config = load_checkpoint(args.model_path)
    trained = collect(tokenizer, model, config, examples)

    results: dict[str, Any] = {
        "model_path": str(args.model_path),
        "data": str(args.data),
        "n": len(examples),
        "dimensions": {stage: int(matrix.shape[1]) for stage, matrix in trained.items()},
        "r2": {"trained": {}, "untrained": {}},
    }
    for stage, matrix in trained.items():
        results["r2"]["trained"][stage] = {
            name: ridge_r2(matrix, statistics[name], args.folds, args.seed) for name in TARGETS
        }

    if not args.skip_untrained:
        del model
        torch.cuda.empty_cache()
        base_tokenizer, base_model, base_config = load_base_model(
            config["base_model"],
            input_dim=config["input_dim"],
            temporal_hidden_dim=config["temporal_hidden_dim"],
            temporal_dim=config["temporal_dim"],
            num_temporal_tokens=config["num_temporal_tokens"],
        )
        untrained = collect(base_tokenizer, base_model, base_config, examples)
        for stage, matrix in untrained.items():
            results["r2"]["untrained"][stage] = {
                name: ridge_r2(matrix, statistics[name], args.folds, args.seed)
                for name in TARGETS
            }

    output = args.output or PROJECT_ROOT / "outputs" / "encoder_fidelity_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")

    header = f"{'objetivo':22s}" + "".join(
        f"{f'{kind}/{stage}':>22s}"
        for kind in results["r2"]
        if results["r2"][kind]
        for stage in ("encoder", "projector")
    )
    print(header)
    groups = {
        "escala (el z-score la elimina)": SCALE_TARGETS,
        "forma (sobrevive al z-score)": SHAPE_TARGETS,
        "estructura": STRUCTURE_TARGETS,
    }
    for title, names in groups.items():
        print(f"-- {title}")
        for name in names:
            row = f"{name:22s}"
            for kind in results["r2"]:
                if not results["r2"][kind]:
                    continue
                for stage in ("encoder", "projector"):
                    row += f"{results['r2'][kind][stage][name]:22.4f}"
            print(row)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
