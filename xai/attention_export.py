"""Persist the temporal-token attention the encoder already computes.

``TimeSeriesEncoder`` returns an attention matrix ``[tokens, T]`` — the weight
each learned temporal query puts on each time step — but nothing ever saved it.
It is the only artefact that answers, per sample, *which part of the series the
model is looking at*, which is what the thesis review asked for.

The raw matrix is one row per token over up to 1000 steps, too wide to plot and
too large to ship to the visualiser, so it is pooled into ``--bins`` contiguous
windows. Because the rows are softmax distributions over time, summing the
weights inside a window is exactly the probability mass the token assigns to it,
and the binned rows still sum to one.

Two summaries accompany each sample: the mass profile aggregated over tokens
(where the encoder looks overall) and the entropy of each token's distribution,
which separates tokens that focus on a segment from tokens that spread out and
effectively average the whole signal away.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.runtime import build_ecg_inputs, build_inputs, load_checkpoint  # noqa: E402
from training.dataset_loader import iter_jsonl  # noqa: E402


def bin_attention(attention: np.ndarray, bins: int) -> np.ndarray:
    """Pool ``[tokens, T]`` into ``[tokens, bins]`` by summing the weight per window."""
    steps = attention.shape[1]
    bins = max(1, min(bins, steps))
    edges = np.linspace(0, steps, bins + 1, dtype=int)
    return np.stack(
        [attention[:, edges[i] : edges[i + 1]].sum(axis=1) for i in range(bins)], axis=1
    )


def row_entropy(attention: np.ndarray) -> list[float]:
    """Shannon entropy of each token's distribution, normalised to [0, 1].

    0 means the token reads a single time step; 1 means it weights the whole
    series uniformly and therefore averages it away.
    """
    steps = attention.shape[1]
    ceiling = math.log(steps) if steps > 1 else 1.0
    out = []
    for row in attention:
        weights = row[row > 0]
        entropy = float(-(weights * np.log(weights)).sum())
        out.append(entropy / ceiling if ceiling else 0.0)
    return out


@torch.no_grad()
def encode(tokenizer, model, config, example: dict[str, Any], data_format: str) -> np.ndarray:
    build = build_inputs if data_format == "stbench" else build_ecg_inputs
    _, _, series, time_mask = build(tokenizer, example, config["input_dim"])
    device = next(model.time_series_encoder.parameters()).device
    _, attention = model.time_series_encoder(series.to(device), time_mask.to(device))
    return attention[0].float().cpu().numpy()


def build_example(row: dict[str, Any], data_format: str) -> dict[str, Any]:
    if data_format == "stbench":
        return row
    signal = np.load(row["ecg_signal_path"]).astype(np.float32)
    return {"question": row.get("question", ""), "ecg_signal": [signal.tolist()]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export temporal-token attention")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-format", choices=["stbench", "ecgqa"], default="ecgqa")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    tokenizer, model, config = load_checkpoint(args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    aggregate = None
    entropies: list[float] = []
    attention = None
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(iter_jsonl([args.data])):
            if args.limit and index >= args.limit:
                break
            attention = encode(
                tokenizer, model, config, build_example(row, args.data_format), args.data_format
            )
            binned = bin_attention(attention, args.bins)
            profile = binned.sum(axis=0) / binned.shape[0]
            entropy = row_entropy(attention)
            entropies.extend(entropy)
            aggregate = profile if aggregate is None else aggregate + profile
            count += 1
            handle.write(
                json.dumps(
                    {
                        "id": row.get("id", index),
                        "index": index,
                        "steps": int(attention.shape[1]),
                        "tokens": int(attention.shape[0]),
                        "bins": int(binned.shape[1]),
                        "attention_binned": binned.round(6).tolist(),
                        "mass_profile": profile.round(6).tolist(),
                        "token_entropy": [round(value, 4) for value in entropy],
                        "question_type": row.get("question_type"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if not count:
        raise SystemExit("no samples processed")

    mean_profile = (aggregate / count).tolist()
    summary = {
        "model_path": str(args.model_path),
        "samples": count,
        "tokens": int(attention.shape[0]),
        "bins": args.bins,
        "mean_mass_profile": [round(value, 6) for value in mean_profile],
        "mean_token_entropy": round(float(np.mean(entropies)), 4),
        "min_token_entropy": round(float(np.min(entropies)), 4),
        "max_token_entropy": round(float(np.max(entropies)), 4),
        # A profile close to uniform means the pooling averages the signal and
        # the encoder is not selecting any segment in particular.
        "uniform_reference": round(1.0 / args.bins, 6),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"muestras: {count}   tokens: {summary['tokens']}   bins: {args.bins}")
    print(
        "entropia por token (0 = mira un instante, 1 = promedia todo): "
        f"media {summary['mean_token_entropy']}  min {summary['min_token_entropy']}  "
        f"max {summary['max_token_entropy']}"
    )
    peak = int(np.argmax(mean_profile))
    print(
        f"perfil de masa: pico en el bin {peak} con {mean_profile[peak]:.4f} "
        f"(uniforme seria {summary['uniform_reference']:.4f})"
    )
    print(f"\nSaved to {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
