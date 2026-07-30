"""F03.2 · Curva dosis-respuesta CFR(δ) por modalidad.

Barre las intensidades del catálogo (mismos puntos que
``delta_calibration.py``), y para cada punto:

1. Aplica la intervención sobre todos los casos del conjunto de evaluación.
2. Mide el δ representativo (media sobre casos) — reutilizando la lógica de
   ``delta_calibration``.
3. Ejecuta el modelo sobre original e intervenido y calcula la tasa de
   cambio de respuesta CFR.

El resultado es una curva ``CFR(δ)`` por modalidad (texto y señal). El brief
§4.2 la propone como reemplazo de los puntos únicos QCFR y ECFR: dos curvas
que puedan superponerse y leerse en el mismo plano δ-CFR son más informativas
y mucho más honestas metodológicamente.

Exporta al contrato del visualizador en el campo ``cfrByDelta`` de cada
``Condition`` (véase ``visualizer/data_contract.md``).

Uso típico::

    python counterfactual/generate_dose_response.py \\
      --model_path checkpoints/ecgqa_small_lora \\
      --data data/ecgqa_small/processed_test.jsonl \\
      --max_samples 150 \\
      --output outputs/audit/cfr_dose_response.json

Costo estimado (Corrida A, RTX 4050, 150 casos):
- 30 intensidades × 150 casos × 2 generaciones = 9000 generaciones.
- ~3-5 h. Reduce ``--max_samples`` para pruebas rápidas.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual.delta_calibration import (  # noqa: E402
    ECG_INTENSITY_GRID, TEXT_INTERVENTIONS, _iter_intensities,
    delta_ecg, delta_text,
)
from counterfactual.transformations_ecg import apply_ecg_transform  # noqa: E402
from counterfactual.transformations_text import apply_text_transform  # noqa: E402
from inference.runtime import load_checkpoint, predict_ecg  # noqa: E402


# --------------------------------------------------------------------------- #
# Normalización de respuestas — misma función que usa flip detection en xai.  #
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", str(text).lower()).strip()


def is_flip(pred_orig: str, pred_pert: str) -> bool:
    return _norm(pred_orig) != _norm(pred_pert)


# --------------------------------------------------------------------------- #
# Cálculo de CFR y δ para un punto (intervención, intensidad) sobre todos los  #
# casos.                                                                       #
# --------------------------------------------------------------------------- #
def dose_point_ecg(
    model,
    tokenizer,
    config,
    rows: list[dict[str, Any]],
    predictions_orig: list[str],
    intervention: str,
    params: dict[str, Any],
    seed_base: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Un punto de la curva para una intervención sobre la señal."""
    deltas_pool: list[float] = []
    deltas_bigru: list[float] = []
    flips: list[int] = []
    for i, row in enumerate(rows):
        signal = np.load(row["ecg_signal_path"]).astype(np.float32)
        seed = seed_base + i
        perturbed = apply_ecg_transform([signal], intervention, params=params, seed=seed)[0]
        perturbed_np = np.asarray(perturbed, dtype=np.float32)
        # δ
        d = delta_ecg(model, signal, perturbed_np)
        deltas_pool.append(d["delta_pool"])
        deltas_bigru.append(d["delta_bigru"])
        # Predicción sobre intervenido (la original ya se calculó una vez fuera).
        example = {"question": str(row.get("question", "")), "ecg_signal": [perturbed_np.tolist()]}
        pred_pert = predict_ecg(tokenizer, model, config, example, max_new_tokens=max_new_tokens)
        flips.append(1 if is_flip(predictions_orig[i], pred_pert) else 0)
    n = len(rows)
    return {
        "modality": "signal",
        "intervention": intervention,
        "params": params,
        "n": n,
        "delta":       float(np.mean(deltas_pool))  if deltas_pool  else None,
        "delta_bigru": float(np.mean(deltas_bigru)) if deltas_bigru else None,
        "cfr":         (sum(flips) / n) if n else None,
        "flips":       int(sum(flips)),
    }


def dose_point_text(
    model,
    tokenizer,
    config,
    rows: list[dict[str, Any]],
    predictions_orig: list[str],
    intervention: str,
    seed_base: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Un punto de la curva para una intervención textual (no paramétrica)."""
    deltas_text: list[float] = []
    flips: list[int] = []
    for i, row in enumerate(rows):
        signal = np.load(row["ecg_signal_path"]).astype(np.float32)
        question = str(row.get("question", ""))
        seed = seed_base + i
        new_q, _meta = apply_text_transform(question, intervention, seed=seed)
        # δ textual
        dt = delta_text(tokenizer, model, question, new_q)
        deltas_text.append(dt["delta_text"])
        # Predicción sobre pregunta intervenida (misma señal).
        example = {"question": new_q, "ecg_signal": [signal.tolist()]}
        pred_pert = predict_ecg(tokenizer, model, config, example, max_new_tokens=max_new_tokens)
        flips.append(1 if is_flip(predictions_orig[i], pred_pert) else 0)
    n = len(rows)
    return {
        "modality": "text",
        "intervention": intervention,
        "params": {},
        "n": n,
        "delta": float(np.mean(deltas_text)) if deltas_text else None,
        "cfr":   (sum(flips) / n) if n else None,
        "flips": int(sum(flips)),
    }


# --------------------------------------------------------------------------- #
# Interpolación monótona para leer CFR a un δ concreto (útil en el paper)     #
# --------------------------------------------------------------------------- #
def cfr_at_delta(points: list[dict[str, Any]], target_delta: float) -> float | None:
    """Devuelve CFR interpolado (lineal) al δ objetivo, o None si el objetivo
    queda fuera del rango observado. Ordena por δ y evita saltos numéricos."""
    ok = [(p["delta"], p["cfr"]) for p in points if p.get("delta") is not None and p.get("cfr") is not None]
    if len(ok) < 2:
        return None
    ok.sort(key=lambda t: t[0])
    xs, ys = zip(*ok)
    if target_delta < xs[0] or target_delta > xs[-1]:
        return None
    return float(np.interp(target_delta, xs, ys))


# --------------------------------------------------------------------------- #
# Export al contrato del visualizador                                          #
# --------------------------------------------------------------------------- #
def export_for_visualizer(all_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Formato ``cfrByDelta`` del contrato de datos:
    ``[{modality, points:[{delta, cfr, intervention, params}, ...]}]``.
    Colapsa todas las intervenciones de una modalidad en una única curva."""
    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in all_points:
        if p.get("delta") is None or p.get("cfr") is None:
            continue
        by_modality[p["modality"]].append({
            "delta": round(float(p["delta"]), 6),
            "cfr":   round(float(p["cfr"]),   4),
            "intervention": p["intervention"],
            "params":       p["params"],
        })
    out: list[dict[str, Any]] = []
    for modality, pts in by_modality.items():
        pts.sort(key=lambda x: x["delta"])
        out.append({"modality": modality, "points": pts})
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _read_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F03.2 · Curva dosis-respuesta CFR(δ) por modalidad."
    )
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Checkpoint E1/E2/E3 con encoder.")
    parser.add_argument("--data", type=Path,
                        default=PROJECT_ROOT / "data/ecgqa_small/processed_test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/audit/cfr_dose_response.json")
    parser.add_argument("--only_modality", choices=["signal", "text", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model, config = load_checkpoint(args.model_path, device=args.device)
    print(f"[dose] modelo cargado: {args.model_path}")

    rows = _read_rows(args.data, args.max_samples)
    print(f"[dose] casos: {len(rows)}")

    # 1) Predicciones originales — una única pasada, se reutilizan en todos los puntos.
    print("[dose] generando predicciones originales…")
    predictions_orig: list[str] = []
    for i, row in enumerate(rows):
        signal = np.load(row["ecg_signal_path"]).astype(np.float32)
        example = {"question": str(row.get("question", "")), "ecg_signal": [signal.tolist()]}
        predictions_orig.append(predict_ecg(tokenizer, model, config, example,
                                            max_new_tokens=args.max_new_tokens))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    all_points: list[dict[str, Any]] = []

    # 2) Signal grid
    if args.only_modality in ("signal", "both"):
        for intervention in ECG_INTENSITY_GRID:
            for params in _iter_intensities(intervention):
                print(f"[dose] signal · {intervention} {params}")
                point = dose_point_ecg(
                    model, tokenizer, config, rows, predictions_orig,
                    intervention, params, args.seed_base, args.max_new_tokens,
                )
                all_points.append(point)
                print(f"  δ={point['delta']:.5f}  CFR={point['cfr']:.3f}  flips={point['flips']}/{point['n']}")

    # 3) Text interventions
    if args.only_modality in ("text", "both"):
        for intervention in TEXT_INTERVENTIONS:
            print(f"[dose] text · {intervention}")
            point = dose_point_text(
                model, tokenizer, config, rows, predictions_orig,
                intervention, args.seed_base, args.max_new_tokens,
            )
            all_points.append(point)
            print(f"  δ={point['delta']:.5f}  CFR={point['cfr']:.3f}  flips={point['flips']}/{point['n']}")

    # 4) Lecturas útiles para el paper: CFR a δ objetivo (comparaciones honestas).
    signal_points = [p for p in all_points if p["modality"] == "signal"]
    text_points   = [p for p in all_points if p["modality"] == "text"]
    highlights: dict[str, Any] = {}
    for tp in text_points:
        if tp.get("delta") is None: continue
        cfr_sig_at_dt = cfr_at_delta(signal_points, tp["delta"])
        highlights[tp["intervention"]] = {
            "delta_text":       tp["delta"],
            "cfr_text":         tp["cfr"],
            "cfr_signal_at_dt": cfr_sig_at_dt,
            "gap":              None if cfr_sig_at_dt is None else round(tp["cfr"] - cfr_sig_at_dt, 4),
        }

    payload = {
        "checkpoint": str(args.model_path),
        "n_cases": len(rows),
        "points": all_points,
        "cfr_by_delta": export_for_visualizer(all_points),
        "highlights_text_vs_signal_at_matched_delta": highlights,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[dose] escrito {args.output}")
    print(f"[dose] puntos totales: {len(all_points)} "
          f"(signal={len(signal_points)}, text={len(text_points)})")


if __name__ == "__main__":
    main()
