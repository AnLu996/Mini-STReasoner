"""F03.1 · Calibración de intervenciones contrafactuales por desplazamiento
representacional δ.

Motivación (Sección VI-C del paper): comparar QCFR y ECFR sin equiparar la
magnitud de la intervención no informa sobre la sensibilidad relativa del
modelo, sino sobre la magnitud arbitraria de la perturbación. En la Corrida A
esto se manifestó de forma extrema: el ruido gaussiano de media cero mueve la
salida del codificador entre 5e-6 y 5e-5, mientras que la oclusión temporal la
mueve entre 1e-2 y 1e-1 — tres a cuatro órdenes de magnitud de diferencia por
la misma etiqueta ``ECFR``.

Este script mide δ (distancia coseno entre representación original e
intervenida) en dos puntos del pipeline:

- ``δ_bigru``  = distancia coseno en la salida del BiGRU (pre-pool).
- ``δ_pool``   = distancia coseno en la salida del pooling atencional.
- ``δ_text``   = distancia coseno en el promedio de embeddings de texto del LLM
  (para intervenciones sobre la pregunta).

Sobre esa tabla, ``pair_by_delta`` empareja intervenciones textuales y de señal
por δ comparable (dentro de un factor 2, siguiendo el brief §4.2), y produce
grupos de calibración que otros scripts (evaluación contrafactual) consumen
para reportar sólo comparaciones honestas.

Este módulo NO ejecuta el LLM ni generación — sólo captura activaciones. Es
barato: una pasada de encoder por caso e intervención. Diseñado para correr en
Ubuntu con GPU CUDA, pero funciona en CPU (más lento).

Uso típico::

    python counterfactual/delta_calibration.py \\
      --model_path checkpoints/ecgqa_small_lora \\
      --data data/ecgqa_small/processed_test.jsonl \\
      --max_samples 100 \\
      --output outputs/audit/delta_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual.transformations_ecg import ECG_TRANSFORMS  # noqa: E402
from counterfactual.transformations_text import TEXT_TRANSFORMS  # noqa: E402
from inference.runtime import build_ecg_inputs, load_checkpoint  # noqa: E402
from xai.representational_tracing import cosine_distance  # noqa: E402


# --------------------------------------------------------------------------- #
# Rejilla de intensidades por intervención — se comparte con                   #
# ``generate_dose_response.py`` (F03.2) para que la curva CFR(δ) y la tabla   #
# de calibración usen exactamente los mismos puntos.                           #
# --------------------------------------------------------------------------- #
ECG_INTENSITY_GRID: dict[str, dict[str, list[float]]] = {
    # add_noise(level): media cero, muy débil por diseño del pooling atencional.
    "ecg_cf_noise":     {"level":    [0.01, 0.05, 0.10, 0.20, 0.40]},
    # scale_amplitude(factor): multiplicativa, sin cambio de media.
    "ecg_cf_scaling":   {"factor":   [1.05, 1.2, 1.5, 2.0, 3.0]},
    # mask_leads(fraction): 1 a 6 derivaciones de 12.
    "ecg_cf_lead_mask": {"fraction": [0.10, 0.25, 0.33, 0.50, 0.75]},
    # mask_time(fraction): 5% a 50% del registro.
    "ecg_cf_time_mask": {"fraction": [0.05, 0.10, 0.25, 0.35, 0.50]},
    # inject_spike(magnitude): σ de la señal.
    "ecg_cf_spike":     {"magnitude": [2.0, 4.0, 8.0, 16.0, 32.0]},
    # shuffle_time(num_segments): 2 a 16 bloques (menos = permutación más destructiva).
    "ecg_cf_shuffle":   {"num_segments": [2, 4, 8, 12, 16]},
}

# Intervenciones textuales — sólo tienen "intensidad" implícita (no paramétrica);
# reportamos una única medida por ellas. Se listan aquí para la interfaz.
TEXT_INTERVENTIONS = tuple(TEXT_TRANSFORMS.keys())


# --------------------------------------------------------------------------- #
# Captura de activaciones (barata: sólo encoder y proyector)                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_signal(model, signal: np.ndarray) -> dict[str, np.ndarray]:
    """Pasa una señal por el codificador y devuelve las dos activaciones que
    necesitamos: h_bigru (pre-pool) y h_pool (post-pool). Ambas se promedian
    sobre la dimensión temporal / de tokens para dar vectores comparables.
    """
    device = next(model.time_series_encoder.parameters()).device
    x = torch.as_tensor(signal, dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)          # [1, T, C]
    temporal_tokens, _attention, bigru_output = model.time_series_encoder(x)
    # Promedio de la dimensión temporal / de tokens para un vector por muestra.
    return {
        "h_bigru": bigru_output.mean(dim=1)[0].float().cpu().numpy(),
        "h_pool":  temporal_tokens.mean(dim=1)[0].float().cpu().numpy(),
    }


@torch.no_grad()
def encode_text(tokenizer, model, question: str) -> np.ndarray:
    """Promedio de los embeddings del LLM sobre los tokens de la pregunta."""
    device = model.input_device
    # Aplicamos el mismo prompt template que en generación para que la unidad
    # sea la que efectivamente ve el LLM.
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = f"User: {question}\nAssistant:"
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    embed_layer = model.llm.get_input_embeddings()
    embeds = embed_layer(ids["input_ids"])
    return embeds.mean(dim=1)[0].float().cpu().numpy()


# --------------------------------------------------------------------------- #
# Cálculo de δ por intervención                                                #
# --------------------------------------------------------------------------- #
def delta_ecg(model, signal_orig: np.ndarray, signal_pert: np.ndarray) -> dict[str, float]:
    """δ en las dos etapas ECG donde la intervención puede notarse."""
    o = encode_signal(model, signal_orig)
    p = encode_signal(model, signal_pert)
    return {
        "delta_bigru": float(cosine_distance(o["h_bigru"], p["h_bigru"])),
        "delta_pool":  float(cosine_distance(o["h_pool"],  p["h_pool"])),
    }


def delta_text(tokenizer, model, q_orig: str, q_pert: str) -> dict[str, float]:
    o = encode_text(tokenizer, model, q_orig)
    p = encode_text(tokenizer, model, q_pert)
    return {"delta_text": float(cosine_distance(o, p))}


# --------------------------------------------------------------------------- #
# Barrido sobre casos                                                          #
# --------------------------------------------------------------------------- #
def _load_signal(row: dict[str, Any]) -> np.ndarray:
    return np.load(row["ecg_signal_path"]).astype(np.float32)


def _iter_intensities(intervention: str) -> list[dict[str, Any]]:
    """Explota el grid de una intervención en una lista de configuraciones,
    cada una con un solo parámetro variable."""
    grid = ECG_INTENSITY_GRID.get(intervention)
    if not grid:
        return [{}]  # sin barrido — usa defaults
    out: list[dict[str, Any]] = []
    for param, values in grid.items():
        for v in values:
            out.append({param: v})
    return out


def calibrate(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    seed_base: int = 42,
) -> dict[str, Any]:
    """Devuelve una tabla ``entries`` con un registro por (caso, intervención, intensidad)
    y un ``summary`` con δ agregado (media, IC 90% empírico) por intervención e intensidad.
    """
    from counterfactual.transformations_ecg import apply_ecg_transform
    from counterfactual.transformations_text import apply_text_transform

    entries: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        signal = _load_signal(row)
        question = str(row.get("question", ""))
        seed = seed_base + idx

        # --- ECG interventions ---
        for name in ECG_INTENSITY_GRID:
            for params in _iter_intensities(name):
                try:
                    perturbed = apply_ecg_transform([signal], name, params=params, seed=seed)[0]
                except Exception as exc:  # pragma: no cover — defensivo
                    entries.append({"case": idx, "modality": "signal", "intervention": name,
                                    "params": params, "error": str(exc)})
                    continue
                perturbed_np = np.asarray(perturbed, dtype=np.float32)
                deltas = delta_ecg(model, signal, perturbed_np)
                entries.append({
                    "case": idx, "modality": "signal",
                    "intervention": name, "params": params, **deltas,
                })

        # --- Text interventions (sin barrido paramétrico) ---
        for name in TEXT_INTERVENTIONS:
            try:
                new_q, _meta = apply_text_transform(question, name, seed=seed)
            except Exception as exc:
                entries.append({"case": idx, "modality": "text", "intervention": name,
                                "error": str(exc)})
                continue
            deltas_t = delta_text(tokenizer, model, question, new_q)
            entries.append({
                "case": idx, "modality": "text",
                "intervention": name, "params": {}, **deltas_t,
            })

        if (idx + 1) % 10 == 0:
            print(f"[calib] {idx + 1}/{len(rows)} casos", flush=True)

    # --- Agregado por (modalidad, intervención, intensidad) ---
    def _key(e: dict[str, Any]) -> str:
        p = e.get("params") or {}
        p_str = ",".join(f"{k}={v}" for k, v in sorted(p.items()))
        return f"{e['modality']}::{e['intervention']}::{p_str}"

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if "error" in e:
            continue
        grouped[_key(e)].append(e)

    def _stat(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": None, "median": None, "p05": None, "p95": None, "n": 0}
        arr = np.array(values, dtype=np.float64)
        return {
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "p05":    float(np.percentile(arr, 5)),
            "p95":    float(np.percentile(arr, 95)),
            "n":      int(arr.size),
        }

    summary: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        modality, intervention, params_str = key.split("::")
        rec = {
            "modality": modality,
            "intervention": intervention,
            "params": params_str,
        }
        # Selecciona el δ representativo por modalidad para la curva CFR(δ):
        # signal -> h_pool (lo que efectivamente entra al proyector).
        # text   -> promedio de embeddings.
        if modality == "signal":
            rec["delta_bigru"] = _stat([i["delta_bigru"] for i in items if "delta_bigru" in i])
            rec["delta_pool"]  = _stat([i["delta_pool"]  for i in items if "delta_pool"  in i])
            rec["delta"] = rec["delta_pool"]
        else:
            rec["delta_text"] = _stat([i["delta_text"] for i in items if "delta_text" in i])
            rec["delta"] = rec["delta_text"]
        summary.append(rec)

    return {"entries": entries, "summary": summary}


# --------------------------------------------------------------------------- #
# Emparejamiento por δ comparable (brief §4.2)                                 #
# --------------------------------------------------------------------------- #
def pair_by_delta(summary: list[dict[str, Any]], factor: float = 2.0) -> list[dict[str, Any]]:
    """Empareja intervenciones textuales y de señal cuyo δ medio caiga dentro de
    un factor multiplicativo ``factor`` entre sí. Devuelve una lista de tríos
    ``{text_intervention, signal_intervention, params, delta_text, delta_signal, ratio}``.
    """
    text_items = [s for s in summary if s["modality"] == "text" and s["delta"]["mean"]]
    sig_items  = [s for s in summary if s["modality"] == "signal" and s["delta"]["mean"]]
    pairs: list[dict[str, Any]] = []
    for t in text_items:
        dt = t["delta"]["mean"]
        for s in sig_items:
            ds = s["delta"]["mean"]
            if dt <= 0 or ds <= 0:
                continue
            ratio = max(dt, ds) / min(dt, ds)
            if ratio <= factor:
                pairs.append({
                    "text_intervention":   t["intervention"],
                    "signal_intervention": s["intervention"],
                    "signal_params":       s["params"],
                    "delta_text":          dt,
                    "delta_signal":        ds,
                    "ratio":               round(ratio, 3),
                })
    # Ordena por ratio (más cercano primero).
    pairs.sort(key=lambda x: x["ratio"])
    return pairs


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
        description="F03.1 · Calibración de intervenciones por desplazamiento representacional δ."
    )
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Checkpoint E1/E2/E3 con encoder (no E0).")
    parser.add_argument("--data", type=Path,
                        default=PROJECT_ROOT / "data/ecgqa_small/processed_test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--factor", type=float, default=2.0,
                        help="Ratio máximo permitido para considerar dos δ 'comparables'.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/audit/delta_calibration.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model, _config = load_checkpoint(args.model_path, device=args.device)
    print(f"[calib] modelo cargado: {args.model_path}")

    rows = _read_rows(args.data, args.max_samples)
    print(f"[calib] casos: {len(rows)}")

    result = calibrate(model, tokenizer, rows, seed_base=args.seed_base)
    pairs = pair_by_delta(result["summary"], factor=args.factor)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.model_path),
        "n_cases": len(rows),
        "factor": args.factor,
        "summary": result["summary"],
        "pairs_by_delta": pairs,
        # Los entries individuales son voluminosos; sólo se guardan si se
        # solicita explícitamente con --output_entries.
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[calib] escrito {args.output}")
    print(f"[calib] pares emparejados por δ (factor={args.factor}): {len(pairs)}")
    for p in pairs[:8]:
        print(f"  {p['text_intervention']:<40}  <->  "
              f"{p['signal_intervention']:<20} {p['signal_params']:<20} "
              f"ratio={p['ratio']:.2f}")


if __name__ == "__main__":
    main()
