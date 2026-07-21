"""Trazado representacional interno (modal sensitivity tracing) para ECG-QA.

Esto NO es backpropagation ni atribucion causal. Es un **trazado representacional
interno**: para cada caso se corre el modelo sobre cinco condiciones y se mide,
*por etapa del flujo multimodal*, cuanto se mueve la representacion interna cuando
se interviene el texto (Original vs Text-CF) frente a cuando se interviene el ECG
(Original vs ECG-CF), usando distancias representacionales simples y estables
(distancia coseno o L2 normalizada).

Condiciones por caso::

    Original : ECG + pregunta
    Text-CF  : ECG + pregunta reescrita (meaning-preserving)
    ECG-CF   : ECG perturbado + pregunta original
    Neutral  : ECG + pregunta neutral (sin pista clinica)
    Conflict : ECG/pregunta contradictorios

Etapas trazadas (resumen por etapa)::

    encoder    -> salida del encoder temporal (tokens temporales)
    projector  -> salida del proyector latente (espacio del LLM)
    fusion     -> inputs_embeds (concatenacion temporal + texto)
    llm        -> hidden_states del LLM (por capa y agrupadas)
    output     -> logits / respuesta final

Por etapa se calcula::

    impacto_texto       = dist(original, Text-CF)     (None en encoder/projector:
                                                       el texto no atraviesa esas etapas)
    impacto_ECG         = dist(original, ECG-CF)
    diferencia_texto_ECG = impacto_texto - impacto_ECG

Si las activaciones internas no estan disponibles (``--output-only`` o un fallo
al capturar hooks), el modulo cae a modo ``contrafactual_global``: solo se reporta
la sensibilidad de salida (flips de la respuesta) y las etapas internas quedan en
``None``. Nunca se inventan valores internos.

Importante sobre el lenguaje: el sesgo NO "nace" en una capa. Se afirma que el
sesgo **se manifiesta o se amplifica** en una etapa.

Salidas::

    outputs/tracing/representational_tracing.jsonl   (un caso por linea)
    outputs/tracing/stage_sensitivity_summary.json   (agregado + diagnostico)
    visualizer/tracing_data.js                        (window.TRACING_DATA, opcional)

Ejemplo::

    python xai/representational_tracing.py \\
      --model_path checkpoints/ecgqa_small_lora \\
      --data data/ecgqa_small/processed_test.jsonl \\
      --max_samples 30 \\
      --metric cosine
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EPS = 1e-8

# Etapas en orden del flujo. ``text_flows`` indica si el texto atraviesa la etapa
# (en encoder/projector solo circula el ECG, asi que impacto_texto = None ahi).
STAGES: tuple[tuple[str, bool], ...] = (
    ("encoder", False),
    ("projector", False),
    ("fusion", True),
    ("llm", True),
    ("output", True),
)
STAGE_LABELS = {
    "encoder": "Encoder temporal",
    "projector": "Proyector latente",
    "fusion": "Fusion inputs_embeds",
    "llm": "LLM",
    "output": "Logits / Respuesta",
}

# Clasificacion de dominancia por caso. Escala-libre: se trabaja sobre la cuota
# textual share = impacto_texto / (impacto_texto + impacto_ECG) en la etapa de
# salida, mas un umbral absoluto para detectar casos INSENSITIVE.
TRACING_THRESHOLDS = {
    "inactive": 0.015,  # suma de impactos de salida por debajo -> INSENSITIVE
    "margin": 0.08,     # |share - 0.5| necesario para declarar dominancia
}
DOMINANCE_CLASSES = ("TEXT_DOMINANT", "ECG_DOMINANT", "BALANCED", "INSENSITIVE", "UNCLEAR")


# --------------------------------------------------------------------------- #
# Distancias representacionales (puras, sin torch)
# --------------------------------------------------------------------------- #
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - similitud coseno. Estable: 0 si algun vector es ~nulo."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.clip(1.0 - float(np.dot(a, b) / (na * nb)), 0.0, 2.0))


def l2_normalized_distance(a: np.ndarray, b: np.ndarray) -> float:
    """||a-b|| normalizada por (||a||+||b||): invariante a la escala global."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    if denom < EPS:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def get_metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "cosine":
        return cosine_distance
    if name == "l2":
        return l2_normalized_distance
    raise ValueError(f"Metrica desconocida: {name!r} (use 'cosine' o 'l2')")


# --------------------------------------------------------------------------- #
# Clasificacion de dominancia por caso (pura)
# --------------------------------------------------------------------------- #
def text_share(impact_text: float | None, impact_ecg: float | None) -> float | None:
    """Cuota textual de la sensibilidad de salida en [0, 1] (None si indefinido)."""
    if impact_text is None or impact_ecg is None:
        return None
    total = impact_text + impact_ecg
    if total < EPS:
        return 0.5
    return float(impact_text / total)


def classify_case(
    impact_text: float | None,
    impact_ecg: float | None,
    any_flip: bool,
    thresholds: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Devuelve (clase_de_dominancia, razon legible) para la etapa de salida.

    ``any_flip`` indica si alguna intervencion (texto o ECG) cambio la respuesta
    discreta; ayuda a separar INSENSITIVE de movimientos representacionales
    pequenos pero reales.
    """
    t = thresholds or TRACING_THRESHOLDS
    if impact_text is None or impact_ecg is None:
        return "UNCLEAR", "sin sensibilidad de salida comparable"
    total = impact_text + impact_ecg
    if total < t["inactive"] and not any_flip:
        return "INSENSITIVE", (
            f"ni el texto ni el ECG mueven la salida (it={impact_text:.3f}, ie={impact_ecg:.3f})"
        )
    share = impact_text / total if total > EPS else 0.5
    gap = share - 0.5
    if gap >= t["margin"]:
        return "TEXT_DOMINANT", f"la intervencion de texto domina (share_texto={share:.2f})"
    if -gap >= t["margin"]:
        return "ECG_DOMINANT", f"la intervencion del ECG domina (share_texto={share:.2f})"
    return "BALANCED", f"texto y ECG pesan de forma comparable (share_texto={share:.2f})"


def text_dominance_level(share: float | None) -> str:
    """Bucket alta/media/baja para el filtro lateral del visualizador."""
    if share is None:
        return "baja"
    if share >= 0.60:
        return "alta"
    if share >= 0.45:
        return "media"
    return "baja"


# --------------------------------------------------------------------------- #
# PCA 2D (pura, para el espacio latente V3)
# --------------------------------------------------------------------------- #
def waveform_summary(signal: np.ndarray, lead: int = 1, plot_len: int = 500,
                     n_patch: int = 20) -> dict[str, Any]:
    """Onda ECG remuestreada + energia por parche, para el panel V4 del visualizador."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    lead = min(lead, arr.shape[1] - 1)
    wave = arr[:, lead]
    if wave.shape[0] != plot_len:
        src = np.linspace(0.0, 1.0, wave.shape[0])
        dst = np.linspace(0.0, 1.0, plot_len)
        wave = np.interp(dst, src, wave)
    bounds = np.linspace(0, plot_len, n_patch + 1, dtype=int)
    energy = np.array([wave[bounds[p]:bounds[p + 1]].std() for p in range(n_patch)], dtype=np.float32)
    lo, hi = float(energy.min()), float(energy.max())
    feat = (energy - lo) / (hi - lo) if hi > lo else np.full(n_patch, 0.3, dtype=np.float32)
    feat = 0.1 + 0.85 * feat
    return {
        "arr": [round(float(v), 4) for v in wave],
        "featPatch": [round(float(v), 4) for v in feat],
    }


def pca_2d(points: np.ndarray) -> np.ndarray:
    """Proyeccion PCA a 2D, centrada y escalada a un rango comodo para graficar."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros((0, 2))
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        comp = vt[:2]
        if comp.shape[0] < 2:
            comp = np.vstack([comp, np.zeros((2 - comp.shape[0], comp.shape[1]))])
        coords = centered @ comp.T
    except np.linalg.LinAlgError:
        coords = centered[:, :2]
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    scale = float(np.abs(coords).max()) or 1.0
    return coords / scale


# --------------------------------------------------------------------------- #
# Resumen agregado por etapa + diagnostico
# --------------------------------------------------------------------------- #
def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def stage_summary(cases: list[dict[str, Any]], mode: str, metric: str) -> dict[str, Any]:
    """Agrega impactos por etapa y redacta un diagnostico (manifestacion/amplificacion)."""
    stages: dict[str, Any] = {}
    for name, text_flows in STAGES:
        it = _mean([c["stages"][name]["impact_text"] for c in cases])
        ie = _mean([c["stages"][name]["impact_ecg"] for c in cases])
        delta = (it - ie) if (it is not None and ie is not None) else None
        stages[name] = {
            "label": STAGE_LABELS[name],
            "mean_impact_text": None if it is None else round(it, 4),
            "mean_impact_ecg": None if ie is None else round(ie, 4),
            "mean_delta_text_ecg": None if delta is None else round(delta, 4),
            "text_flows": text_flows,
        }

    # Donde se AMPLIFICA el desbalance texto-ECG: mayor salto positivo de delta
    # entre etapas consecutivas donde delta esta definido (fusion -> llm -> output).
    defined = [(n, stages[n]["mean_delta_text_ecg"]) for n, _ in STAGES
               if stages[n]["mean_delta_text_ecg"] is not None]
    amplification: dict[str, Any] = {"stage": None, "jump": None, "note": ""}
    if defined:
        first_stage = defined[0][0]
        max_jump, max_stage = 0.0, None
        for (prev_name, prev_d), (curr_name, curr_d) in zip(defined, defined[1:]):
            jump = curr_d - prev_d
            if jump > max_jump:
                max_jump, max_stage = jump, curr_name
        target = max_stage or defined[-1][0]
        amplification = {
            "stage": target,
            "jump": round(max_jump, 4),
            "note": (
                f"El desbalance texto-ECG se manifiesta desde {STAGE_LABELS[first_stage]} "
                f"y se amplifica en {STAGE_LABELS[target]}. "
                "No se afirma que el sesgo nazca en esa etapa."
            ),
        }

    # Diagnostico textual por etapa (cuidando el lenguaje).
    for name, info in stages.items():
        delta = info["mean_delta_text_ecg"]
        if delta is None:
            info["diagnostico"] = (
                "Solo circula el ECG: se reporta impacto_ECG; impacto_texto no aplica."
            )
        else:
            if delta > TRACING_THRESHOLDS["margin"]:
                tone = "predomina la sensibilidad al texto"
            elif delta < -TRACING_THRESHOLDS["margin"]:
                tone = "predomina la sensibilidad al ECG"
            else:
                tone = "sensibilidad texto/ECG comparable"
            extra = " (etapa donde se amplifica el desbalance)" if name == amplification["stage"] else ""
            info["diagnostico"] = (
                f"En esta etapa {tone} (delta={delta:+.3f}); el sesgo se manifiesta o se amplifica aqui{extra}."
            )

    counts = Counter(c["case_dominance"] for c in cases)
    return {
        "mode": mode,
        "metric": metric,
        "count": len(cases),
        "thresholds": TRACING_THRESHOLDS,
        "stages": stages,
        "amplification": amplification,
        "class_counts": {k: counts.get(k, 0) for k in DOMINANCE_CLASSES},
        "class_fractions": (
            {k: counts.get(k, 0) / len(cases) for k in DOMINANCE_CLASSES} if cases else {}
        ),
    }


# --------------------------------------------------------------------------- #
# Parte que ejecuta el modelo (torch importado de forma diferida)
# --------------------------------------------------------------------------- #
def _pool_seq(tensor: "np.ndarray") -> np.ndarray:
    """[1, n, d] o [1, d] -> vector [d] promediando sobre la dimension de tokens."""
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 3:
        return arr[0].mean(axis=0)
    if arr.ndim == 2:
        return arr[0]
    return arr.ravel()


def _capture_activations(model, tokenizer, config, question: str, signal: np.ndarray) -> dict[str, Any]:
    """Una pasada hacia adelante (sin gradiente) capturando resumenes por etapa.

    Usa forward hooks del modelo sobre el encoder temporal y el proyector latente,
    y ``output_hidden_states=True`` para los estados ocultos del LLM. No hay
    backpropagation: todo corre bajo ``inference_mode``.
    """
    import torch

    from inference.runtime import build_ecg_inputs  # noqa: E402

    example = {"question": question, "ecg_signal": [np.asarray(signal, dtype=np.float32).tolist()]}
    input_ids, attention_mask, series, time_mask = build_ecg_inputs(tokenizer, example, config["input_dim"])

    captured: dict[str, Any] = {}

    def enc_hook(_module, _inp, out):
        captured["encoder"] = out[0].detach().float().cpu().numpy()

    def proj_hook(_module, _inp, out):
        captured["projector"] = out.detach().float().cpu().numpy()

    h1 = model.time_series_encoder.register_forward_hook(enc_hook)
    h2 = model.temporal_projector.register_forward_hook(proj_hook)
    try:
        with torch.inference_mode():
            inputs_embeds, combined_mask, _ = model.encode_modalities(
                input_ids, attention_mask, series, time_mask
            )
            out = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_mask,
                output_hidden_states=True,
            )
            fusion = inputs_embeds.detach().float().cpu().numpy()
            hidden = [h.detach().float().cpu().numpy() for h in out.hidden_states]
            logits = out.logits[:, -1].detach().float().cpu().numpy()
            text_embeds = model.llm.get_input_embeddings()(input_ids.to(model.input_device))
            text_embeds = text_embeds.detach().float().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    if "encoder" not in captured or "projector" not in captured:
        raise RuntimeError("No se capturaron las activaciones del encoder/proyector")

    return {
        "encoder_vec": _pool_seq(captured["encoder"]),
        "projector_vec": _pool_seq(captured["projector"]),
        "projector_tokens": captured["projector"][0],   # [K, H] para V3
        "fusion_vec": _pool_seq(fusion),
        "hidden_layers": [_pool_seq(h) for h in hidden],  # lista de [H] por capa
        "llm_vec": _pool_seq(hidden[-1]),
        "logits_vec": logits.ravel(),
        "text_tokens": text_embeds[0],                  # [seq, H] para V3
        "input_ids": input_ids[0].tolist(),
    }


def _build_latent(orig: dict[str, Any], ecgcf: dict[str, Any], textcf: dict[str, Any],
                  tokenizer, max_text_tokens: int = 12) -> dict[str, Any]:
    """Espacio latente 2D (V3): tokens de pregunta/opciones + ECG original/perturbado.

    Se ajusta un PCA conjunto sobre original y contrafactual para que las flechas
    de desplazamiento (ECG orig->pert, pregunta orig->modificada) vivan en el mismo
    plano.
    """
    text_tok = orig["text_tokens"]
    n_text = min(max_text_tokens, text_tok.shape[0])
    # Tomar los ultimos tokens (donde suele caer la pregunta tras el chat template).
    text_sel = text_tok[-n_text:]
    text_ids = orig["input_ids"][-n_text:]
    text_words = [tokenizer.decode([tid]).strip() or "·" for tid in text_ids]

    textcf_tok = textcf["text_tokens"]
    nt = min(max_text_tokens, textcf_tok.shape[0])
    textcf_sel = textcf_tok[-nt:]

    ecg_orig = orig["projector_tokens"]   # [K, H]
    ecg_pert = ecgcf["projector_tokens"]  # [K, H]

    blocks = [text_sel, textcf_sel, ecg_orig, ecg_pert]
    stacked = np.vstack(blocks)
    coords = pca_2d(stacked)

    idx = 0
    points: list[dict[str, Any]] = []
    arrows: list[dict[str, Any]] = []

    text_coords = coords[idx:idx + n_text]; idx += n_text
    for w, xy in zip(text_words, text_coords):
        points.append({"label": w, "kind": "texto", "x": round(float(xy[0]), 4), "y": round(float(xy[1]), 4)})

    textcf_coords = coords[idx:idx + nt]; idx += nt
    # Una flecha de centroide: pregunta original -> pregunta modificada.
    if n_text and nt:
        c0 = text_coords.mean(axis=0)
        c1 = textcf_coords.mean(axis=0)
        arrows.append({
            "kind": "texto",
            "from": [round(float(c0[0]), 4), round(float(c0[1]), 4)],
            "to": [round(float(c1[0]), 4), round(float(c1[1]), 4)],
        })

    k = ecg_orig.shape[0]
    ecg_o_coords = coords[idx:idx + k]; idx += k
    ecg_p_coords = coords[idx:idx + k]; idx += k
    for j in range(k):
        points.append({"label": f"ECG t{j} (orig)", "kind": "ecg_orig",
                       "x": round(float(ecg_o_coords[j][0]), 4), "y": round(float(ecg_o_coords[j][1]), 4)})
        points.append({"label": f"ECG t{j} (pert)", "kind": "ecg_pert",
                       "x": round(float(ecg_p_coords[j][0]), 4), "y": round(float(ecg_p_coords[j][1]), 4)})
        arrows.append({
            "kind": "ecg",
            "from": [round(float(ecg_o_coords[j][0]), 4), round(float(ecg_o_coords[j][1]), 4)],
            "to": [round(float(ecg_p_coords[j][0]), 4), round(float(ecg_p_coords[j][1]), 4)],
        })
    return {"points": points, "arrows": arrows}


def trace_case(model, tokenizer, config, row: dict[str, Any], metric_fn,
               seed: int, max_new_tokens: int, ecg_segments: int,
               output_only: bool, ecg_transform: str = "ecg_cf_time_mask") -> dict[str, Any]:
    """Traza un caso completo sobre las 5 condiciones y devuelve su registro."""
    from counterfactual.transformations_ecg import apply_ecg_transform, mask_time
    from counterfactual.transformations_text import apply_text_transform
    from inference.runtime import predict_ecg
    from scripts.ecgqa_metrics import answer_to_text, normalize

    question = row["question"]
    attribute_type = row.get("attribute_type", "")
    signal = np.load(row["ecg_signal_path"]).astype(np.float32)

    # --- construir condiciones ---
    q_textcf, _ = apply_text_transform(question, "question_cf", attribute_type=attribute_type, seed=seed)
    q_neutral, _ = apply_text_transform(question, "neutral_question", attribute_type=attribute_type, seed=seed)
    q_conflict, _ = apply_text_transform(
        question, "conflict_question_normal_ecg_abnormal", attribute_type=attribute_type, seed=seed
    )
    # La intervencion por defecto es la oclusion temporal, no el ruido gaussiano:
    # el pooling atencional del encoder promedia el ruido de media cero y lo
    # cancela, de modo que medir con el subestima el impacto del ECG por
    # construccion (seccion 9 de RESULTADOS_CORRIDA_A.txt).
    signal_ecgcf = np.asarray(
        apply_ecg_transform([signal.tolist()], ecg_transform, seed=seed)[0], dtype=np.float32
    )

    def gen(q: str, sig: np.ndarray, use_series: bool = True) -> str:
        return predict_ecg(
            tokenizer, model, config,
            {"question": q, "ecg_signal": [sig.tolist()]},
            max_new_tokens=max_new_tokens,
            use_series=use_series,
        )

    predictions = {
        "original": gen(question, signal),
        "text_cf": gen(q_textcf, signal),
        "ecg_cf": gen(question, signal_ecgcf),
        "neutral": gen(q_neutral, signal),
        "conflict": gen(q_conflict, signal),
        # Escenario "Sin ECG": se elimina toda la modalidad temporal (offline, no en vivo).
        "no_ecg": gen(question, signal, use_series=False),
    }
    base = normalize(predictions["original"])
    flips = {k: (normalize(v) != base) for k, v in predictions.items() if k != "original"}
    any_flip = any(flips.values())

    # --- etapas internas ---
    stages: dict[str, Any] = {}
    llm_layers = {"impact_text": None, "impact_ecg": None}
    latent: dict[str, Any] | None = None
    mode = "contrafactual_global"

    if not output_only:
        try:
            act_orig = _capture_activations(model, tokenizer, config, question, signal)
            act_textcf = _capture_activations(model, tokenizer, config, q_textcf, signal)
            act_ecgcf = _capture_activations(model, tokenizer, config, question, signal_ecgcf)
            mode = "internal"
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] sin activaciones internas para {row['id']}: {exc}", flush=True)
            act_orig = act_textcf = act_ecgcf = None

        if act_orig is not None:
            def stage_impacts(key_vec: str, text_flows: bool) -> dict[str, Any]:
                ie = metric_fn(act_orig[key_vec], act_ecgcf[key_vec])
                it = metric_fn(act_orig[key_vec], act_textcf[key_vec]) if text_flows else None
                delta = (it - ie) if it is not None else None
                return {
                    "impact_ecg": round(float(ie), 4),
                    "impact_text": None if it is None else round(float(it), 4),
                    "delta_text_ecg": None if delta is None else round(float(delta), 4),
                }

            stages["encoder"] = stage_impacts("encoder_vec", text_flows=False)
            stages["projector"] = stage_impacts("projector_vec", text_flows=False)
            stages["fusion"] = stage_impacts("fusion_vec", text_flows=True)
            stages["llm"] = stage_impacts("llm_vec", text_flows=True)
            stages["output"] = stage_impacts("logits_vec", text_flows=True)

            # Por capa del LLM (hidden_states agrupados como serie).
            n_layers = min(len(act_orig["hidden_layers"]), len(act_ecgcf["hidden_layers"]),
                           len(act_textcf["hidden_layers"]))
            llm_layers = {
                "impact_text": [round(float(metric_fn(act_orig["hidden_layers"][i],
                                                      act_textcf["hidden_layers"][i])), 4)
                                for i in range(n_layers)],
                "impact_ecg": [round(float(metric_fn(act_orig["hidden_layers"][i],
                                                     act_ecgcf["hidden_layers"][i])), 4)
                               for i in range(n_layers)],
            }
            latent = _build_latent(act_orig, act_ecgcf, act_textcf, tokenizer)

    # En modo contrafactual_global no hay etapas internas: solo salida via flips.
    if mode == "contrafactual_global" or not stages:
        for name, text_flows in STAGES:
            if name == "output":
                it = 1.0 if flips["text_cf"] else 0.0
                ie = 1.0 if flips["ecg_cf"] else 0.0
                stages[name] = {
                    "impact_ecg": ie,
                    "impact_text": it,
                    "delta_text_ecg": round(it - ie, 4),
                }
            else:
                stages[name] = {"impact_ecg": None, "impact_text": None, "delta_text_ecg": None}

    out_it = stages["output"]["impact_text"]
    out_ie = stages["output"]["impact_ecg"]
    dominance, reason = classify_case(out_it, out_ie, any_flip)
    share = text_share(out_it, out_ie)

    # --- V4: evidencia local por intervencion ---
    interventions: list[dict[str, Any]] = [
        {"label": "Pregunta reescrita (CF)", "kind": "texto", "detail": q_textcf,
         "answer": predictions["text_cf"], "changed": flips["text_cf"]},
        {"label": "Pregunta neutral", "kind": "texto", "detail": q_neutral,
         "answer": predictions["neutral"], "changed": flips["neutral"]},
        {"label": "Texto en conflicto", "kind": "texto", "detail": q_conflict,
         "answer": predictions["conflict"], "changed": flips["conflict"]},
        {"label": "ECG perturbado (ruido)", "kind": "ecg", "detail": "ruido gaussiano por derivacion",
         "answer": predictions["ecg_cf"], "changed": flips["ecg_cf"]},
    ]
    for seg in range(ecg_segments):
        start = seg / max(ecg_segments, 1)
        masked = mask_time(signal, fraction=1.0 / max(ecg_segments, 1), start=start, seed=seed)
        ans = gen(question, np.asarray(masked, dtype=np.float32))
        interventions.append({
            "label": f"ECG segmento {seg + 1}/{ecg_segments}",
            "kind": "ecg",
            "segment": [round(start, 4), round(start + 1.0 / max(ecg_segments, 1), 4)],
            "detail": f"ventana temporal [{start:.2f}, {start + 1.0 / ecg_segments:.2f}] enmascarada",
            "answer": ans,
            "changed": normalize(ans) != base,
        })

    return {
        "case_id": row["id"],
        "question_type": row.get("question_type", ""),
        "attribute_type": row.get("attribute_type", ""),
        "question": question,
        "expected": answer_to_text(row.get("answer", "")),
        "prediction_original": predictions["original"],
        "prediction_text_cf": predictions["text_cf"],
        "prediction_ecg_cf": predictions["ecg_cf"],
        "prediction_neutral": predictions["neutral"],
        "prediction_conflict": predictions["conflict"],
        "prediction_no_ecg": predictions["no_ecg"],
        "flips": flips,
        "stages": stages,
        "llm_layers": llm_layers,
        "case_dominance": dominance,
        "dominance_reason": reason,
        "text_share": None if share is None else round(share, 4),
        "text_dominance_level": text_dominance_level(share),
        "metric": None,  # rellenado por el caller
        "mode": mode,
        "any_flip": any_flip,
        "latent": latent,
        "interventions": interventions,
        # Onda original + onda intervenida (ECG-CF ruido), para la comparacion
        # contrafactual local de V4 (todo precomputado, no en vivo).
        #
        # altered_segments marca lo que altera el ESCENARIO activo. El ECG-CF aplica
        # ruido gaussiano sobre toda la senal, asi que no hay un tramo concreto que
        # marcar y queda vacio. Antes se rellenaba con las ventanas de occlusion, que
        # cubren el 100% del registro: V4 las pintaba todas en rojo y tapaba la onda
        # entera sin transmitir informacion. Las ventanas de occlusion viven en
        # `interventions` (kind="ecg" con `segment`) y el visualizador las lee de ahi.
        "sig": {
            **waveform_summary(signal),
            "arr_cf": waveform_summary(signal_ecgcf)["arr"],
            "altered_segments": [],
        },
    }


# --------------------------------------------------------------------------- #
# Exportacion para el visualizador
# --------------------------------------------------------------------------- #
def build_viz_payload(cases: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    qtypes = sorted({c["question_type"] for c in cases if c["question_type"]})
    attrs = sorted({c["attribute_type"] for c in cases if c["attribute_type"]})
    mode = summary["mode"]
    note = (
        f"Trazado representacional interno · {len(cases)} casos · metrica {summary['metric']}"
        if mode == "internal"
        else f"Contrafactual global · {len(cases)} casos · solo sensibilidad de salida"
    )
    with_attention = [c for c in cases if c.get("encoder_attention", {}).get("entropy")]
    attention_meta = None
    if with_attention:
        entropies = [e for c in with_attention for e in c["encoder_attention"]["entropy"]]
        bins = with_attention[0]["encoder_attention"].get("bins") or 1
        attention_meta = {
            "n": len(with_attention),
            "bins": bins,
            "uniform": 1.0 / bins,
            "mean_entropy": sum(entropies) / len(entropies),
            "min_entropy": min(entropies),
        }
    return {
        "meta": {"mode": mode, "metric": summary["metric"], "n": len(cases), "note": note,
                 "attention": attention_meta},
        "thresholds": TRACING_THRESHOLDS,
        "stage_summary": summary["stages"],
        "amplification": summary["amplification"],
        "class_counts": summary["class_counts"],
        "classes": list(DOMINANCE_CLASSES),
        "qtypes": qtypes,
        "attributes": attrs,
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trazado representacional interno (modal sensitivity)")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed_test.jsonl")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--ecg_transform",
        default="ecg_cf_time_mask",
        choices=["ecg_cf_time_mask", "ecg_cf_lead_mask", "ecg_cf_noise", "ecg_cf_spike",
                 "ecg_cf_scaling", "ecg_cf_shuffle"],
        help="intervencion de ECG del escenario ecg_cf; el ruido gaussiano se "
             "cancela en el pooling del encoder y subestima el impacto",
    )
    parser.add_argument("--ecg_segments", type=int, default=6,
                        help="Intervenciones de segmento ECG por caso para V4 (0 = desactivar)")
    parser.add_argument("--output_only", action="store_true",
                        help="No capturar activaciones internas: modo contrafactual global")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/tracing/representational_tracing.jsonl")
    parser.add_argument("--summary_output", type=Path,
                        default=PROJECT_ROOT / "outputs/tracing/stage_sensitivity_summary.json")
    parser.add_argument("--viz_output", type=Path, default=PROJECT_ROOT / "visualizer/tracing_data.js",
                        help="Archivo window.TRACING_DATA para el visualizador (vacio = omitir)")
    parser.add_argument("--encoder_attention", type=Path,
                        help="Salida de xai/attention_export.py; adjunta a cada caso el perfil "
                             "de atencion temporal del encoder para la vista V6")
    parser.add_argument("--attributions", type=Path,
                        default=PROJECT_ROOT / "outputs/ecgqa_small/attributions.jsonl",
                        help="Saliencia por gradiente (compute_attributions_small.py) para los pesos "
                             "texto/ECG de la vista V4; opcional, no es backprop del trazado")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no_quantization", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from inference.runtime import load_checkpoint

    metric_fn = get_metric(args.metric)
    tokenizer, model, config = load_checkpoint(args.model_path, not args.no_quantization, device=args.device)

    # Pesos texto/ECG para V4: saliencia por gradiente ya calculada por la Etapa 7b
    # (compute_attributions_small.py). Es opcional y se *lee* de disco; el trazado
    # en si no hace backpropagation.
    attributions: dict[str, Any] = {}
    if args.attributions and args.attributions.exists():
        with args.attributions.open(encoding="utf-8") as af:
            for line in af:
                if line.strip():
                    rec = json.loads(line)
                    attributions[rec["id"]] = rec
        print(f"Saliencia por gradiente cargada para V4: {len(attributions)} casos", flush=True)

    # Atencion temporal del encoder (xai/attention_export.py). Responde, por caso,
    # que parte de la serie mira el modelo; sin ella la vista V6 queda oculta.
    encoder_attention: dict[str, Any] = {}
    if args.encoder_attention and args.encoder_attention.exists():
        with args.encoder_attention.open(encoding="utf-8") as af:
            for line in af:
                if line.strip():
                    rec = json.loads(line)
                    encoder_attention[str(rec.get("id"))] = rec
        print(f"Atencion del encoder cargada para V6: {len(encoder_attention)} casos", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    count = 0
    with args.output.open("w", encoding="utf-8") as handle, args.data.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if args.max_samples and count >= args.max_samples:
                break
            row = json.loads(line)
            case = trace_case(
                model, tokenizer, config, row, metric_fn,
                seed=args.seed, max_new_tokens=args.max_new_tokens,
                ecg_segments=args.ecg_segments, output_only=args.output_only,
                ecg_transform=args.ecg_transform,
            )
            case["metric"] = args.metric
            # Fusionar pesos de relevancia para V4 (texto y serie temporal).
            attr = attributions.get(row["id"])
            if attr:
                if attr.get("token_saliency"):
                    case["tokens"] = attr["token_saliency"]            # pesos del texto
                    case["tokens_source"] = "saliencia por gradiente"
                if attr.get("ecg_patch_saliency"):
                    case["sig"]["featPatch"] = attr["ecg_patch_saliency"]  # pesos de la serie temporal
                    case["sig"]["feat_source"] = "saliencia por gradiente"
            else:
                case.setdefault("sig", {})["feat_source"] = "proxy de energia"
            att = encoder_attention.get(str(row.get("id")))
            if att:
                case["encoder_attention"] = {
                    "bins": att.get("bins"),
                    "steps": att.get("steps"),
                    "tokens": att.get("tokens"),
                    "profile": att.get("mass_profile"),
                    "per_token": att.get("attention_binned"),
                    "entropy": att.get("token_entropy"),
                }
            cases.append(case)
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            out = case["stages"]["output"]
            print(f"[{count}] {row['id']} {case['case_dominance']} "
                  f"it={out['impact_text']} ie={out['impact_ecg']} ({case['mode']})", flush=True)

    mode = "internal" if any(c["mode"] == "internal" for c in cases) else "contrafactual_global"
    summary = stage_summary(cases, mode=mode, metric=args.metric)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if str(args.viz_output):
        payload = build_viz_payload(cases, summary)
        args.viz_output.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(payload, ensure_ascii=False)
        args.viz_output.write_text(f"window.TRACING_DATA = {body};\n", encoding="utf-8")
        args.viz_output.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print("\n" + json.dumps({
        "cases": len(cases),
        "mode": mode,
        "metric": args.metric,
        "class_counts": summary["class_counts"],
        "amplification": summary["amplification"]["stage"],
        "output": str(args.output),
        "summary": str(args.summary_output),
        "viz": str(args.viz_output) if str(args.viz_output) else None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
