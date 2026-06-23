"""Export the small-run results into the D3 visualizer's data file.

Reads the artefacts produced by the ECG-QA small pipeline and writes
``Visualization/ecgqa_viz_data.js`` (a ``window.ECGQA_DATA = {...}`` assignment).
The visualizer loads that file via ``<script src>`` -- which works from a plain
``file://`` double-click, unlike ``fetch`` -- and, when present, replaces its
synthetic demo data with the real run.

Primary source is ``counterfactual_results.jsonl`` (it carries per-variant
predictions + dominance, which feed the intervention table and the dominance
verdict). ``evaluation.jsonl`` is used as a fallback / to widen coverage, and
``processed_test.jsonl`` provides the real ECG waveform for each id.

The dominance shown matches the project definition: ``dTexto = QCFR`` (question
flip), ``dEcg = ECFR`` (mean ECG-perturbation flip), ``D = dTexto/(dTexto+dEcg)``.

Example::

    python scripts/export_visualizer_data.py \\
      --results_dir outputs/ecgqa_small \\
      --processed data/ecgqa_small/processed_test.jsonl \\
      --output Visualization/ecgqa_viz_data.js
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ecgqa_metrics import answer_to_text, normalize  # noqa: E402

N_PATCH = 20            # visualizer patch count (T_LEN / PATCH = 1000 / 50)
PLOT_LEN = 1000         # visualizer T_LEN
EPS = 1e-6

QTYPE_MAP = {
    "single-verify": "verificación",
    "single-choose": "elección",
    "single-query": "consulta",
}


def map_qtype(question_type: str) -> str:
    if question_type.startswith("comparison"):
        return "comparación"
    return QTYPE_MAP.get(question_type, "consulta")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def clean_answer(text: Any) -> str:
    """Short, single-line form of a prediction for option matching/display."""
    s = " ".join(str(text).split())
    if len(s) > 48:
        s = s[:48].rstrip() + "…"
    return s or "(vacío)"


def correctness(prediction: str, gold: str) -> bool:
    pred_n, gold_n = normalize(prediction), normalize(gold)
    return bool(gold_n) and (pred_n == gold_n or gold_n in pred_n)


def signal_for_plot(npy_path: str, lead: int) -> tuple[list[float], list[float]]:
    """Return (waveform[<=PLOT_LEN], featPatch[N_PATCH]) for one ECG file."""
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    lead = min(lead, arr.shape[1] - 1)
    wave = arr[:, lead]
    # Resample to PLOT_LEN points for the SVG line.
    if wave.shape[0] != PLOT_LEN:
        src = np.linspace(0.0, 1.0, wave.shape[0])
        dst = np.linspace(0.0, 1.0, PLOT_LEN)
        wave = np.interp(dst, src, wave)
    # Per-patch saliency proxy = normalised local energy (std) across patches.
    bounds = np.linspace(0, PLOT_LEN, N_PATCH + 1, dtype=int)
    energy = np.array([wave[bounds[p]:bounds[p + 1]].std() for p in range(N_PATCH)], dtype=np.float32)
    lo, hi = float(energy.min()), float(energy.max())
    feat = (energy - lo) / (hi - lo) if hi > lo else np.full(N_PATCH, 0.3, dtype=np.float32)
    feat = 0.1 + 0.85 * feat  # keep within the visualizer's expected 0..1 band
    return [round(float(v), 4) for v in wave], [round(float(v), 4) for v in feat]


def level_for(d: float | None) -> str:
    if d is None:
        return "low"
    return "high" if d > 0.65 else "mid" if d > 0.45 else "low"


INTERVENTION_LABELS = [
    ("original", "Entrada completa"),
    ("question_cf", "Pregunta reescrita (CF)"),
    ("neutral_question", "Pregunta neutra"),
    ("ecg_noise", "ECG + ruido"),
    ("ecg_lead_mask", "ECG enmascarado (derivaciones)"),
    ("ecg_time_mask", "ECG enmascarado (tiempo)"),
    ("ecg_spike", "ECG pico artificial"),
    ("conflict_question_normal_ecg_abnormal", "Texto afirma: normal"),
    ("conflict_question_abnormal_ecg_normal", "Texto afirma: anormal"),
]


def build_interventions(predictions: dict[str, str]) -> list[dict[str, Any]]:
    base = normalize(predictions.get("original", ""))
    rows = []
    for key, label in INTERVENTION_LABELS:
        if key not in predictions:
            continue
        resp = clean_answer(predictions[key])
        flip = key != "original" and normalize(predictions[key]) != base
        rows.append({"c": label, "resp": resp, "flip": flip})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export small-run results for the D3 visualizer")
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_small")
    parser.add_argument("--processed", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed_test.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "Visualization/ecgqa_viz_data.js")
    parser.add_argument("--lead", type=int, default=1, help="ECG lead to plot (default 1 = lead II)")
    parser.add_argument("--max_samples", type=int, default=0, help="Cap exported samples (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cf_rows = load_jsonl(args.results_dir / "counterfactual_results.jsonl")
    eval_rows = load_jsonl(args.results_dir / "evaluation.jsonl")
    processed = {r["id"]: r for r in load_jsonl(args.processed)}
    eval_by_id = {r["id"]: r for r in eval_rows}

    if not cf_rows and not eval_rows:
        raise SystemExit(
            f"No results found in {args.results_dir}. Run the pipeline stages first "
            "(at least evaluate_ecgqa_small.py / run_ecgqa_counterfactual_small.py)."
        )

    # Prefer the counterfactual set (richer); fall back to the evaluation set.
    source_rows = cf_rows if cf_rows else eval_rows
    source_name = "counterfactual_results.jsonl" if cf_rows else "evaluation.jsonl"

    samples: list[dict[str, Any]] = []
    qtypes_seen: list[str] = []
    classes_seen: set[str] = set()
    for row in source_rows:
        if args.max_samples and len(samples) >= args.max_samples:
            break
        rid = row["id"]
        gold = answer_to_text(row.get("answer", ""))
        predictions = row.get("predictions")
        if predictions:  # counterfactual row
            pred_raw = predictions.get("original", "")
            metrics = row.get("metrics", {})
            qcfr = metrics.get("qcfr")
            ecfr = metrics.get("ecfr")
            d_texto = float(qcfr) if qcfr is not None else None
            d_ecg = float(ecfr) if ecfr is not None else None
            interventions = build_interventions(predictions)
        else:  # evaluation-only row
            pred_raw = row.get("prediction", "")
            d_texto = d_ecg = None
            interventions = [{"c": "Entrada completa", "resp": clean_answer(pred_raw), "flip": False}]

        pred = clean_answer(pred_raw)
        # Enrich correctness from the evaluation file when available.
        eval_row = eval_by_id.get(rid)
        if eval_row is not None and "exact_match" in eval_row:
            correct = bool(eval_row["exact_match"]) or correctness(eval_row.get("prediction", pred_raw), gold)
        else:
            correct = correctness(pred_raw, gold)

        d_val = (d_texto / (d_texto + d_ecg + EPS)) if (d_texto is not None and d_ecg is not None) else None
        d_val = round(d_val, 3) if d_val is not None else None

        qtype = map_qtype(row.get("question_type", ""))
        if qtype not in qtypes_seen:
            qtypes_seen.append(qtype)
        cls = clean_answer(gold)
        classes_seen.add(cls)

        proc = processed.get(rid)
        if proc and Path(proc["ecg_signal_path"]).exists():
            wave, feat = signal_for_plot(proc["ecg_signal_path"], args.lead)
        else:
            wave, feat = [0.0] * PLOT_LEN, [0.3] * N_PATCH

        # Option set for the QA panel: gold + model answer (deduped, order-stable).
        options: list[str] = []
        for opt in (cls, pred):
            if opt not in options:
                options.append(opt)

        samples.append({
            "id": len(samples),
            "real_id": rid,
            "ecg_id": ",".join(str(e) for e in row.get("ecg_id", [])) or rid,
            "qtype": qtype,
            "attr": {
                "name": row.get("attribute_type", "") or "atributo",
                "cls": cls,
                "feat": row.get("attribute_type", "") or "—",
                "k": row.get("attribute_type", ""),
            },
            "question": row.get("question", ""),
            "options": options,
            "gt": cls,
            "pred": pred,
            "correct": correct,
            "dTexto": d_texto,
            "dEcg": d_ecg,
            "D": d_val,
            "level": level_for(d_val),
            "interventions": interventions,
            "textImplied": pred,
            "sig": {"arr": wave, "featPatch": feat},
        })

    payload = {
        "meta": {
            "source": "ecgqa_small",
            "from": source_name,
            "n": len(samples),
            "lead": args.lead,
            "note": f"Datos reales · {len(samples)} muestras ECG-QA · fuente: {source_name}",
        },
        "globals": {
            "evaluation": load_json(args.results_dir / "evaluation_summary.json").get("global", {}),
            "counterfactual": load_json(args.results_dir / "counterfactual_summary.json").get("global", {}),
        },
        "qtypes": qtypes_seen,
        "classes": sorted(classes_seen),
        "configs": ["completo"],
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False)
    args.output.write_text(f"window.ECGQA_DATA = {body};\n", encoding="utf-8")
    # A plain JSON sibling for any other consumer.
    args.output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "samples": len(samples),
        "source": source_name,
        "qtypes": qtypes_seen,
        "classes": len(classes_seen),
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
