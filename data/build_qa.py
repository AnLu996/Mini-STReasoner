"""F04.1 · Construcción del conjunto controlado F1-F5 sobre PTB-XL / PTB-XL+.

Materializa el diseño del brief (§3): cinco familias de pregunta que varían dos
ejes ortogonales — localidad temporal (puntual, corta, global, doble) y
especificidad de derivación (I, II, V1..V6, o global) — de modo que cada
respuesta sea *decidible desde la señal* y tenga un intervalo de evidencia
conocido. Sin esta anotación de span, la tarea T4 (anclaje temporal) no puede
reportarse a nivel de caso; con ella, la Vista V4 del visualizador se activa
con anotación gold.

Fuentes:

- **PTB-XL** [Wagner et al. 2020] — 21 799 registros de 10 s × 12 derivaciones,
  con folds oficiales disjuntos por paciente (`ptbxl_database.csv`,
  `records500/`).
- **PTB-XL+** [Strodthoff et al. 2023] — dos algoritmos de delineación (12SL y
  ECGDeli) con on/offsets por derivación (`labels/12sl_features.csv` y
  `labels/ecgdeli_features.csv`) y anotaciones fiduciales wfdb.
- **PTB-XL Soft Segmentations** [Zenodo 7610236] — máscaras de delineación en
  formato numpy, alternativa sin dependencia de MATLAB.

Salida (JSONL, un ítem por línea)::

    {
      "ecg_id":             12345,
      "family":             "F1",
      "question":           "¿La duración del QRS en V1 supera los 120 ms?",
      "answer":             "sí",
      "span":               [2.05, 2.19],
      "leads_gold":         [6],
      "condition":          "balanced",
      "swap_partner":       67890,          # emparejamiento contrafactual
      "ecg_signal_path":    "data/ptbxl/records500/12000/12345_hr.npy"
    }

Requisitos externos:

    pip install pandas wfdb tqdm

Uso típico (Ubuntu)::

    python data/build_qa.py \\
      --ptbxl_root data/ptbxl \\
      --ptbxlplus_root data/ptbxlplus \\
      --masks_root data/ptbxl_masks \\
      --output_dir data/qa_controlled \\
      --seed 42

Este script **no** entrena nada. Sólo lee tablas, aplica filtros y escribe
JSONL. El coste dominante es leer los .hea/.dat de PTB-XL para convertir a
.npy, pero se hace una única vez (~30 min sobre SSD, ~2 h sobre HDD).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Descripción declarativa de las cinco familias                             #
# --------------------------------------------------------------------------- #
# Cada familia define: cómo se computa la respuesta gold desde los features
# de PTB-XL+, qué span temporal representa la evidencia, qué derivación(es)
# son las de gold, y cómo se plantilla la pregunta. Cambiar una familia aquí
# se propaga automáticamente al balance, al emparejamiento y al export.

@dataclass
class Family:
    fid: str
    label: str
    question_template: str
    lead_gold: tuple[int, ...]        # 0-based indices en el orden estándar de PTB-XL
    locality: str                      # "puntual" | "corta" | "global" | "doble"
    answer_type: str                   # "sí/no" (por ahora, todas binarias)

    # Cómputo del gold desde el registro fiducial. Es una función que recibe el
    # dict de features de un ecg_id (ya con la información concordada 12SL/ECGDeli)
    # y devuelve (answer, span_seconds), donde span es (t0, t1).
    # None → skip (no aplicable a ese registro).
    compute_gold: Any = None


# Índices de derivaciones estándar en PTB-XL (orden estable):
#   0:I  1:II  2:III  3:aVR  4:aVL  5:aVF  6:V1  7:V2  8:V3  9:V4  10:V5  11:V6
LEAD_INDEX = {name: i for i, name in enumerate(
    ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
)}


def _gold_f1(rec: dict[str, Any]) -> tuple[str, tuple[float, float]] | None:
    """F1 — ¿QRS en V1 > 120 ms? Evidencia = complejo QRS de un latido en V1."""
    qrs_ms = rec.get("qrs_duration_v1_ms")
    if qrs_ms is None or not np.isfinite(qrs_ms):
        return None
    onset  = rec.get("qrs_onset_v1_s")
    offset = rec.get("qrs_offset_v1_s")
    if onset is None or offset is None:
        return None
    answer = "sí" if float(qrs_ms) > 120.0 else "no"
    return answer, (float(onset), float(offset))


def _gold_f2(rec: dict[str, Any]) -> tuple[str, tuple[float, float]] | None:
    """F2 — ¿Intervalo PR prolongado (>200 ms)? Evidencia = inicio P → inicio QRS."""
    pr_ms = rec.get("pr_interval_ms")
    p_onset = rec.get("p_onset_ii_s")
    qrs_onset = rec.get("qrs_onset_ii_s")
    if pr_ms is None or p_onset is None or qrs_onset is None:
        return None
    if not np.isfinite(pr_ms):
        return None
    answer = "sí" if float(pr_ms) > 200.0 else "no"
    return answer, (float(p_onset), float(qrs_onset))


def _gold_f3(rec: dict[str, Any]) -> tuple[str, tuple[float, float]] | None:
    """F3 — ¿Onda T en V5 negativa? Evidencia = ventana T en V5."""
    t_amp_v5 = rec.get("t_amplitude_v5_mv")
    t_onset  = rec.get("t_onset_v5_s")
    t_offset = rec.get("t_offset_v5_s")
    if t_amp_v5 is None or t_onset is None or t_offset is None:
        return None
    answer = "sí" if float(t_amp_v5) < 0 else "no"
    return answer, (float(t_onset), float(t_offset))


def _gold_f4(rec: dict[str, Any]) -> tuple[str, tuple[float, float]] | None:
    """F4 — ¿Frecuencia cardiaca > 100 lpm? Evidencia = los 10 s completos."""
    hr = rec.get("heart_rate_bpm")
    if hr is None or not np.isfinite(hr):
        return None
    answer = "sí" if float(hr) > 100.0 else "no"
    return answer, (0.0, 10.0)


def _gold_f5(rec: dict[str, Any]) -> tuple[str, tuple[float, float]] | None:
    """F5 — ¿Amplitud R en V5 > R en V1? Evidencia = ambos complejos QRS."""
    r_v1 = rec.get("r_amplitude_v1_mv")
    r_v5 = rec.get("r_amplitude_v5_mv")
    qrs_onset_v1  = rec.get("qrs_onset_v1_s")
    qrs_offset_v5 = rec.get("qrs_offset_v5_s")
    if r_v1 is None or r_v5 is None or qrs_onset_v1 is None or qrs_offset_v5 is None:
        return None
    answer = "sí" if float(r_v5) > float(r_v1) else "no"
    # Span cubre desde el inicio del QRS en V1 hasta el final del QRS en V5
    # (mismo latido, por eso el rango entero es informativo).
    return answer, (float(qrs_onset_v1), float(qrs_offset_v5))


FAMILIES: tuple[Family, ...] = (
    Family("F1", "duración QRS en V1",
           "¿La duración del QRS en V1 supera los 120 ms?",
           lead_gold=(LEAD_INDEX["V1"],), locality="puntual",
           answer_type="sí/no", compute_gold=_gold_f1),
    Family("F2", "intervalo PR",
           "¿El intervalo PR está prolongado (mayor a 200 ms)?",
           lead_gold=(LEAD_INDEX["II"],), locality="corta",
           answer_type="sí/no", compute_gold=_gold_f2),
    Family("F3", "polaridad de la onda T en V5",
           "¿La onda T en V5 es negativa?",
           lead_gold=(LEAD_INDEX["V5"],), locality="puntual",
           answer_type="sí/no", compute_gold=_gold_f3),
    Family("F4", "frecuencia cardiaca",
           "¿La frecuencia cardiaca supera los 100 latidos por minuto?",
           lead_gold=(LEAD_INDEX["II"], LEAD_INDEX["V5"]), locality="global",
           answer_type="sí/no", compute_gold=_gold_f4),
    Family("F5", "amplitud R en V5 vs V1",
           "¿La amplitud R es mayor en V5 que en V1?",
           lead_gold=(LEAD_INDEX["V1"], LEAD_INDEX["V5"]), locality="doble",
           answer_type="sí/no", compute_gold=_gold_f5),
)


# --------------------------------------------------------------------------- #
# 2. Lectura de PTB-XL + PTB-XL+ con filtro de concordancia                    #
# --------------------------------------------------------------------------- #
def load_ptbxl_index(ptbxl_root: Path) -> "pd.DataFrame":
    """Lee ``ptbxl_database.csv`` y devuelve un DataFrame indexado por ``ecg_id``."""
    import pandas as pd
    csv = ptbxl_root / "ptbxl_database.csv"
    df = pd.read_csv(csv, index_col="ecg_id")
    # Los folds oficiales son 1..10; 9 = valid, 10 = test, 1..8 = train.
    df["split"] = df["strat_fold"].apply(
        lambda k: "train" if k <= 8 else ("valid" if k == 9 else "test")
    )
    return df


def load_features(ptbxlplus_root: Path) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Carga las tablas de features 12SL y ECGDeli de PTB-XL+.

    Los nombres de columna dependen del release; los reintitulamos a un
    esquema común con `_normalize_features`.
    """
    import pandas as pd
    labels = ptbxlplus_root / "labels"
    df_12sl    = pd.read_csv(labels / "12sl_features.csv",    index_col="ecg_id")
    df_ecgdeli = pd.read_csv(labels / "ecgdeli_features.csv", index_col="ecg_id")
    return _normalize_features(df_12sl, source="12sl"), _normalize_features(df_ecgdeli, source="ecgdeli")


def _normalize_features(df: "pd.DataFrame", source: str) -> "pd.DataFrame":
    """Renombra columnas al esquema común que consumen las funciones ``_gold_f*``.

    Este mapa es el punto único de mantenimiento: si el release de PTB-XL+ cambia
    los nombres de columna, sólo hay que ajustar este dict.
    """
    # NOTE: los nombres de columna se documentan en el release notes de PTB-XL+.
    # Aquí listamos los alias más comunes; si el CSV real usa otros nombres, se
    # extiende esta tabla.
    rename_maps = {
        "12sl": {
            "QRS_dur_V1":   "qrs_duration_v1_ms",
            "QRSon_V1":     "qrs_onset_v1_s",
            "QRSoff_V1":    "qrs_offset_v1_s",
            "QRSon_V5":     "qrs_onset_v5_s",
            "QRSoff_V5":    "qrs_offset_v5_s",
            "Pon_II":       "p_onset_ii_s",
            "QRSon_II":     "qrs_onset_ii_s",
            "PR_interval":  "pr_interval_ms",
            "Ton_V5":       "t_onset_v5_s",
            "Toff_V5":      "t_offset_v5_s",
            "T_amp_V5":     "t_amplitude_v5_mv",
            "R_amp_V1":     "r_amplitude_v1_mv",
            "R_amp_V5":     "r_amplitude_v5_mv",
            "HR":           "heart_rate_bpm",
        },
        "ecgdeli": {
            # ECGDeli usa nombres ligeramente distintos.
            "qrs_dur_V1":   "qrs_duration_v1_ms",
            "qrs_on_V1":    "qrs_onset_v1_s",
            "qrs_off_V1":   "qrs_offset_v1_s",
            "qrs_on_V5":    "qrs_onset_v5_s",
            "qrs_off_V5":   "qrs_offset_v5_s",
            "p_on_II":      "p_onset_ii_s",
            "qrs_on_II":    "qrs_onset_ii_s",
            "pr_int":       "pr_interval_ms",
            "t_on_V5":      "t_onset_v5_s",
            "t_off_V5":     "t_offset_v5_s",
            "t_amp_V5":     "t_amplitude_v5_mv",
            "r_amp_V1":     "r_amplitude_v1_mv",
            "r_amp_V5":     "r_amplitude_v5_mv",
            "hr":           "heart_rate_bpm",
        },
    }
    mapping = {src: dst for src, dst in rename_maps[source].items() if src in df.columns}
    return df.rename(columns=mapping)


def concordance_filter(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    tol_ms: float = 10.0,
    tol_mv: float = 0.05,
) -> bool:
    """Devuelve True si las mediciones de dos proveedores coinciden dentro de tolerancia.

    Aplica sobre las columnas que consumen las ``_gold_f*``:
    - Duraciones e intervalos (en ms): |a − b| ≤ ``tol_ms``.
    - Amplitudes (en mV): |a − b| ≤ ``tol_mv``.
    - On/offsets (en s): |a − b| ≤ ``tol_ms``/1000.
    """
    def _match(key: str, is_ms: bool = False, is_mv: bool = False, is_s: bool = False) -> bool:
        av, bv = row_a.get(key), row_b.get(key)
        if av is None or bv is None or not np.isfinite(av) or not np.isfinite(bv):
            return False
        tol = tol_ms if is_ms else (tol_mv if is_mv else (tol_ms / 1000.0 if is_s else tol_ms))
        return abs(float(av) - float(bv)) <= tol

    checks = [
        _match("qrs_duration_v1_ms", is_ms=True),
        _match("pr_interval_ms",     is_ms=True),
        _match("heart_rate_bpm",     is_ms=True),  # bpm, tolerancia igual a ms (10 unidades)
        _match("t_amplitude_v5_mv",  is_mv=True),
        _match("r_amplitude_v1_mv",  is_mv=True),
        _match("r_amplitude_v5_mv",  is_mv=True),
        _match("qrs_onset_v1_s",     is_s=True),
        _match("qrs_offset_v1_s",    is_s=True),
        _match("t_onset_v5_s",       is_s=True),
        _match("t_offset_v5_s",      is_s=True),
    ]
    # No exigimos que todas las columnas existan (una familia puede necesitar
    # sólo un subconjunto): un ítem pasa el filtro si al menos las columnas
    # que efectivamente usa concuerdan.
    return all(checks)


# --------------------------------------------------------------------------- #
# 3. Instanciación de las cinco familias                                       #
# --------------------------------------------------------------------------- #
@dataclass
class QAItem:
    ecg_id: int
    family: str
    question: str
    answer: str
    span: tuple[float, float]
    leads_gold: list[int]
    condition: str = "natural"
    split: str = "train"
    signal_path: str = ""
    swap_partner: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "ecg_id":          self.ecg_id,
            "family":          self.family,
            "question":        self.question,
            "answer":          self.answer,
            "span":            list(self.span),
            "leads_gold":      list(self.leads_gold),
            "condition":       self.condition,
            "split":           self.split,
            "ecg_signal_path": self.signal_path,
            "swap_partner":    self.swap_partner,
            "metadata":        self.metadata,
        }


def instantiate_families(
    features: "pd.DataFrame",
    ptbxl_index: "pd.DataFrame",
    signal_root: Path,
) -> list[QAItem]:
    """Recorre las features concordadas y produce ítems (uno por familia por
    registro, si aplica)."""
    items: list[QAItem] = []
    for ecg_id, row in features.iterrows():
        rec = row.to_dict()
        if ecg_id not in ptbxl_index.index:
            continue
        split = str(ptbxl_index.at[ecg_id, "split"])
        signal_path = _signal_path_for(ecg_id, signal_root)
        for fam in FAMILIES:
            gold = fam.compute_gold(rec)
            if gold is None:
                continue
            answer, span = gold
            items.append(QAItem(
                ecg_id=int(ecg_id),
                family=fam.fid,
                question=fam.question_template,
                answer=answer,
                span=(round(span[0], 4), round(span[1], 4)),
                leads_gold=list(fam.lead_gold),
                condition="natural",
                split=split,
                signal_path=str(signal_path),
                metadata={"locality": fam.locality, "answer_type": fam.answer_type},
            ))
    return items


def _signal_path_for(ecg_id: int, signal_root: Path) -> Path:
    """Ruta al .npy correspondiente al ecg_id. Convención PTB-XL:
    ``records500/{ecg_id//1000 * 1000:05d}/{ecg_id:05d}_hr``."""
    folder = f"{(ecg_id // 1000) * 1000:05d}"
    return signal_root / "records500" / folder / f"{ecg_id:05d}_hr.npy"


# --------------------------------------------------------------------------- #
# 4. Balanceo y emparejamiento contrafactual                                   #
# --------------------------------------------------------------------------- #
def balance_by_answer(items: list[QAItem], seed: int) -> list[QAItem]:
    """Rebalancea cada (familia, split) para que P(answer | plantilla) ≈ uniforme.
    Estrategia: undersampling de la clase mayoritaria."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[QAItem]] = defaultdict(list)
    for it in items:
        buckets[(it.family, it.split)].append(it)

    balanced: list[QAItem] = []
    for (fam, split), bucket in buckets.items():
        by_answer: dict[str, list[QAItem]] = defaultdict(list)
        for it in bucket:
            by_answer[it.answer].append(it)
        if len(by_answer) < 2:
            # una sola clase — no rebalanceable
            balanced.extend(bucket)
            continue
        min_size = min(len(v) for v in by_answer.values())
        for cls_items in by_answer.values():
            rng.shuffle(cls_items)
            for it in cls_items[:min_size]:
                it.condition = "balanced"
                balanced.append(it)
    return balanced


def pair_swaps(
    items: list[QAItem],
    metadata: "pd.DataFrame",
    seed: int,
    tolerate_age: int = 5,
) -> list[QAItem]:
    """Empareja cada ítem con un compañero de la misma familia, mismo split,
    respuesta gold DISTINTA, apareado en edad (±tolerate_age), sexo y hr
    (±10 bpm). El swap alimenta la intervención ``swap``/``swap_null`` de F03.
    """
    rng = random.Random(seed + 1)
    by_key: dict[tuple[str, str, str], list[QAItem]] = defaultdict(list)
    for it in items:
        by_key[(it.family, it.split, it.answer)].append(it)

    def _cand_ok(a: QAItem, b: QAItem) -> bool:
        if a.ecg_id == b.ecg_id:
            return False
        try:
            age_a = float(metadata.at[a.ecg_id, "age"])
            age_b = float(metadata.at[b.ecg_id, "age"])
            sex_a = metadata.at[a.ecg_id, "sex"]
            sex_b = metadata.at[b.ecg_id, "sex"]
        except KeyError:
            return True   # sin metadatos → aceptamos el emparejamiento
        return sex_a == sex_b and abs(age_a - age_b) <= tolerate_age

    for it in items:
        opposite = [a for a in ("sí", "no") if a != it.answer]
        cands: list[QAItem] = []
        for ans in opposite:
            cands.extend(by_key[(it.family, it.split, ans)])
        rng.shuffle(cands)
        for cand in cands:
            if _cand_ok(it, cand):
                it.swap_partner = cand.ecg_id
                break
    return items


# --------------------------------------------------------------------------- #
# 5. Conversión de señales .dat a .npy (una única vez)                         #
# --------------------------------------------------------------------------- #
def ensure_npy_cache(ecg_ids: Iterable[int], ptbxl_root: Path, force: bool = False) -> int:
    """Convierte los registros wfdb (.hea/.dat) a .npy 12×5000 en la carpeta
    ``records500/`` del propio PTB-XL. Idempotente: salta los ya convertidos.

    Requiere ``wfdb`` instalado. Devuelve el número de nuevos .npy escritos.
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "F04.1 requiere el paquete `wfdb` para leer los registros de PTB-XL. "
            "Instálalo con `pip install wfdb`."
        ) from exc

    written = 0
    for ecg_id in ecg_ids:
        npy_path = _signal_path_for(ecg_id, ptbxl_root)
        if npy_path.exists() and not force:
            continue
        folder = f"{(ecg_id // 1000) * 1000:05d}"
        hea_stem = ptbxl_root / "records500" / folder / f"{ecg_id:05d}_hr"
        if not (hea_stem.with_suffix(".hea")).exists():
            continue
        signal, _fields = wfdb.rdsamp(str(hea_stem))
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, signal.astype(np.float32))
        written += 1
    return written


# --------------------------------------------------------------------------- #
# 6. Escritura del JSONL por split                                             #
# --------------------------------------------------------------------------- #
def write_split(items: list[QAItem], out_dir: Path) -> dict[str, int]:
    """Escribe un JSONL por split. Devuelve conteos por split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        "train": (out_dir / "processed_train.jsonl").open("w", encoding="utf-8"),
        "valid": (out_dir / "processed_valid.jsonl").open("w", encoding="utf-8"),
        "test":  (out_dir / "processed_test.jsonl").open("w", encoding="utf-8"),
    }
    counts: dict[str, int] = defaultdict(int)
    try:
        for it in items:
            handles[it.split].write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")
            counts[it.split] += 1
    finally:
        for h in handles.values():
            h.close()
    return dict(counts)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F04.1 · Construcción del conjunto controlado F1-F5 sobre PTB-XL/PTB-XL+."
    )
    parser.add_argument("--ptbxl_root", type=Path, required=True,
                        help="Raíz de PTB-XL (contiene ptbxl_database.csv y records500/).")
    parser.add_argument("--ptbxlplus_root", type=Path, required=True,
                        help="Raíz de PTB-XL+ (contiene labels/12sl_features.csv y labels/ecgdeli_features.csv).")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("data/qa_controlled"))
    parser.add_argument("--tol_ms", type=float, default=10.0)
    parser.add_argument("--tol_mv", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_npy_cache", action="store_true",
                        help="No convertir wfdb→.npy (útil si ya está cacheado).")
    return parser.parse_args()


def main() -> None:
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "F04.1 requiere pandas. Instálalo con `pip install pandas`."
        ) from exc

    args = parse_args()

    print(f"[qa] cargando PTB-XL desde {args.ptbxl_root}")
    ptbxl_index = load_ptbxl_index(args.ptbxl_root)
    print(f"[qa]   registros: {len(ptbxl_index)}")

    print(f"[qa] cargando PTB-XL+ features desde {args.ptbxlplus_root}")
    df_12sl, df_ecgdeli = load_features(args.ptbxlplus_root)
    print(f"[qa]   12SL rows:    {len(df_12sl)}")
    print(f"[qa]   ECGDeli rows: {len(df_ecgdeli)}")

    # Merge por ecg_id, filtro de concordancia por fila.
    common = df_12sl.index.intersection(df_ecgdeli.index)
    kept_rows: dict[int, dict[str, Any]] = {}
    discarded = 0
    for ecg_id in common:
        row_a = df_12sl.loc[ecg_id].to_dict()
        row_b = df_ecgdeli.loc[ecg_id].to_dict()
        if concordance_filter(row_a, row_b, tol_ms=args.tol_ms, tol_mv=args.tol_mv):
            # Media de ambos proveedores como valor canónico.
            merged = {k: (row_a.get(k) + row_b.get(k)) / 2
                      if isinstance(row_a.get(k), (int, float)) and isinstance(row_b.get(k), (int, float))
                      else row_a.get(k)
                      for k in row_a.keys() | row_b.keys()}
            kept_rows[int(ecg_id)] = merged
        else:
            discarded += 1
    print(f"[qa] concordancia 12SL↔ECGDeli: kept={len(kept_rows)} discarded={discarded}")

    features = _dict_to_df(kept_rows)

    print("[qa] instanciando familias F1-F5")
    items = instantiate_families(features, ptbxl_index, args.ptbxl_root)
    per_fam = defaultdict(int)
    for it in items:
        per_fam[it.family] += 1
    for fid, n in sorted(per_fam.items()):
        print(f"[qa]   {fid}: {n}")

    if not args.skip_npy_cache:
        print("[qa] convirtiendo señales wfdb → .npy (idempotente)")
        needed_ids = sorted({it.ecg_id for it in items})
        written = ensure_npy_cache(needed_ids, args.ptbxl_root)
        print(f"[qa]   .npy escritos: {written}")

    print(f"[qa] balanceando por respuesta (seed={args.seed})")
    balanced = balance_by_answer(items, seed=args.seed)
    per_fam_b = defaultdict(int)
    for it in balanced:
        per_fam_b[it.family] += 1
    print("[qa]   balanceados por familia:", dict(per_fam_b))

    print("[qa] emparejando swaps contrafactuales")
    balanced = pair_swaps(balanced, ptbxl_index, seed=args.seed)
    paired = sum(1 for it in balanced if it.swap_partner is not None)
    print(f"[qa]   ítems con swap_partner: {paired}/{len(balanced)}")

    counts = write_split(balanced, args.output_dir)
    print(f"[qa] escrito en {args.output_dir}")
    print("[qa]   splits:", counts)


def _dict_to_df(rows: dict[int, dict[str, Any]]) -> "pd.DataFrame":
    import pandas as pd
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ecg_id"
    return df


if __name__ == "__main__":
    main()
