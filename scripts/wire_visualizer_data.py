"""F06 · Compone un artefacto ``audit_data.js`` que el ``auditor_ecg.html``
consume, uniendo las salidas reales de las fases F02, F03 y F04.

Motivación: el visualizador actual (``auditor_ecg.html``) trae los datos
hardcodeados como ``const ACC = {...}, LEDGER = {...}, DELTA = {...}``. Este
script genera un archivo ``audit_data.js`` con la misma forma pero rellenado
desde los JSON reales del pipeline, para que el HTML pueda consumirlos con un
único ``<script src="audit_data.js"></script>`` sin editar el HTML a mano.

Entradas:

- ``outputs/e0_text_only/a_blind_by_family.json``  (F02.2)
- ``outputs/oracle/a_oracle_by_family.json``       (F04.3)
- ``outputs/audit/cfr_dose_response.json``         (F03.2)
- ``outputs/audit/delta_calibration.json``         (F03.1)
- ``outputs/audit/ledger_<cfg>.json``              (F1 cascada de sondas)

Salida:

- ``Mini-STReasoner/visualizer/audit_data.js`` con un objeto ``AUDIT`` que
  reemplaza los ``const ACC/LEDGER/DELTA/GEO/PEAK`` del ``auditor_ecg.html``.

Diseño: si un archivo de entrada falta, el campo correspondiente en ``AUDIT``
queda como ``null`` en lugar de fallar. Así el visualizador puede degradar
graciosamente y mostrar el estado real de la corrida (algunas fases medidas,
otras pendientes).

Uso típico::

    python scripts/wire_visualizer_data.py \\
      --a_blind      outputs/e0_text_only/a_blind_by_family.json \\
      --a_oracle     outputs/oracle/a_oracle_by_family.json \\
      --dose         outputs/audit/cfr_dose_response.json \\
      --delta_calib  outputs/audit/delta_calibration.json \\
      --ledger_e1    outputs/audit/ledger_e1.json \\
      --output       visualizer/audit_data.js
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[wire] ADVERTENCIA: no pude leer {path}: {exc}", file=sys.stderr)
        return None


def compose_audit(
    a_blind: Any,
    a_oracle: Any,
    dose: Any,
    delta_calib: Any,
    ledger_e1: Any,
    ledger_e2: Any,
    ledger_e3: Any,
) -> dict[str, Any]:
    """Formato consumible por ``auditor_ecg.html``.

    El HTML actual espera cuatro objetos globales: ``ACC``, ``LEDGER``,
    ``DELTA``, ``GEO``, ``PEAK``. Los envolvemos bajo ``AUDIT`` y añadimos
    metadatos para que quede claro qué es medición y qué es proyección.
    """
    families = _extract_families(a_blind, a_oracle)
    acc = _compose_acc(a_blind, a_oracle, dose, families)
    ledger = _compose_ledger(ledger_e1, ledger_e2, ledger_e3)
    delta = _compose_delta(dose, delta_calib)
    return {
        "meta": {
            "source":       "wire_visualizer_data.py",
            "has_a_blind":  a_blind is not None,
            "has_a_oracle": a_oracle is not None,
            "has_dose":     dose is not None,
            "has_ledger_e1": ledger_e1 is not None,
            "has_ledger_e2": ledger_e2 is not None,
            "has_ledger_e3": ledger_e3 is not None,
        },
        "families":  families,
        "ACC":       acc,
        "LEDGER":    ledger,
        "DELTA":     delta,
        # PEAK y GEO se dejan para completar cuando existan los agregados
        # respectivos (test de oclusión, geometría por etapa).
        "PEAK": None,
        "GEO":  None,
    }


def _extract_families(a_blind: Any, a_oracle: Any) -> list[str]:
    """Devuelve la unión ordenada de familias entre ambas fuentes."""
    fam: set[str] = set()
    for src in (a_blind, a_oracle):
        if isinstance(src, dict):
            fam.update(src.keys())
    return sorted(fam)


def _compose_acc(
    a_blind: Any, a_oracle: Any, dose: Any, families: list[str],
) -> dict[str, dict[str, list[float | None]]]:
    """Reproduce el ``ACC[cfg][fam] = [blind, full, oracle, cfr, cfr0]`` del HTML.

    - ``blind``  proviene de F02.2 (a_blind_by_family.json).
    - ``oracle`` proviene de F04.3 (a_oracle_by_family.json).
    - ``full``, ``cfr`` y ``cfr0``: pendientes hasta tener evaluaciones por
      familia con el modelo E1 real; se dejan como ``None`` para que el
      visualizador los muestre como "pendiente".
    """
    out: dict[str, dict[str, list[float | None]]] = {"E1": {}, "E2": {}, "E3": {}}
    for fam in families:
        blind_val  = None
        oracle_val = None
        if isinstance(a_blind, dict) and fam in a_blind:
            blind_val = a_blind[fam].get("blind")
        if isinstance(a_oracle, dict) and fam in a_oracle:
            oracle_val = a_oracle[fam].get("oracle")
        # Por ahora sólo E1 tiene medición real. Para E2/E3, mantener los mismos
        # blind/oracle (son propiedades del dataset, no de la config del modelo).
        for cfg in ("E1", "E2", "E3"):
            out[cfg][fam] = [blind_val, None, oracle_val, None, None]
    return out


def _compose_ledger(*ledgers: Any) -> dict[str, dict[str, Any]]:
    """Ledger por configuración: si existe la salida de la cascada de sondas
    (F1 del plan, aún pendiente de instanciar sobre corridas reales) se pone
    tal cual; si no, ``None``."""
    out: dict[str, Any] = {}
    for cfg, payload in zip(("E1", "E2", "E3"), ledgers):
        if payload is None:
            out[cfg] = None
            continue
        out[cfg] = {
            "stages":        payload.get("stages")        or payload.get("ledgerStages"),
            "ledger":        payload.get("ledger"),
            "ledgerControl": payload.get("ledgerControl") or payload.get("control"),
            "selectivity":   payload.get("selectivity"),
        }
    return out


def _compose_delta(dose: Any, delta_calib: Any) -> dict[str, Any]:
    """Combina curva CFR(δ) (F03.2) con la tabla de calibración (F03.1)."""
    return {
        "cfr_by_delta": (dose or {}).get("cfr_by_delta") if isinstance(dose, dict) else None,
        "highlights":   (dose or {}).get("highlights_text_vs_signal_at_matched_delta") if isinstance(dose, dict) else None,
        "calibration":  (delta_calib or {}).get("summary") if isinstance(delta_calib, dict) else None,
        "pairs":        (delta_calib or {}).get("pairs_by_delta") if isinstance(delta_calib, dict) else None,
    }


def write_js(audit: dict[str, Any], out_path: Path) -> None:
    """Escribe ``audit_data.js`` con la asignación global."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "/* Autogenerado por scripts/wire_visualizer_data.py — NO editar a mano.\n"
        " * Los campos null representan mediciones que aún no se han producido\n"
        " * en la corrida real. El auditor_ecg.html degrada graciosamente:\n"
        " * muestra 'pendiente' en lugar de sintéticos cuando encuentra null. */\n"
    )
    body = "window.AUDIT = " + json.dumps(audit, indent=2, ensure_ascii=False) + ";\n"
    out_path.write_text(header + body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F06 · Compone audit_data.js desde las salidas reales del pipeline."
    )
    parser.add_argument("--a_blind",     type=Path, default=PROJECT_ROOT / "outputs/e0_text_only/a_blind_by_family.json")
    parser.add_argument("--a_oracle",    type=Path, default=PROJECT_ROOT / "outputs/oracle/a_oracle_by_family.json")
    parser.add_argument("--dose",        type=Path, default=PROJECT_ROOT / "outputs/audit/cfr_dose_response.json")
    parser.add_argument("--delta_calib", type=Path, default=PROJECT_ROOT / "outputs/audit/delta_calibration.json")
    parser.add_argument("--ledger_e1",   type=Path, default=PROJECT_ROOT / "outputs/audit/ledger_e1.json")
    parser.add_argument("--ledger_e2",   type=Path, default=PROJECT_ROOT / "outputs/audit/ledger_e2.json")
    parser.add_argument("--ledger_e3",   type=Path, default=PROJECT_ROOT / "outputs/audit/ledger_e3.json")
    parser.add_argument("--output",      type=Path, default=PROJECT_ROOT / "visualizer/audit_data.js")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = compose_audit(
        a_blind      = _read_json(args.a_blind),
        a_oracle     = _read_json(args.a_oracle),
        dose         = _read_json(args.dose),
        delta_calib  = _read_json(args.delta_calib),
        ledger_e1    = _read_json(args.ledger_e1),
        ledger_e2    = _read_json(args.ledger_e2),
        ledger_e3    = _read_json(args.ledger_e3),
    )
    write_js(audit, args.output)
    print(f"[wire] escrito {args.output}")
    print(f"[wire] meta: {json.dumps(audit['meta'], indent=2)}")
    print("[wire] Para conectar el visualizador, añade en <head> de auditor_ecg.html:")
    print('       <script src="audit_data.js"></script>')
    print("       y reemplaza el bloque de constantes por lecturas de window.AUDIT.")


if __name__ == "__main__":
    main()
