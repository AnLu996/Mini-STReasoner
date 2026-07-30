"""F04.3 · Evaluación de las sondas oracle y export para el visualizador.

Carga los checkpoints producidos por ``training/train_oracle_probe.py`` y
reporta ``A_oracle`` por familia sobre el conjunto de test. Combinado con
``A_blind`` (de ``scripts/evaluate_text_only_baseline.py``), permite calcular
el índice normalizado ``TDI = 1 − ΔS / (A_oracle − A_blind)`` que la Sección
III-C del paper define como "índice normalizado por decidibilidad".

Salida principal: ``a_oracle_by_family.json`` en el formato que consume el
visualizador (campo ``oracle`` de la ``Condition``).

Uso típico::

    python scripts/evaluate_oracle.py \\
      --qa_test data/qa_controlled/processed_test.jsonl \\
      --checkpoints_dir checkpoints/oracle \\
      --oracle_export outputs/oracle/a_oracle_by_family.json \\
      --a_blind outputs/e0_text_only/a_blind_by_family.json
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_oracle_probe import OracleCNN, QAFamilyDataset, collate  # noqa: E402


def load_oracle(path: Path, device: torch.device) -> OracleCNN:
    ckpt = torch.load(path, map_location=device)
    model = OracleCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_family(
    ckpt_path: Path,
    rows: list[dict[str, Any]],
    family: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if not ckpt_path.exists():
        return {"family": family, "oracle": None, "reason": "checkpoint no encontrado",
                "checkpoint": str(ckpt_path)}
    model = load_oracle(ckpt_path, device)
    ds = QAFamilyDataset(rows, family)
    if len(ds) == 0:
        return {"family": family, "oracle": None, "reason": "sin ejemplos test",
                "checkpoint": str(ckpt_path)}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    correct = total = 0
    yesno_correct = yesno_total = 0
    for signals, labels in loader:
        signals = signals.to(device)
        labels_dev = labels.to(device)
        logits = model(signals)
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels_dev).sum())
        total += signals.size(0)
        # Todas las familias son binarias por diseño; sí/no coincide con exactitud.
        yesno_correct += int((preds == labels_dev).sum())
        yesno_total += signals.size(0)
    return {
        "family": family,
        "oracle": correct / total,
        "yesno_accuracy": yesno_correct / yesno_total,
        "count": total,
        "checkpoint": str(ckpt_path),
    }


def build_export(
    per_family: list[dict[str, Any]],
    a_blind: dict[str, Any] | None,
) -> dict[str, Any]:
    """Formato de consumo del visualizador. Añade ``TDI`` cuando ``a_blind`` está
    disponible y el denominador supera el umbral 0.25 (Sección III-C)."""
    out: dict[str, Any] = {}
    for fam in per_family:
        fid = fam["family"]
        oracle = fam.get("oracle")
        entry: dict[str, Any] = {
            "oracle": None if oracle is None else round(oracle, 4),
            "count":  fam.get("count", 0),
        }
        if a_blind and fid in a_blind and oracle is not None:
            blind = a_blind[fid].get("blind")
            if blind is not None:
                denom = oracle - blind
                entry["blind"] = round(float(blind), 4)
                entry["denominator"] = round(float(denom), 4)
                if denom < 0.25:
                    entry["tdi"] = None
                    entry["tdi_note"] = "no interpretable (A_oracle − A_blind < 0.25)"
                else:
                    # TDI requiere ΔS = A_full − A_blind, que el visualizador ya
                    # computa a partir de sus propios campos. Aquí sólo dejamos
                    # las bases; el TDI final se compone en el visualizador.
                    entry["tdi"] = None
                    entry["tdi_note"] = "listo para composición en el visualizador"
        out[fid] = entry
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F04.3 · Evalúa oracle CNN por familia.")
    parser.add_argument("--qa_test", type=Path,
                        default=PROJECT_ROOT / "data/qa_controlled/processed_test.jsonl")
    parser.add_argument("--checkpoints_dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints/oracle")
    parser.add_argument("--families", nargs="+", default=["F1", "F2", "F3", "F4", "F5"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--summary", type=Path,
                        default=PROJECT_ROOT / "outputs/oracle/metrics_test.json")
    parser.add_argument("--oracle_export", type=Path,
                        default=PROJECT_ROOT / "outputs/oracle/a_oracle_by_family.json")
    parser.add_argument("--a_blind", type=Path,
                        default=PROJECT_ROOT / "outputs/e0_text_only/a_blind_by_family.json",
                        help="Ruta al export de F02.2; si existe se compone la base para TDI.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu": return torch.device("cpu")
    if requested == "cuda": return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rows = [json.loads(line) for line in args.qa_test.open(encoding="utf-8") if line.strip()]
    print(f"[oracle-eval] test rows: {len(rows)}")

    per_family: list[dict[str, Any]] = []
    for family in args.families:
        ckpt = args.checkpoints_dir / f"oracle_{family}.pt"
        result = evaluate_family(ckpt, rows, family, device, args.batch_size)
        per_family.append(result)
        oracle = result.get("oracle")
        print(f"[oracle-eval] {family}: A_oracle={oracle} (n={result.get('count', 0)})")

    a_blind = None
    if args.a_blind.exists():
        try:
            a_blind_raw = json.loads(args.a_blind.read_text(encoding="utf-8"))
            # F02.2 emitía claves como F_verify, F_query… Aquí las familias son F1..F5.
            # Aceptamos ambos esquemas; si no hay match exacto, a_blind queda vacío.
            a_blind = a_blind_raw if any(k in ("F1", "F2", "F3", "F4", "F5") for k in a_blind_raw) else None
        except Exception:  # noqa: BLE001
            a_blind = None

    export = build_export(per_family, a_blind)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({"per_family": per_family}, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    args.oracle_export.parent.mkdir(parents=True, exist_ok=True)
    args.oracle_export.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[oracle-eval] escrito {args.summary}")
    print(f"[oracle-eval] escrito {args.oracle_export} (formato visualizador)")


if __name__ == "__main__":
    main()
