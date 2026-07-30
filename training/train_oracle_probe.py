"""F04.2 · Sonda oracle: CNN 1D por familia sobre la señal cruda de ECG.

Papel en el marco: proveer el techo ``A_oracle`` de decidibilidad que hace
falta para calcular ``TDI = 1 − ΔS / (A_oracle − A_blind)`` (Sección III-C del
paper). Sin este techo, sólo se puede reportar ``D_cf = QCFR − ECFR`` (índice
contrafactual) y no queda claro si un ``ΔS`` bajo se debe a que la señal es
irrelevante para la tarea o a que el modelo no la usa.

Diseño de la sonda:

- **Un CNN 1D por familia** (F1..F5), no un modelo compartido. Cada familia
  tiene distintos ejes de evidencia (localidad temporal, especificidad de
  derivación), y un solo modelo confundiría los patrones.
- **Pequeña**: ~50 k parámetros. La sonda debe medir *decidibilidad* de la
  tarea, no memorizar el conjunto de entrenamiento.
- **Entrenada sobre el mismo split que E1**: usa el JSONL producido por
  ``build_qa.py`` con condición ``balanced`` para que ``A_blind ≈ azar`` por
  construcción y ``A_oracle − A_blind`` sea informativo.

Uso típico (Ubuntu, GPU CUDA)::

    python training/train_oracle_probe.py \\
      --qa_train data/qa_controlled/processed_train.jsonl \\
      --qa_valid data/qa_controlled/processed_valid.jsonl \\
      --output_dir checkpoints/oracle \\
      --epochs 30 \\
      --batch_size 32

Salidas:

- ``checkpoints/oracle/oracle_F1.pt`` .. ``oracle_F5.pt`` — pesos por familia.
- ``outputs/oracle/metrics_train.json`` — loss y exactitud por época y familia.

Coste estimado en RTX 4050: 5-15 min por familia (5000-8000 ejemplos, 30 épocas
con early stopping). Total: < 1 h.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Arquitectura: pequeña 1D CNN con GAP                                          #
# --------------------------------------------------------------------------- #
class OracleCNN(nn.Module):
    """CNN 1D pequeña (~50k params). Entrada [B, 12, T] → probabilidad binaria.

    Estructura:
      - Conv1d(12, 32, k=15, s=2) + BN + ReLU + MaxPool(2)   # T=5000 → ~625
      - Conv1d(32, 64, k=11, s=2) + BN + ReLU + MaxPool(2)   # → ~78
      - Conv1d(64, 96, k=7, s=1)  + BN + ReLU                 # → 72
      - AdaptiveAvgPool1d(1)                                  # → [B, 96, 1]
      - Linear(96 → 2)
    """

    def __init__(self, in_leads: int = 12, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_leads, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] o [B, C, T]. Aceptamos ambos.
        if x.dim() == 3 and x.shape[1] != 12 and x.shape[2] == 12:
            x = x.transpose(1, 2)  # → [B, 12, T]
        h = self.features(x).squeeze(-1)  # [B, 96]
        return self.classifier(h)         # [B, 2]


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
class QAFamilyDataset(Dataset):
    """Dataset por familia: carga ECG + etiqueta binaria (sí/no)."""

    def __init__(self, rows: list[dict[str, Any]], family: str) -> None:
        self.rows = [r for r in rows if r.get("family") == family]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        signal = np.load(row["ecg_signal_path"]).astype(np.float32)  # [T, 12]
        # z-score por derivación para estabilidad numérica.
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
        std = np.where(std > 1e-6, std, 1.0)
        signal = (signal - mean) / std
        label = 1 if str(row["answer"]).strip().lower() in ("sí", "si", "yes", "y") else 0
        return torch.from_numpy(signal), label


def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    signals, labels = zip(*batch)
    # Padding (raro en PTB-XL: todos son 5000, pero robustos).
    max_t = max(s.shape[0] for s in signals)
    C = signals[0].shape[1]
    out = torch.zeros(len(signals), max_t, C, dtype=torch.float32)
    for i, s in enumerate(signals):
        out[i, : s.shape[0]] = s
    return out, torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Loop de entrenamiento                                                        #
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_one_family(
    family: str,
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    device: torch.device,
    out_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train_ds = QAFamilyDataset(train_rows, family)
    valid_ds = QAFamilyDataset(valid_rows, family)
    if len(train_ds) < 32:
        return {"family": family, "skipped": True, "reason": f"pocos ejemplos train ({len(train_ds)})"}

    print(f"[oracle {family}] train={len(train_ds)}  valid={len(valid_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = OracleCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_valid_acc = -1.0
    best_epoch = -1
    epochs_no_imp = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for signals, labels in train_loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * signals.size(0)
            train_correct += int((logits.argmax(dim=-1) == labels).sum())
            train_total += signals.size(0)
        scheduler.step()

        train_acc = train_correct / max(train_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)
        valid_acc = _evaluate(model, valid_loader, device) if len(valid_ds) else None
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "valid_acc": valid_acc})
        print(f"[oracle {family}] epoch={epoch} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} valid_acc={valid_acc}")

        if valid_acc is not None and valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            epochs_no_imp = 0
            torch.save({"state_dict": model.state_dict(), "family": family,
                        "valid_acc": valid_acc, "epoch": epoch}, out_path)
        else:
            epochs_no_imp += 1
        if patience and epochs_no_imp >= patience:
            print(f"[oracle {family}] early stop: mejor época {best_epoch} valid_acc={best_valid_acc:.4f}")
            break

    return {"family": family, "best_epoch": best_epoch,
            "best_valid_acc": best_valid_acc, "history": history,
            "checkpoint": str(out_path), "train_examples": len(train_ds),
            "valid_examples": len(valid_ds)}


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)
        logits = model(signals)
        correct += int((logits.argmax(dim=-1) == labels).sum())
        total += signals.size(0)
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F04.2 · Entrena oracle CNN 1D por familia.")
    parser.add_argument("--qa_train", type=Path,
                        default=PROJECT_ROOT / "data/qa_controlled/processed_train.jsonl")
    parser.add_argument("--qa_valid", type=Path,
                        default=PROJECT_ROOT / "data/qa_controlled/processed_valid.jsonl")
    parser.add_argument("--output_dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints/oracle")
    parser.add_argument("--log_dir", type=Path,
                        default=PROJECT_ROOT / "outputs/oracle")
    parser.add_argument("--families", nargs="+", default=["F1", "F2", "F3", "F4", "F5"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[oracle] device={device}")

    train_rows = read_jsonl(args.qa_train)
    valid_rows = read_jsonl(args.qa_valid) if args.qa_valid.exists() else []
    print(f"[oracle] cargados: train={len(train_rows)} valid={len(valid_rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for family in args.families:
        out_path = args.output_dir / f"oracle_{family}.pt"
        result = train_one_family(
            family, train_rows, valid_rows, device, out_path,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, patience=args.patience,
            seed=args.seed,
        )
        all_metrics.append(result)

    summary = {
        "device": str(device),
        "families": all_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.log_dir / "metrics_train.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
