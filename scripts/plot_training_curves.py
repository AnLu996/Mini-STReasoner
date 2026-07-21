"""Training curves (train + validation) from ``training_log.jsonl``.

The paper figure previously showed only the training loss, because that was the
only curve the pipeline produced. This script reads the log written by
``training/train_ecgqa_lora_small.py`` and plots both channels:

* per-step training loss, plus the per-epoch mean,
* per-epoch validation loss, exact match and token-F1,
* the selected (best) epoch, so the figure shows *which* checkpoint was kept.

The log is append-only and every record carries a ``run`` id. By default the
latest run is plotted; ``--run all`` overlays every run found, which is how the
control run and the improved-encoder run get compared on one figure.

Example::

    python scripts/plot_training_curves.py \\
      --log outputs/ecgqa_small/training_log.jsonl \\
      --output outputs/ecgqa_small/training_curves.png
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # sin display: el pipeline corre headless
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_log(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # linea truncada por una corrida interrumpida
    return records


def group_runs(records: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    """Split records by run id, preserving file order.

    Records written before runs were tagged have no ``run`` field; they are all
    attributed to a single ``legacy`` run so old logs still plot.
    """
    runs: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for rec in records:
        runs.setdefault(str(rec.get("run", "legacy")), []).append(rec)
    return runs


def split_records(run: list[dict[str, Any]]) -> tuple[list[dict], list[dict], dict]:
    """Return (step records, epoch records, run_start metadata)."""
    meta: dict[str, Any] = {}
    steps, epochs = [], []
    for rec in run:
        event = rec.get("event")
        if event == "run_start":
            meta = rec
        elif event:
            continue  # run_end u otros marcadores
        elif "step" in rec:
            steps.append(rec)
        elif "epoch" in rec:
            epochs.append(rec)
    return steps, epochs, meta


def epoch_to_step(steps: list[dict], epochs: list[dict], meta: dict) -> dict[int, float]:
    """Map each epoch to an x position on the step axis (its last step)."""
    spe = meta.get("steps_per_epoch")
    last: dict[int, float] = {}
    for rec in steps:
        last[rec["epoch"]] = max(last.get(rec["epoch"], 0), rec["step"])
    mapping: dict[int, float] = {}
    for rec in epochs:
        e = rec["epoch"]
        if e in last:
            mapping[e] = last[e]
        elif spe:
            mapping[e] = e * spe
        else:
            mapping[e] = e
    return mapping


def plot(runs: "OrderedDict[str, list[dict]]", output: Path, title: str) -> dict[str, Any]:
    fig, (ax_loss, ax_metric) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    palette = plt.get_cmap("tab10")
    summary: dict[str, Any] = {"runs": []}

    for i, (run_id, records) in enumerate(runs.items()):
        colour = palette(i % 10)
        steps, epochs, meta = split_records(records)
        if not steps and not epochs:
            continue
        tag = run_id if len(runs) > 1 else ""
        suffix = f" · {tag}" if tag else ""
        e2s = epoch_to_step(steps, epochs, meta)

        if steps:
            ax_loss.plot([r["step"] for r in steps], [r["train_loss"] for r in steps],
                         color=colour, alpha=0.28, linewidth=1,
                         label=f"train (por paso){suffix}")
        if epochs:
            xs = [e2s[r["epoch"]] for r in epochs]
            tr = [r.get("train_loss") for r in epochs]
            if any(v is not None for v in tr):
                ax_loss.plot(xs, tr, "o-", color=colour, linewidth=1.8,
                             label=f"train (media por época){suffix}")
            vl = [r.get("valid_loss") for r in epochs]
            if any(v is not None for v in vl):
                ax_loss.plot(xs, vl, "s--", color=colour, linewidth=1.8, markerfacecolor="white",
                             label=f"validación{suffix}")
            for key, marker, style in (("token_f1", "o", "-"), ("exact_match", "^", "--")):
                ys = [r.get(key) for r in epochs]
                if any(v is not None for v in ys):
                    ax_metric.plot(xs, ys, marker=marker, linestyle=style, color=colour,
                                   linewidth=1.6, label=f"{key}{suffix}")

            # Epoca seleccionada: la ultima marcada como mejora.
            best = [r for r in epochs if r.get("improved")]
            if best:
                bx = e2s[best[-1]["epoch"]]
                for ax in (ax_loss, ax_metric):
                    ax.axvline(bx, color=colour, linestyle=":", linewidth=1.4, alpha=0.8)
                # Anclada al eje (no a los datos) y abajo, para no chocar con la leyenda.
                ax_loss.annotate(f"checkpoint · época {best[-1]['epoch']}", xy=(bx, 0.02),
                                 xycoords=ax_loss.get_xaxis_transform(),
                                 xytext=(4, 0), textcoords="offset points",
                                 fontsize=8, color=colour, rotation=90, va="bottom")
            summary["runs"].append({
                "run": run_id,
                "epochs": len(epochs),
                "best_epoch": best[-1]["epoch"] if best else None,
                "final_valid_loss": epochs[-1].get("valid_loss"),
                "final_token_f1": epochs[-1].get("token_f1"),
            })

    ax_loss.set_ylabel("pérdida")
    ax_loss.set_title(title)
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(fontsize=8, loc="best")
    ax_metric.set_ylabel("métrica en validación")
    ax_metric.set_xlabel("paso de optimización")
    ax_metric.set_ylim(0, 1)
    ax_metric.grid(alpha=0.25)
    ax_metric.legend(fontsize=8, loc="best")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/validation curves for Mini-STReasoner")
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "outputs/ecgqa_small/training_log.jsonl")
    parser.add_argument("--output", type=Path, default=None,
                        help="PNG de salida (por defecto training_curves.png junto al log)")
    parser.add_argument("--run", default="latest",
                        help="'latest' (por defecto), 'all', o un id de corrida concreto")
    parser.add_argument("--title", default="Mini-STReasoner · curvas de entrenamiento y validación")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log.exists():
        raise SystemExit(f"log no encontrado: {args.log}")
    runs = group_runs(read_log(args.log))
    if not runs:
        raise SystemExit(f"sin registros utilizables en {args.log}")

    if args.run == "latest":
        key = next(reversed(runs))
        runs = OrderedDict([(key, runs[key])])
    elif args.run != "all":
        if args.run not in runs:
            raise SystemExit(f"corrida '{args.run}' no encontrada. Disponibles: {', '.join(runs)}")
        runs = OrderedDict([(args.run, runs[args.run])])

    output = args.output or args.log.with_name("training_curves.png")
    summary = plot(runs, output, args.title)
    print(json.dumps({"output": str(output), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
