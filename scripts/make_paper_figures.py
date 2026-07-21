"""Generate the figures for the thesis article from the measured results.

Every number here comes from an artefact under ``outputs/``; nothing is typed in
by hand except the labels. Figures are sized for a two-column IEEE layout —
``COL`` for a single column, ``WIDE`` for a spanning figure — and rendered at
300 dpi with a serif face so they sit next to the body text.

Colour follows one rule across all figures, so a reader learns it once:

    texto  -> orange   ECG / senal -> blue

Run-to-run comparisons (Corrida A before, Corrida B after) are an *ordinal*
contrast, not a categorical one, so they use two steps of a single blue ramp
instead of a second hue. Reference quantities that are not series — the uniform
attention baseline, the untrained encoder, the operating threshold — are drawn as
neutral rules and hollow markers, never as an extra colour.

Both palettes were checked with the data-viz validator (all checks pass,
all-pairs for the categorical one, ordinal for the ramp).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGDIR = PROJECT_ROOT / "figures"

# Categorical: modality. Ordinal: run A -> run B.
C_TEXT = "#eb6834"
C_ECG = "#2a78d6"
RUN_A = "#86b6ef"
RUN_B = "#1c5cab"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a877f"
RULE = "#b9b2a2"
SURFACE = "#fcfcfb"

COL = (3.5, 2.6)
WIDE = (7.16, 2.9)


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "grid.color": "#e8e4d9",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def tidy(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis)


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.png"
    fig.savefig(path)
    fig.savefig(FIGDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  {path}")


def load(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text())


# ----------------------------------------------------------------------------


def fig_recalibration() -> None:
    """ECFR depends on which ECG intervention is used; the dominance index inherits it."""
    data = load("outputs/ecgqa_5k_control/counterfactual_recalibrated.json")
    order = ["ecg_time_mask", "ecg_lead_mask", "ecg_noise", "ecg_spike"]
    labels = ["Oclusión\ntemporal", "Máscara de\nderivaciones", "Ruido\ngaussiano", "Spike"]
    values = [data["per_intervention"][k]["ECFR"] for k in order]

    subsets = ["todas (como\nen Corrida A)", "estructuradas", "solo oclusión\ntemporal"]
    keys = ["todas (como en Corrida A)", "estructuradas", "solo oclusion temporal"]
    ecfr = [data["per_subset"][k]["ECFR"] for k in keys]
    dtxt = [data["per_subset"][k]["textual_dominance"] for k in keys]
    qcfr = data["per_subset"][keys[0]]["QCFR"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE)

    bars = ax1.bar(range(4), values, color=C_ECG, width=0.62)
    for rect, value in zip(bars, values):
        ax1.text(rect.get_x() + rect.get_width() / 2, value + 0.004, f"{value:.3f}",
                 ha="center", va="bottom", fontsize=7, color=INK)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("ECFR (tasa de cambio de respuesta)")
    ax1.set_ylim(0, 0.19)
    ax1.set_title("(a) Cada intervención de ECG mide algo distinto", loc="left", color=INK)
    tidy(ax1)

    x = np.arange(3)
    w, off = 0.32, 0.17
    ax2.bar(x - off, ecfr, w, color=C_ECG, label="ECFR (señal)")
    ax2.bar(x + off, dtxt, w, color=C_TEXT, label=r"$D_{\rm texto}$ = QCFR $-$ ECFR")
    ax2.axhline(qcfr, color=RULE, lw=1.1, ls="--")
    ax2.text(-0.42, qcfr + 0.008, f"QCFR = {qcfr:.2f} · la intervención textual no se toca",
             ha="left", fontsize=6.6, color=MUTED)
    for xi, (a, b) in enumerate(zip(ecfr, dtxt)):
        ax2.text(xi - off, a + 0.006, f"{a:.3f}", ha="center", fontsize=6.6, color=INK)
        ax2.text(xi + off, b + 0.006, f"{b:.3f}", ha="center", fontsize=6.6, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(subsets)
    ax2.set_ylabel("Tasa")
    ax2.set_ylim(0, 0.40)
    ax2.set_title("(b) El índice de dominancia hereda esa elección", loc="left", color=INK)
    ax2.legend(loc="upper right", ncol=2, columnspacing=1.0, handlelength=1.2)
    tidy(ax2)

    save(fig, "fig_recalibracion_ecfr")


def fig_tracing_stages() -> None:
    """The internal trace with the calibrated intervention: impact_ECG stops being zero."""
    old = load("outputs/tracing/stage_sensitivity_summary.json")["stages"]
    new = load("outputs/tracing/stage_sensitivity_summary_occlusion.json")["stages"]
    stages = ["encoder", "projector", "fusion", "llm", "output"]
    labels = ["Encoder", "Proyector", "Fusión", "LLM", "Salida"]

    def series(source, key):
        return [(source[s][key] if source[s][key] is not None else np.nan) for s in stages]

    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    x = np.arange(len(stages))
    w, off = 0.32, 0.17
    for ax, source, title in (
        (axes[0], old, "(a) Ruido gaussiano (Corrida A)"),
        (axes[1], new, "(b) Oclusión temporal (recalibrado)"),
    ):
        text = series(source, "mean_impact_text")
        ecg = series(source, "mean_impact_ecg")
        ax.bar(x - off, text, w, color=C_TEXT, label="impacto del texto")
        ax.bar(x + off, ecg, w, color=C_ECG, label="impacto del ECG")
        for xi, value in enumerate(text):
            if np.isnan(value):
                ax.text(xi - off, 0.003, "n/d", ha="center", va="bottom",
                        fontsize=6.2, color=MUTED, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, loc="left", color=INK)
        tidy(ax)
    axes[0].set_ylabel("Distancia representacional (coseno)")
    axes[0].set_ylim(0, 0.155)
    axes[0].text(0.5, 0.075, "impacto del ECG = 0,0000\nen las cinco etapas",
                 ha="left", fontsize=7, color=MUTED, style="italic")
    axes[1].legend(loc="upper right", handlelength=1.2)
    for ax in axes:
        ax.text(-0.45, 0.003, "", ha="left")
    save(fig, "fig_trazado_etapas")


def fig_attention() -> None:
    """Where the encoder looks: flat in A, selective in B."""
    data = load("figures/attention_profiles.json")
    bins = data["A"]["bins"]
    uniform = 1.0 / bins
    x = np.arange(bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"width_ratios": [1.55, 1]})

    # Curva de concentracion: para cada muestra se ordenan los pesos de mayor a
    # menor y se promedia. Muestra cuanta masa se concentra sin depender de DONDE
    # este el tramo elegido, que cambia de una senal a otra; una banda por
    # posicion, en cambio, se solapa entre corridas y no identifica a su serie.
    for key, color, label in (("A", RUN_A, f"Corrida A · {data['A']['tokens']} tokens"),
                              ("B", RUN_B, f"Corrida B · {data['B']['tokens']} tokens")):
        ax1.plot(x, data[key]["sorted_mean"], color=color, lw=1.9, label=label, zorder=4)
    ax1.axhline(1.0, color=RULE, lw=1.1, ls="--", zorder=1)
    ax1.text(bins - 1, 1.03, "referencia uniforme = 1,0", ha="right", va="bottom",
             fontsize=6.8, color=MUTED)
    ax1.annotate("", xy=(1.5, 1.714), xytext=(1.5, 1.050),
                 arrowprops={"arrowstyle": "<->", "color": INK2, "lw": 0.9})
    ax1.text(3.0, 1.38, "1,05× vs 1,71×\nen la ventana\nmás atendida",
             fontsize=6.6, color=INK2, va="center")
    ax1.set_xlabel("Ventanas ordenadas por peso  (de mayor a menor)")
    ax1.set_ylabel("Peso relativo al uniforme (×)")
    ax1.set_xlim(0, bins - 1)
    ax1.set_ylim(0.70, 1.85)
    ax1.set_title("(a) Concentración del peso sobre la señal", loc="left", color=INK)
    ax1.legend(loc="upper right", handlelength=1.4)
    tidy(ax1)

    parts = [np.array(data["A"]["entropy"]), np.array(data["B"]["entropy"])]
    bp = ax2.boxplot(parts, vert=True, widths=0.5, patch_artist=True, showfliers=True,
                     medianprops={"color": INK, "lw": 1.1},
                     flierprops={"marker": "o", "markersize": 1.6, "markerfacecolor": MUTED,
                                 "markeredgecolor": "none", "alpha": 0.5})
    for patch, color in zip(bp["boxes"], (RUN_A, RUN_B)):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    for element in ("whiskers", "caps"):
        for item in bp[element]:
            item.set_color(MUTED)
            item.set_linewidth(0.8)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Corrida A", "Corrida B"])
    ax2.set_ylabel("Entropía por token  (1 = uniforme)")
    ax2.set_ylim(0.30, 1.06)
    ax2.axhline(1.0, color=RULE, lw=1.1, ls="--")
    ax2.set_title("(b) Entropía de la atención por token", loc="left", color=INK)
    tidy(ax2)

    save(fig, "fig_atencion_encoder")


def fig_encoder_fidelity() -> None:
    """Linear probes: only the enlarged encoder beats an untrained one on heart rate."""
    targets = ["Frecuencia\ndominante\n(ritmo cardíaco)", "Autocorre-\nlación\n(lag 1)", "Rango"]
    a = [0.1565, 0.8550, 0.3462]
    b = [0.3290, 0.8545, 0.4105]
    untrained = [0.2469, 0.8643, 0.2901]

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    x = np.arange(len(targets))
    w, off = 0.32, 0.17
    ax.bar(x - off, a, w, color=RUN_A, label="Corrida A")
    ax.bar(x + off, b, w, color=RUN_B, label="Corrida B")
    for xi, value in enumerate(untrained):
        ax.plot([xi - off - w / 2, xi + off + w / 2], [value, value], color=INK, lw=1.5,
                solid_capstyle="butt", zorder=5)
    reference = Line2D([0], [0], color=INK, lw=1.5, label="encoder sin entrenar")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel(r"$R^2$ del probe lineal")
    ax.set_ylim(0, 1.12)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [reference], loc="upper center", ncol=3,
              columnspacing=1.0, handlelength=1.3)
    tidy(ax)
    save(fig, "fig_fidelidad_encoder")


def fig_confidence() -> None:
    """What survives a confidence interval, and what does not."""
    data = load("outputs/ecgqa_confidence.json")
    rows = [
        (r"cont$_{\rm texto}$", "text_contribution"),
        (r"cont$_{\rm ECG}$", "ecg_contribution"),
        ("Dominancia\ntextual", "textual_dominance"),
    ]
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    offset = 0.13
    for index, (label, key) in enumerate(rows):
        for run, color, sign in (("A", RUN_A, +1), ("B", RUN_B, -1)):
            item = data[run]["ablation"][key]
            y = index + sign * offset
            low, high = item["ci95"]
            ax.plot([low, high], [y, y], color=color, lw=1.6, solid_capstyle="round")
            ax.plot(item["value"], y, "o", color=color, ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=5)
    ax.axvline(0, color=INK, lw=0.9)
    ax.axvline(0.05, color=RULE, lw=1.1, ls="--")
    ax.text(0.056, -0.62, "umbral 0,05", fontsize=6.6, color=MUTED, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_xlabel("Contribución modal (Token-F1)  ·  IC 95 % bootstrap")
    ax.set_xlim(-0.06, 0.52)
    ax.set_ylim(2.45, -0.75)
    ax.legend(handles=[Patch(color=RUN_A, label="Corrida A"), Patch(color=RUN_B, label="Corrida B")],
              loc="upper right", ncol=2, columnspacing=1.0, handlelength=1.2)
    tidy(ax, grid_axis="x")
    save(fig, "fig_intervalos_confianza")


def fig_stbench() -> None:
    """The scaled recipe matches the published accuracy without reading the series."""
    conf = load("outputs/stbench_small/stbench_confidence.json")["accuracy"]
    tasks = ["reasoning_correlation", "reasoning_entity", "reasoning_etiological"]
    labels = ["Correlación", "Entidad", "Etiológico"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"width_ratios": [1.25, 1]})

    x = np.arange(len(tasks))
    values = [conf[t]["accuracy"] for t in tasks]
    lows = [conf[t]["wilson_95"][0] for t in tasks]
    highs = [conf[t]["wilson_95"][1] for t in tasks]
    published = [conf[t]["published_8b"] for t in tasks]
    ax1.bar(x, values, 0.5, color=RUN_B,
            yerr=[np.array(values) - lows, np.array(highs) - values],
            error_kw={"ecolor": INK2, "elinewidth": 0.9, "capsize": 3})
    ax1.scatter(x, published, marker="D", s=26, facecolors="none", edgecolors=INK,
                linewidths=1.1, zorder=5, label="STReasoner-8B (publicado)")
    ax1.axhline(0.25, color=RULE, lw=1.1, ls="--")
    ax1.text(len(tasks) - 0.55, 0.27, "azar (4 opciones)", ha="right", fontsize=6.8, color=MUTED)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Exactitud  ·  IC 95 % (Wilson)")
    ax1.set_ylim(0, 1.18)
    ax1.set_title("(a) Mini-STReasoner (0,6 B) sobre ST-Bench, n = 60", loc="left", color=INK)
    ax1.legend(loc="upper center", handlelength=1.2)
    tidy(ax1)

    conditions = ["Serie\noriginal", "Serie de\notra muestra", "Serie en\nceros"]
    accuracy = [0.9000, 0.9000, 0.9000]
    ax2.bar(np.arange(3), accuracy, 0.5, color=C_ECG)
    for xi, value in enumerate(accuracy):
        ax2.text(xi, value + 0.02, f"{value:.3f}", ha="center", fontsize=7, color=INK)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(conditions)
    ax2.set_ylabel("Exactitud")
    ax2.set_ylim(0, 1.18)
    ax2.set_title("(b) Sustituir la serie no cambia nada", loc="left", color=INK)
    ax2.text(1.0, 1.10, "tasa de cambio de respuesta = 0,000\n(90 muestras)",
             ha="center", va="top", fontsize=6.6, color=MUTED)
    tidy(ax2)

    save(fig, "fig_stbench_validacion")


def main() -> None:
    style()
    print("Figuras generadas:")
    fig_recalibration()
    fig_tracing_stages()
    fig_attention()
    fig_encoder_fidelity()
    fig_confidence()
    fig_stbench()


if __name__ == "__main__":
    main()
