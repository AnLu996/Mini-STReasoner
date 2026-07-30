"""Linear probe cascade with control task, over the 8 audit stages.

Implements Section 5 (ledger) and Section 7 (probes) of ``brief-implementacion.md``.
The cascade fits one ridge classifier per stage on the balanced set, and one
ridge classifier per stage on a control task with permuted labels (Hewitt &
Liang, 2019). The reported quantity per stage is:

    ledger[s]        = accuracy of the probe on the real task at stage s
    ledgerControl[s] = accuracy of the probe on the control task at stage s
    selectivity[s]   = ledger[s] - ledgerControl[s]

A stage with high ledger and low selectivity has memorised the balanced set,
not extracted the target label. Selectivity is what makes the diagnosis
publishable: a colleague auditing the audit can not dismiss it as memorisation.

Design notes
------------
* Stages are the eight of the brief: ``h_raw``, ``h_bigru``, ``h_pool``,
  ``h_proj``, ``h_fusion``, ``h_llm_0``, ``h_llm_kmid``, ``h_llm_k``. The
  first four come from the newly separated hooks; the LLM hidden states use
  ``output_hidden_states=True``.
* The probe is a scikit-learn ``RidgeClassifier``. Ridge over logistic keeps
  the analysis linear and closed-form, which matches the accessibility
  claim (Alain & Bengio, 2016) and does not introduce a second optimisation
  loop that could learn the label by itself.
* The measurement is on the balanced set only. Numbers on the natural set
  are meaningless as a diagnosis of the model (they reflect A_blind, not
  representational content).

TODOs to unblock the full audit
-------------------------------
1. Build the balanced dataset per family (Sec. 3.5 of the brief) and wire it
   into ``iter_balanced_examples``. The current placeholder yields dummy
   samples so the script runs end-to-end and validates the plumbing.
2. Train and dump the untrained-encoder reference (``untrained_ledger``) as
   a floor: a stage whose probe does not beat the random-init encoder has
   learned nothing useful there.
3. Add the geometric metrics (norm ratio, effective rank, attention mass, JS
   divergence to text key-space) alongside the probe accuracies. They live
   in the same JSON export and feed V3 of the visualiser.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

# NOTE: torch and sklearn are imported lazily inside the functions that need
# them so that the module (and the CLI --help) can be inspected without a
# working torch install. This mirrors the pattern used across Mini-STReasoner.


# ---------------------------------------------------------------------------
# Ledger stages: name, hook target (see ``xai.representational_tracing``), and
# whether the text path is expected at this stage. Kept in sync with the
# ``STAGES`` tuple of ``xai/representational_tracing.py``.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerStage:
    key: str            # short id used as JSON key
    label: str          # human-readable label (used in the visualiser)
    text_flows: bool    # True if the text path is present at this stage
    llm_layer: int | None = None  # index in ``output_hidden_states`` if applicable


LEDGER_STAGES: tuple[LedgerStage, ...] = (
    LedgerStage("h_raw",     "Señal preprocesada",     False),
    LedgerStage("h_bigru",   "GRU (pre-pool)",         False),
    LedgerStage("h_pool",    "Pooling atencional",     False),
    LedgerStage("h_proj",    "Proyector latente",      False),
    LedgerStage("h_fusion",  "Fusión inputs_embeds",   True),
    LedgerStage("h_llm_0",   "LLM · capa 0",           True,  llm_layer=0),
    LedgerStage("h_llm_mid", "LLM · capa k/2",         True,  llm_layer=-2),
    LedgerStage("h_llm_k",   "LLM · capa final",       True,  llm_layer=-1),
)


# ---------------------------------------------------------------------------
# In-memory container for a single audit run.
# ---------------------------------------------------------------------------
@dataclass
class LedgerResult:
    """Result of running the cascade over one family and one configuration."""

    family: str
    config: str
    stage_keys: list[str] = field(default_factory=list)
    ledger: list[float] = field(default_factory=list)          # real task acc
    ledger_control: list[float] = field(default_factory=list)  # control task acc
    selectivity: list[float] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    seed: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "config": self.config,
            "stages": self.stage_keys,
            "ledger": self.ledger,
            "ledgerControl": self.ledger_control,
            "selectivity": self.selectivity,
            "nTrain": self.n_train,
            "nTest": self.n_test,
            "seed": self.seed,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Hook registry: extract per-stage vectors from a full forward pass.
# ---------------------------------------------------------------------------
def collect_stage_vectors(model, tokenizer, config, question: str,
                          signal: np.ndarray) -> dict[str, np.ndarray]:
    """Run one forward pass and return one pooled vector per ledger stage.

    Requires the encoder to return ``bigru_output`` (see F0 in the brief).
    LLM hidden states are taken from ``output_hidden_states=True``.
    """
    import torch  # local import; the rest of the module doesn't need torch
    from inference.runtime import build_ecg_inputs
    from xai.representational_tracing import _pool_seq

    example = {"question": question,
               "ecg_signal": [np.asarray(signal, dtype=np.float32).tolist()]}
    input_ids, attention_mask, series, time_mask = build_ecg_inputs(
        tokenizer, example, config["input_dim"]
    )

    captured: dict[str, np.ndarray] = {}
    captured["h_raw"] = np.asarray(signal, dtype=np.float32).mean(axis=0)

    def enc_hook(_m, _inp, out):
        # out = (temporal_tokens[h_pool], attention, bigru_output[h_bigru])
        captured["h_pool"] = _pool_seq(out[0].detach().float().cpu().numpy())
        if len(out) >= 3 and out[2] is not None:
            captured["h_bigru"] = _pool_seq(out[2].detach().float().cpu().numpy())

    def proj_hook(_m, _inp, out):
        captured["h_proj"] = _pool_seq(out.detach().float().cpu().numpy())

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
            captured["h_fusion"] = _pool_seq(inputs_embeds.detach().float().cpu().numpy())
            hidden = out.hidden_states
            n = len(hidden)
            captured["h_llm_0"]   = _pool_seq(hidden[0].detach().float().cpu().numpy())
            captured["h_llm_mid"] = _pool_seq(hidden[n // 2].detach().float().cpu().numpy())
            captured["h_llm_k"]   = _pool_seq(hidden[-1].detach().float().cpu().numpy())
    finally:
        h1.remove()
        h2.remove()

    # h_bigru may be missing if an older checkpoint is loaded; fill with zeros
    # so downstream loops do not crash. Selectivity will be near zero there.
    if "h_bigru" not in captured:
        captured["h_bigru"] = np.zeros_like(captured["h_pool"])
    return captured


# ---------------------------------------------------------------------------
# Fitting the probes.
# ---------------------------------------------------------------------------
def _fit_probe(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a ridge classifier. Returns a callable that predicts labels."""
    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(X, y)
    return clf.predict


def _accuracy(predict: Callable[[np.ndarray], np.ndarray],
              X: np.ndarray, y: np.ndarray) -> float:
    yhat = predict(X)
    return float((yhat == y).mean())


def run_cascade(
    features_by_stage: dict[str, np.ndarray],
    labels: np.ndarray,
    split_train: np.ndarray,
    split_test: np.ndarray,
    seed: int = 0,
    alpha: float = 1.0,
) -> LedgerResult:
    """Fit one ridge probe per stage on the real task and on the control task.

    ``features_by_stage[key]`` is expected to be an ``[N, D_key]`` matrix; N
    is the same across stages (one row per example) but ``D_key`` may differ.
    """
    rng = np.random.default_rng(seed)
    y_ctrl = rng.permutation(labels)  # etiquetas permutadas, dependencia rota

    result = LedgerResult(family="", config="", seed=seed,
                          n_train=int(split_train.sum()),
                          n_test=int(split_test.sum()))
    for stage in LEDGER_STAGES:
        X = features_by_stage.get(stage.key)
        if X is None or X.ndim != 2 or X.shape[0] != labels.shape[0]:
            result.stage_keys.append(stage.key)
            result.ledger.append(float("nan"))
            result.ledger_control.append(float("nan"))
            result.selectivity.append(float("nan"))
            continue
        Xtr, Xte = X[split_train], X[split_test]
        ytr, yte = labels[split_train], labels[split_test]
        yctr_tr, yctr_te = y_ctrl[split_train], y_ctrl[split_test]

        real = _fit_probe(Xtr, ytr, alpha=alpha)
        ctrl = _fit_probe(Xtr, yctr_tr, alpha=alpha)
        acc_real = _accuracy(real, Xte, yte)
        acc_ctrl = _accuracy(ctrl, Xte, yctr_te)

        result.stage_keys.append(stage.key)
        result.ledger.append(round(acc_real, 4))
        result.ledger_control.append(round(acc_ctrl, 4))
        result.selectivity.append(round(acc_real - acc_ctrl, 4))
    return result


# ---------------------------------------------------------------------------
# Iterating over balanced examples. Placeholder until the F1-F5 controlled
# set (Sec. 3 of the brief) is available; currently walks the small ECG-QA
# jsonl already produced by ``training.prepare_ecgqa``.
# ---------------------------------------------------------------------------
def iter_balanced_examples(family: str, split: str = "train",
                           limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``{"question", "signal", "label"}`` dicts.

    TODO: replace with a real reader over the balanced F1-F5 conjunto. The
    stub below reads existing ECG-QA small samples and treats each unique
    answer as a label; that is enough to smoke-test the cascade but is NOT
    the balanced set the ledger needs. Do not read numeric conclusions from
    a run on the stub — treat them as a plumbing test.
    """
    from training.dataset_loader import iter_jsonl
    root = Path(__file__).resolve().parents[1] / "outputs" / "ecgqa_small"
    jsonl = root / f"{split}.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(
            f"No se encontro {jsonl}. Ejecuta training/prepare_ecgqa.py o "
            f"reemplaza este iterador por el loader del conjunto balanceado F1-F5."
        )
    count = 0
    for row in iter_jsonl(jsonl):
        # TODO: filter by ``family`` once the controlled set carries it.
        if limit is not None and count >= limit:
            break
        signal_path = row.get("ecg_signal_path")
        if signal_path is None:
            continue
        signal = np.load(signal_path).astype(np.float32)
        yield {
            "question": row.get("question", ""),
            "signal":   signal,
            "label":    row.get("answer", row.get("gt", "unk")),
        }
        count += 1


# ---------------------------------------------------------------------------
# Batch runner: collect features across a split and fit the cascade.
# ---------------------------------------------------------------------------
def build_features(model, tokenizer, config, examples: Sequence[dict[str, Any]],
                   ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build the per-stage feature matrices and the label vector."""
    per_stage: dict[str, list[np.ndarray]] = {s.key: [] for s in LEDGER_STAGES}
    labels: list[str] = []
    for ex in examples:
        vecs = collect_stage_vectors(model, tokenizer, config, ex["question"], ex["signal"])
        for s in LEDGER_STAGES:
            per_stage[s.key].append(np.asarray(vecs[s.key]).ravel())
        labels.append(str(ex["label"]))

    features_by_stage = {}
    for key, seq in per_stage.items():
        if not seq:
            continue
        # Ridge classifier accepts variable-D per stage; pad within a stage
        # so it becomes a rectangular matrix.
        max_d = max(v.shape[0] for v in seq)
        M = np.zeros((len(seq), max_d), dtype=np.float32)
        for i, v in enumerate(seq):
            M[i, : v.shape[0]] = v
        features_by_stage[key] = M

    # Convert string labels to integers.
    uniq = sorted(set(labels))
    label_to_id = {l: i for i, l in enumerate(uniq)}
    y = np.array([label_to_id[l] for l in labels], dtype=np.int64)
    return features_by_stage, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Linear-probe ledger cascade over the 8 audit stages.",
    )
    parser.add_argument("--checkpoint", required=True, help="Trained MiniSTReasoner ckpt.")
    parser.add_argument("--family", default="ALL",
                        help="Family id (F1..F5) or ALL. Filters iter_balanced_examples.")
    parser.add_argument("--config-name", default="E1",
                        help="Config identifier stored in the JSON export.")
    parser.add_argument("--limit", type=int, default=200,
                        help="Cap on examples used per split (both train and test).")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/audit/ledger.json"))
    args = parser.parse_args(argv)

    # Late imports: torch + inference plumbing.
    from inference.runtime import load_base_model, load_checkpoint

    tokenizer, base_model, config = load_base_model(args.checkpoint)
    model = load_checkpoint(base_model, args.checkpoint, config)
    model.eval()

    # Collect examples. Split into train/test with a fixed seed.
    rng = np.random.default_rng(args.seed)
    train_examples = list(iter_balanced_examples(args.family, "train", limit=args.limit))
    if not train_examples:
        raise SystemExit("No hay ejemplos: revisa el conjunto o ejecuta la fase de datos.")
    idx = rng.permutation(len(train_examples))
    n_test = max(1, int(len(idx) * args.test_fraction))
    is_test = np.zeros(len(idx), dtype=bool)
    is_test[idx[:n_test]] = True
    is_train = ~is_test

    print(f"[cascade] building features on {len(train_examples)} examples...")
    features, y = build_features(model, tokenizer, config, train_examples)
    print(f"[cascade] fitting ridge probes (real + control) over {len(LEDGER_STAGES)} stages...")
    result = run_cascade(features, y, is_train, is_test, seed=args.seed)
    result.family = args.family
    result.config = args.config_name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result.to_json(), indent=2))
    print(f"[cascade] wrote {args.out}")
    for k, r, c, s in zip(result.stage_keys, result.ledger,
                           result.ledger_control, result.selectivity):
        print(f"  {k:<10}  ledger={r:.3f}  control={c:.3f}  selectivity={s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
