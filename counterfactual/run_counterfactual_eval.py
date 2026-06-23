"""Run the model on every counterfactual variant and record predictions.

Inputs:
- ``--originals``        : normalized ECG-QA JSONL (carries the real signals).
- ``--counterfactuals``  : output of ``generate_counterfactuals.py``.

For each variant the perturbed example is materialised (text variants reuse the
original signal; ECG variants apply their transform spec) and fed to the model.

Two back-ends:
- real model via ``inference.runtime`` (needs a trained checkpoint + GPU);
- ``--mock`` : a deterministic, signal-driven heuristic so the *whole* metric /
  classification / aggregation chain can be exercised without a GPU. The mock
  reads only what the model can see (question + signal), never the gold answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual.transformations_ecg import apply_ecg_transform  # noqa: E402
from training.ecgqa_loader import iter_ecgqa, merge_signals  # noqa: E402


def load_originals(path: Path) -> dict[str, dict[str, Any]]:
    return {sample["id"]: sample for sample in iter_ecgqa([path])}


def materialise_signal(original: dict[str, Any], variant: dict[str, Any]) -> list[list[list[float]]]:
    """Return the (possibly perturbed) list of ECG signals for a variant."""
    signals = original.get("ecg_signal", [])
    if variant.get("type") == "ecg":
        spec = variant["transform"]
        return apply_ecg_transform(signals, spec["name"], spec.get("params"), spec.get("seed", 0))
    return signals


def _signal_mean(signals: list[list[list[float]]]) -> float:
    merged = merge_signals(signals)
    if not merged:
        return 0.0
    total = 0.0
    count = 0
    for row in merged:
        for value in row:
            total += float(value)
            count += 1
    return total / max(count, 1)


def mock_predict(example: dict[str, Any], variant_name: str, variant: dict[str, Any]) -> str:
    """Deterministic heuristic predictor used for plumbing tests.

    Decision = sign of (signal abnormality score + text bias). It mirrors a
    model that mostly reads the ECG but is partly swayed by explicit textual
    claims, so the resulting metrics are non-trivial.
    """
    question = str(example.get("question", "")).lower()
    score = _signal_mean(example.get("ecg_signal", [])) - 0.45  # >0 -> abnormal

    # Neutral prompt: no clinical anchor, the model defaults to "normal".
    if variant_name == "neutral_question":
        score -= 0.5

    # Explicit textual claim partly pulls the decision toward the claim.
    claim = variant.get("claim_polarity")
    if claim == "normal":
        score -= 0.18
    elif claim == "abnormal":
        score += 0.18

    abnormal = score > 0.0
    verify = "does" in question or "do " in question or "share" in question
    if verify:
        return "yes" if abnormal else "no"
    return "abnormal finding" if abnormal else "normal pattern"


def predict_variants(
    record: dict[str, Any],
    original: dict[str, Any],
    backend,
) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for name, variant in record["variants"].items():
        example = {
            "question": variant.get("question", original.get("question", "")),
            "ecg_signal": materialise_signal(original, variant),
        }
        predictions[name] = backend(example, name, variant)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ECG-QA counterfactuals")
    parser.add_argument("--originals", type=Path, default=PROJECT_ROOT / "data/processed/ecgqa.jsonl")
    parser.add_argument("--counterfactuals", type=Path, default=PROJECT_ROOT / "outputs/counterfactuals/counterfactuals.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/predictions/counterfactual_predictions.jsonl")
    parser.add_argument("--model-path", type=Path, help="Checkpoint dir (required unless --mock)")
    parser.add_argument("--mock", action="store_true", help="Use the deterministic heuristic predictor")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="cpu = no GPU power draw (safe but slow)")
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="Seconds to sleep after each sample to keep GPU power/temperature down")
    parser.add_argument("--resume", action="store_true",
                        help="Skip ids already present in the output and append (crash-safe)")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    originals = load_originals(args.originals)

    # Resume support: never lose finished work if a run is interrupted.
    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        print(f"[resume] {len(done_ids)} cases already done, will skip them")

    if args.mock:
        backend = mock_predict
    else:
        if not args.model_path:
            parser.error("--model-path is required unless --mock is set")
        from inference.runtime import load_checkpoint, predict_ecg  # noqa: E402

        tokenizer, model, config = load_checkpoint(
            args.model_path, not args.no_quantization, device=args.device
        )

        def backend(example, name, variant):  # noqa: ARG001 - signature shared with mock
            return predict_ecg(tokenizer, model, config, example, args.max_new_tokens)

    real_model = not args.mock
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    open_mode = "a" if (args.resume and args.output.exists()) else "w"
    with args.counterfactuals.open("r", encoding="utf-8") as source, args.output.open(open_mode, encoding="utf-8") as sink:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            if args.limit and index >= args.limit:
                break
            record = json.loads(line)
            if record["id"] in done_ids:
                continue
            original = originals.get(record["id"])
            if original is None:
                print(f"[warn] no original signal for {record['id']}, skipping", file=sys.stderr)
                continue
            predictions = predict_variants(record, original, backend)
            sink.write(json.dumps({
                "id": record["id"],
                "question_type": record.get("question_type", ""),
                "attribute_type": record.get("attribute_type", ""),
                "answer": record.get("answer", ""),
                "ecg_abnormal_hint": record.get("ecg_abnormal_hint"),
                "predictions": predictions,
                "variants_meta": {
                    name: {k: v for k, v in variant.items() if k in ("claim_polarity", "meta", "type")}
                    for name, variant in record["variants"].items()
                },
            }, ensure_ascii=False) + "\n")
            sink.flush()  # persist each case so an interruption never loses finished work
            written += 1
            if index % 25 == 0:
                print(f"[{index}] {record['id']}")
            # Let the GPU cool / power settle between samples on the real model.
            if real_model and args.cooldown > 0:
                time.sleep(args.cooldown)

    print(json.dumps({"evaluated": written, "output": str(args.output), "backend": "mock" if args.mock else "model"}, indent=2))


if __name__ == "__main__":
    main()
