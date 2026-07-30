"""F02.2 · Evaluación del baseline solo-texto (E0) sobre ECG-QA.

Corre el checkpoint entrenado por ``training/train_text_only_baseline.py`` sobre
``processed_test.jsonl`` y reporta:

- Exactitud global (EM, Token-F1, sí/no).
- Desglose por ``question_type`` y por ``attribute_type``.
- ``A_blind`` por familia (mapeo directo desde ``question_type``): éste es el
  techo textual que separa el sesgo del modelo del sesgo del conjunto.

Además exporta un JSON con el mismo layout que el visualizador espera para el
campo ``blind`` en ``Condition`` (véase ``visualizer/data_contract.md``), de
modo que la vista V1 (triaje por familia) pueda consumirlo cuando E0 y E1
convivan en la misma corrida.

Uso (Ubuntu, GPU CUDA)::

    python scripts/evaluate_text_only_baseline.py \\
      --model_path checkpoints/e0_text_only \\
      --test data/ecgqa_small/processed_test.jsonl \\
      --output outputs/e0_text_only/evaluation.jsonl \\
      --blind_export outputs/e0_text_only/a_blind_by_family.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ecgqa_metrics import (  # noqa: E402
    answer_to_text,
    exact_match,
    is_valid_prediction,
    is_yesno,
    token_f1,
    yesno_correct,
)


# --------------------------------------------------------------------------- #
# Carga del E0 (LLM + LoRA · sin encoder ni proyector)                         #
# --------------------------------------------------------------------------- #
def load_e0(model_path: Path, device: str = "auto"):
    """Carga el checkpoint E0. Requiere ``config.json`` con ``model_kind='e0_text_only'``."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_kind") != "e0_text_only":
        raise ValueError(
            f"El checkpoint en {root} no es E0 (model_kind='{config.get('model_kind')}'). "
            "Usa scripts/evaluate_ecgqa_small.py para checkpoints con encoder."
        )
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", trust_remote_code=True)
    use_cpu = device == "cpu"
    quantized = config.get("qlora", False) and not use_cpu
    if use_cpu:
        kwargs: dict[str, Any] = {"torch_dtype": torch.float32, "trust_remote_code": True}
    else:
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    llm = AutoModelForCausalLM.from_pretrained(config["base_model"], **kwargs)
    if use_cpu:
        llm = llm.to("cpu")
    llm = PeftModel.from_pretrained(llm, root / "lora_adapter")
    llm.config.use_cache = True
    llm.eval()
    return tokenizer, llm, config


@torch.no_grad()
def predict_text_only(tokenizer, llm, question: str, max_new_tokens: int = 64) -> str:
    device = next(llm.parameters()).device
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = f"User: {question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


# --------------------------------------------------------------------------- #
# Agregación (idéntica a evaluate_ecgqa_small.py para permitir comparación)   #
# --------------------------------------------------------------------------- #
def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"count": 0, "exact_match": 0.0, "token_f1": 0.0, "yesno_accuracy": None, "yesno_count": 0}
    yesno = [r for r in records if r["is_yesno"]]
    return {
        "count": n,
        "exact_match": sum(r["exact_match"] for r in records) / n,
        "token_f1": sum(r["token_f1"] for r in records) / n,
        "yesno_accuracy": (sum(r["yesno_correct"] for r in yesno) / len(yesno)) if yesno else None,
        "yesno_count": len(yesno),
    }


def grouped(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record.get(key) or "unknown"].append(record)
    return {name: aggregate(items) for name, items in sorted(buckets.items())}


def write_breakdown_csv(path: Path, breakdown: dict[str, dict[str, Any]], key_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([key_name, "count", "exact_match", "token_f1", "yesno_accuracy", "yesno_count"])
        for name, metrics in breakdown.items():
            writer.writerow([
                name, metrics["count"],
                f"{metrics['exact_match']:.4f}", f"{metrics['token_f1']:.4f}",
                "" if metrics["yesno_accuracy"] is None else f"{metrics['yesno_accuracy']:.4f}",
                metrics["yesno_count"],
            ])


# --------------------------------------------------------------------------- #
# Export para el visualizador · A_blind por familia                            #
# --------------------------------------------------------------------------- #
# Mapeo de question_type de ECG-QA -> "familia" del visualizador. En cuanto se
# construya el conjunto controlado F1-F5 sobre PTB-XL+ este mapeo será directo;
# de momento las cuatro familias de ECG-QA se muestran así.
QTYPE_TO_FAMILY = {
    "single-verify":  "F_verify",
    "single-query":   "F_query",
    "single-choose":  "F_choose",
    "comparison":     "F_compare",
    "comparison-consecutive-verify": "F_compare",
    "comparison-consecutive-query":  "F_compare",
    "comparison-irrelevant-verify":  "F_compare",
    "comparison-irrelevant-query":   "F_compare",
    "comparison-irrelevant-choose":  "F_compare",
}


def export_a_blind_by_family(by_qtype: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Escribe un JSON con ``{familia: {blind, count, yesno_accuracy}}`` que el
    visualizador puede leer para el campo ``blind`` de cada Condition (V1).

    La métrica de ``blind`` es la exactitud sí/no cuando existe (interpretable en
    [0, 1] con línea base 0.5); si no, cae al Exact Match.
    """
    families: dict[str, dict[str, Any]] = {}
    aggregated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qtype, metrics in by_qtype.items():
        fam = QTYPE_TO_FAMILY.get(qtype, "F_other")
        aggregated[fam].append({"qtype": qtype, **metrics})
    for fam, entries in aggregated.items():
        total_count = sum(e["count"] for e in entries)
        if total_count == 0:
            continue
        em_weighted = sum(e["exact_match"] * e["count"] for e in entries) / total_count
        yesno_count = sum(e["yesno_count"] for e in entries)
        yesno_correct = sum(
            (e["yesno_accuracy"] or 0.0) * e["yesno_count"]
            for e in entries if e["yesno_accuracy"] is not None
        )
        yesno_acc = (yesno_correct / yesno_count) if yesno_count else None
        # blind: preferimos sí/no cuando existe; si no, EM.
        blind = yesno_acc if yesno_acc is not None else em_weighted
        families[fam] = {
            "blind": round(float(blind), 4),
            "count": int(total_count),
            "yesno_accuracy": None if yesno_acc is None else round(float(yesno_acc), 4),
            "exact_match_weighted": round(float(em_weighted), 4),
            "qtypes": [e["qtype"] for e in entries],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(families, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F02.2 · Evaluación del baseline E0 (solo texto).")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data/ecgqa_small/processed_test.jsonl")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/e0_text_only/evaluation.jsonl")
    parser.add_argument("--summary", type=Path,
                        default=PROJECT_ROOT / "outputs/e0_text_only/metrics_test.json")
    parser.add_argument("--blind_export", type=Path,
                        default=PROJECT_ROOT / "outputs/e0_text_only/a_blind_by_family.json")
    parser.add_argument("--breakdown_qtype",   type=Path,
                        default=PROJECT_ROOT / "outputs/e0_text_only/breakdown_qtype.csv")
    parser.add_argument("--breakdown_attribute", type=Path,
                        default=PROJECT_ROOT / "outputs/e0_text_only/breakdown_attribute.csv")
    parser.add_argument("--max_samples", type=int, default=750)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, llm, config = load_e0(args.model_path, args.device)
    print(f"[eval e0] loaded {args.model_path} (model_kind={config.get('model_kind')})")

    # Carga rows
    rows: list[dict[str, Any]] = []
    with args.test.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.max_samples and len(rows) >= args.max_samples:
                break
    print(f"[eval e0] test rows: {len(rows)}")

    records: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_handle:
        for i, row in enumerate(rows, 1):
            question = str(row.get("question", ""))
            gold = answer_to_text(row.get("answer", ""))
            pred = predict_text_only(tokenizer, llm, question, args.max_new_tokens)
            record = {
                "index": i,
                "question": question,
                "question_type": row.get("question_type"),
                "attribute_type": row.get("attribute_type"),
                "expected": gold,
                "prediction": pred,
                "exact_match": exact_match(pred, gold),
                "token_f1": token_f1(pred, gold),
                "is_yesno": is_yesno(gold),
                "yesno_correct": yesno_correct(pred, gold),
                "is_valid": is_valid_prediction(pred),
            }
            records.append(record)
            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"[eval e0] {i}/{len(rows)}", flush=True)

    summary = {
        "model_kind": "e0_text_only",
        "checkpoint": str(args.model_path),
        "test_file": str(args.test),
        "test_count": len(records),
        "global": aggregate(records),
        "by_qtype": grouped(records, "question_type"),
        "by_attribute": grouped(records, "attribute_type"),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_breakdown_csv(args.breakdown_qtype, summary["by_qtype"], "question_type")
    write_breakdown_csv(args.breakdown_attribute, summary["by_attribute"], "attribute_type")
    export_a_blind_by_family(summary["by_qtype"], args.blind_export)

    print(json.dumps(summary["global"], indent=2))
    print(f"[eval e0] wrote {args.summary}")
    print(f"[eval e0] wrote {args.blind_export} (A_blind por familia para el visualizador)")


if __name__ == "__main__":
    main()
