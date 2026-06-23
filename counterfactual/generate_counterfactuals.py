"""Generate counterfactual variants for ECG-QA samples.

For every original sample this writes one record describing all variants:

- ``original``               : untouched.
- ``question_cf``            : meaning-preserving question rewrite.
- ``neutral_question``       : clinical cue removed.
- ``conflict_*``             : a normal/abnormal claim injected into the text.
- ``ecg_cf_*``               : signal perturbations (noise, scaling, lead mask,
                               time mask, spike, time shuffle).

Text variants store the new question inline (small). ECG variants store only a
transform *spec* (name + params + seed) and reference the original signal by
id, so the counterfactual file stays small and fully reproducible. The actual
perturbed signals are materialised at evaluation time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counterfactual import (  # noqa: E402
    CONFLICT_VARIANTS,
    ECG_VARIANTS,
    NEUTRAL_VARIANT,
    TEXT_MEANING_PRESERVING,
)
from counterfactual.transformations_ecg import default_params  # noqa: E402
from counterfactual.transformations_text import apply_text_transform  # noqa: E402
from training.ecgqa_loader import iter_ecgqa  # noqa: E402


def stable_seed(text: str) -> int:
    """Process-independent seed derived from a string."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_variants(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    question = str(sample.get("question", ""))
    attribute_type = str(sample.get("attribute_type", ""))
    abnormal = bool(sample.get("metadata", {}).get("ecg_abnormal_hint", True))
    seed = stable_seed(sample["id"])

    variants: dict[str, dict[str, Any]] = {
        "original": {"type": "original", "question": question},
    }

    # Text-side counterfactuals (question unchanged signal).
    for name in (*TEXT_MEANING_PRESERVING, NEUTRAL_VARIANT, *CONFLICT_VARIANTS):
        new_question, meta = apply_text_transform(question, name, attribute_type, seed)
        entry = {"type": "text", "question": new_question, "meta": meta}
        if name in CONFLICT_VARIANTS:
            claim = CONFLICT_VARIANTS[name]
            entry["claim_polarity"] = claim
            # Whether this conflict actually contradicts the ECG.
            entry["meta"]["contradicts_ecg"] = (
                (claim == "normal" and abnormal) or (claim == "abnormal" and not abnormal)
            )
        variants[name] = entry

    # ECG-side counterfactuals (signal perturbed, question unchanged).
    for name in ECG_VARIANTS:
        variants[name] = {
            "type": "ecg",
            "question": question,
            "transform": {"name": name, "params": default_params(name), "seed": seed},
        }
    return variants


def iter_records(source: Path, limit: int) -> Iterator[dict[str, Any]]:
    for index, sample in enumerate(iter_ecgqa([source])):
        if limit and index >= limit:
            break
        yield {
            "id": sample["id"],
            "question_type": sample.get("question_type", ""),
            "attribute_type": sample.get("attribute_type", ""),
            "answer": sample.get("answer", ""),
            "original_question": sample.get("question", ""),
            "ecg_abnormal_hint": bool(sample.get("metadata", {}).get("ecg_abnormal_hint", True)),
            "variants": build_variants(sample),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ECG-QA counterfactuals")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/processed/ecgqa.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/counterfactuals/counterfactuals.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in iter_records(args.input, args.limit):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"counterfactual_records": written, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
