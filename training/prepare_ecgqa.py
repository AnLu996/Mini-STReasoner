"""Build normalized ECG-QA JSONL for the counterfactual pipeline.

Two sources are supported:

1. ``--mapped-dir``: the JSON produced by ``download_ecgqa_full.bash`` (ECG-QA
   samples enriched with ``ecg_path``). Real WFDB signals are loaded through
   :func:`training.ecgqa_loader.load_wfdb_signal`.
2. ``--synthetic-samples N``: fabricate a self-contained ECG-QA-shaped dataset
   covering every supported question type. This needs no downloads and lets the
   whole pipeline be exercised on a laptop.

Each output row follows the schema documented in ``training/ecgqa_loader.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.ecgqa_loader import (  # noqa: E402
    DEFAULT_LEADS,
    DEFAULT_STEPS,
    QUESTION_TYPES,
    load_wfdb_signal,
    resample_signal,
    synth_ecg,
)


def _first(record: dict[str, Any], *names: str, default: Any = None) -> Any:
    lowered = {str(k).lower(): k for k in record}
    for name in names:
        if name in lowered and record[lowered[name]] is not None:
            return record[lowered[name]]
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _answer_to_str(answer: Any) -> str:
    items = _as_list(answer)
    return ", ".join(str(item) for item in items) if items else str(answer or "")


def normalize_sample(
    record: dict[str, Any],
    source: str,
    steps: int,
    leads: int,
    synthetic_signals: bool,
) -> dict[str, Any] | None:
    """Map one ECG-QA record to the normalized schema, loading its signals."""
    question = str(_first(record, "question", default="")).strip()
    answer = _answer_to_str(_first(record, "answer", default=""))
    question_type = str(_first(record, "question_type", default="")).strip()
    if not question or not answer:
        return None

    ecg_ids = _as_list(_first(record, "ecg_id", default=[]))
    ecg_paths = _as_list(_first(record, "ecg_path", default=[]))
    sample_id = _first(record, "sample_id", "question_id", default=None)
    sample_tag = f"{sample_id}" if sample_id is not None else f"{abs(hash(question)) % 10**8}"
    identifier = f"{source}/{sample_tag}"

    # The "normal/abnormal" hint drives synthetic signals and conflict probes.
    answer_lower = answer.lower()
    abnormal = not any(token in answer_lower for token in ("no", "none", "normal", "sinus rhythm"))

    signals: list[list[list[float]]] = []
    used_synthetic = False
    if synthetic_signals or not ecg_paths:
        used_synthetic = True
        seeds = ecg_ids or [abs(hash(identifier)) % 10**6]
        for offset, seed in enumerate(seeds):
            signals.append(synth_ecg(int(seed) + offset, steps, leads, abnormal=abnormal))
    else:
        for path in ecg_paths:
            try:
                signals.append(load_wfdb_signal(str(path), steps, leads))
            except Exception as exc:  # noqa: BLE001 - fall back to synthetic on any IO error
                used_synthetic = True
                signals.append(
                    synth_ecg(abs(hash(str(path))) % 10**6, steps, leads, abnormal=abnormal)
                )
                print(f"[warn] WFDB load failed for {path} ({exc}); used synthetic", file=sys.stderr)

    if not signals:
        return None

    used = {"question", "answer", "question_type", "ecg_id", "ecg_path"}
    metadata = {k: v for k, v in record.items() if str(k).lower() not in used}
    metadata.update({"source": source, "synthetic_signal": used_synthetic})

    return {
        "id": identifier,
        "question": question,
        "answer": answer,
        "ecg_id": ecg_ids,
        "question_type": question_type,
        "attribute_type": str(_first(record, "attribute_type", default="")),
        "ecg_signal": signals,
        "metadata": {
            **metadata,
            "template_id": _first(record, "template_id", default=None),
            "attribute": _as_list(_first(record, "attribute", default=[])),
            "ecg_abnormal_hint": abnormal,
        },
    }


def iter_mapped(mapped_dir: Path) -> Iterator[tuple[dict[str, Any], str]]:
    for path in sorted(mapped_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload if isinstance(payload, list) else [payload]
        rel = str(path.relative_to(mapped_dir)).replace(".json", "")
        for row in rows:
            if isinstance(row, dict):
                yield row, rel
    for path in sorted(mapped_dir.rglob("*.jsonl")):
        rel = str(path.relative_to(mapped_dir)).replace(".jsonl", "")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line), rel


# Templates used to fabricate a self-contained synthetic ECG-QA dataset.
_SYNTH_TEMPLATES = [
    ("single-verify", "rhythm", "Does this ECG show {attr}?", ["yes", "no"]),
    ("single-choose", "morphology", "Which is present, {attr} or a normal pattern?", ["{attr}", "normal pattern"]),
    ("single-query", "rhythm", "What rhythm finding is shown in this ECG?", ["{attr}", "sinus rhythm"]),
    ("comparison_consecutive-verify", "morphology", "Does the second ECG show more {attr} than the first?", ["yes", "no"]),
    ("comparison_consecutive-query", "rhythm", "Which ECG shows {attr}, the first or the second?", ["the first", "the second"]),
    ("comparison_irrelevant-verify", "axis", "Do both ECGs share the same {attr}?", ["yes", "no"]),
    ("comparison_irrelevant-query", "axis", "Which ECG has an abnormal {attr}?", ["the first", "the second"]),
]
_SYNTH_ATTRIBUTES = ["atrial fibrillation", "left bundle branch block", "ST elevation", "axis deviation", "tachycardia"]


def iter_synthetic(count: int) -> Iterator[tuple[dict[str, Any], str]]:
    for i in range(count):
        template_type, attribute_type, template, answers = _SYNTH_TEMPLATES[i % len(_SYNTH_TEMPLATES)]
        attribute = _SYNTH_ATTRIBUTES[(i // len(_SYNTH_TEMPLATES)) % len(_SYNTH_ATTRIBUTES)]
        answer = answers[i % len(answers)].replace("{attr}", attribute)
        comparison = template_type.startswith("comparison")
        record = {
            "sample_id": i,
            "question_id": i,
            "template_id": i % len(_SYNTH_TEMPLATES),
            "question_type": template_type,
            "attribute_type": attribute_type,
            "attribute": [attribute],
            "question": template.replace("{attr}", attribute),
            "answer": [answer],
            "ecg_id": [1000 + i, 2000 + i] if comparison else [1000 + i],
        }
        yield record, f"synthetic/{template_type}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized ECG-QA JSONL")
    parser.add_argument("--mapped-dir", type=Path, help="ECG-QA JSON with ecg_path (from download_ecgqa_full.bash)")
    parser.add_argument("--synthetic-samples", type=int, default=0, help="Fabricate N synthetic samples instead")
    parser.add_argument("--synthetic-signals", action="store_true", help="Force synthetic signals even with a mapped dir")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/ecgqa.jsonl")
    parser.add_argument("--question-types", nargs="*", default=list(QUESTION_TYPES))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--leads", type=int, default=DEFAULT_LEADS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.mapped_dir and not args.synthetic_samples:
        parser.error("Provide either --mapped-dir or --synthetic-samples")

    rows = (
        iter_synthetic(args.synthetic_samples)
        if args.synthetic_samples
        else iter_mapped(args.mapped_dir)
    )
    allowed = set(args.question_types)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record, source in rows:
            sample = normalize_sample(
                record, source, args.steps, args.leads,
                synthetic_signals=args.synthetic_signals or bool(args.synthetic_samples),
            )
            if sample is None:
                continue
            if allowed and sample["question_type"] and sample["question_type"] not in allowed:
                continue
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[sample["question_type"] or "unknown"] += 1
            written += 1
            if args.limit and written >= args.limit:
                break

    manifest = args.output.parent / "ecgqa_manifest.json"
    manifest.write_text(json.dumps(dict(counts), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"written": written, "by_question_type": dict(counts), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
