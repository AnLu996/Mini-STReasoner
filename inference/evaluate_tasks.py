from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_loader import TASKS, iter_jsonl  # noqa: E402


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).lower(), flags=re.UNICODE))


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize(prediction).split()
    expected = normalize(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_file(path: Path) -> dict[str, float | int]:
    count = exact = contains = 0
    f1_sum = 0.0
    for row in iter_jsonl([path]):
        prediction = str(row.get("prediction", ""))
        answer = str(row.get("answer", ""))
        pred_norm, answer_norm = normalize(prediction), normalize(answer)
        count += 1
        exact += pred_norm == answer_norm
        contains += bool(answer_norm and answer_norm in pred_norm)
        f1_sum += token_f1(prediction, answer)
    denominator = max(count, 1)
    return {"count": count, "exact_match": exact / denominator, "accuracy": exact / denominator, "token_f1": f1_sum / denominator, "text_contains": contains / denominator}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved Mini-STReasoner predictions")
    parser.add_argument("--predictions-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/evaluation_results.json")
    args = parser.parse_args()
    results = {}
    for task in args.tasks:
        path = args.predictions_dir / f"predictions_{task}.jsonl"
        if path.exists():
            results[task] = score_file(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
