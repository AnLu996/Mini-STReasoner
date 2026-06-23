"""Shared, torch-free text metrics for the small ECG-QA runs.

Kept dependency-free (only the standard library) so every stage -- baseline
inference, training-time validation, evaluation and the counterfactual run --
scores answers exactly the same way. ECG-QA answers are *lists* (usually a single
token such as ``["yes"]`` or ``["sinus rhythm"]``); :func:`answer_to_text`
flattens them to the gold string used by every metric here.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def normalize(text: Any) -> str:
    """Lowercase, unicode-aware, whitespace-collapsed token string."""
    return " ".join(re.findall(r"\w+", str(text).lower(), flags=re.UNICODE))


def answer_to_text(answer: Any) -> str:
    """Flatten an ECG-QA answer (list / scalar) into a single gold string."""
    if isinstance(answer, (list, tuple)):
        return ", ".join(str(item) for item in answer)
    return str(answer if answer is not None else "")


def exact_match(prediction: Any, gold: Any) -> float:
    """1.0 when the normalized prediction equals the normalized gold answer."""
    return float(normalize(prediction) == normalize(gold))


def token_f1(prediction: Any, gold: Any) -> float:
    """Token-overlap F1 between prediction and gold (SQuAD-style)."""
    predicted = normalize(prediction).split()
    expected = normalize(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def is_yesno(gold: Any) -> bool:
    """True when the gold answer is a yes/no answer."""
    return normalize(gold) in {"yes", "no"}


def yesno_correct(prediction: Any, gold: Any) -> float:
    """1.0 when the prediction expresses the same yes/no polarity as the gold.

    Generous on the prediction side (it may be a longer phrase) but requires an
    unambiguous polarity: a prediction containing both "yes" and "no" scores 0.
    """
    gold_norm = normalize(gold)
    tokens = set(normalize(prediction).split())
    if gold_norm == "yes":
        return float("yes" in tokens and "no" not in tokens)
    if gold_norm == "no":
        return float("no" in tokens and "yes" not in tokens)
    return 0.0


def is_valid_prediction(prediction: Any) -> bool:
    """A prediction counts as valid when it has at least one word token."""
    return bool(normalize(prediction))
