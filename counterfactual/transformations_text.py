"""Question / text counterfactual transformations for ECG-QA.

These edits change only the textual side of a sample; the ECG signal is left
untouched. Three families are produced:

- ``question_cf``  -> meaning-preserving rewording. The clinical content is
  kept, so a correctly grounded model should give the *same* answer. A flip
  reveals sensitivity to surface form.
- ``neutral_question`` -> the clinical cue is removed, leaving a generic prompt.
- ``conflict_*`` -> a claim about the ECG (normal / abnormal) is injected into
  the question so it conflicts with the real signal, probing whether the model
  follows the text or the ECG.

Everything is rule based on purpose: the pipeline must run offline on a 6 GB
laptop without calling an external paraphrasing model.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# Light, meaning-preserving lexical swaps for question_cf.
_REWRITE_RULES: tuple[tuple[str, str], ...] = (
    (r"\bwhat\b", "which"),
    (r"\bis there\b", "can we observe"),
    (r"\bare there\b", "can we observe"),
    (r"\bdoes this ecg show\b", "is there evidence in this ecg of"),
    (r"\bdo the\b", "do these"),
    (r"\bdoes the\b", "does this"),
    (r"\bshow\b", "exhibit"),
    (r"\bpresent\b", "shown"),
    (r"\bthis ecg\b", "the recording"),
    (r"\bthe ecg\b", "the recording"),
    (r"\bsymptom\b", "finding"),
    (r"\bdiagnosis\b", "clinical finding"),
)

_NEUTRAL_TEMPLATES = (
    "What can be observed in this ECG recording?",
    "Describe this ECG recording.",
    "What is your assessment of this ECG?",
)

# Prefixes that assert a clinical state, used by the conflict probes.
_CONFLICT_PREFIX = {
    "normal": "Note: this ECG has already been read as completely normal. ",
    "abnormal": "Note: this ECG has already been flagged as clearly abnormal. ",
}


def question_cf(question: str, attribute_type: str = "", seed: int = 0) -> tuple[str, dict[str, Any]]:
    """Meaning-preserving rewrite of the question."""
    text = str(question)
    rewritten = text
    applied = 0
    for pattern, replacement in _REWRITE_RULES:
        new = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        if new != rewritten:
            applied += 1
            rewritten = new
    if rewritten.strip().lower() == text.strip().lower():
        # Guarantee a surface change even when no rule matched.
        rewritten = f"Considering the recording, {text[0].lower()}{text[1:]}" if text else text
        applied += 1
    meta = {"kind": "meaning_preserving", "rules_applied": applied}
    return rewritten, meta


def neutral_question(question: str, attribute_type: str = "", seed: int = 0) -> tuple[str, dict[str, Any]]:
    """Replace the clinical question with a neutral, cue-free prompt."""
    template = _NEUTRAL_TEMPLATES[seed % len(_NEUTRAL_TEMPLATES)]
    return template, {"kind": "neutral", "removed_question": str(question)}


def _conflict(question: str, polarity: str) -> tuple[str, dict[str, Any]]:
    prefix = _CONFLICT_PREFIX[polarity]
    return prefix + str(question), {"kind": "conflict", "claim": polarity}


def conflict_normal(question: str, attribute_type: str = "", seed: int = 0) -> tuple[str, dict[str, Any]]:
    """Inject a 'the ECG is normal' claim (used when the ECG is abnormal)."""
    return _conflict(question, "normal")


def conflict_abnormal(question: str, attribute_type: str = "", seed: int = 0) -> tuple[str, dict[str, Any]]:
    """Inject a 'the ECG is abnormal' claim (used when the ECG is normal)."""
    return _conflict(question, "abnormal")


# variant name -> builder
TEXT_TRANSFORMS: dict[str, Callable[..., tuple[str, dict[str, Any]]]] = {
    "question_cf": question_cf,
    "neutral_question": neutral_question,
    "conflict_question_normal_ecg_abnormal": conflict_normal,
    "conflict_question_abnormal_ecg_normal": conflict_abnormal,
}


def apply_text_transform(
    question: str, name: str, attribute_type: str = "", seed: int = 0
) -> tuple[str, dict[str, Any]]:
    if name not in TEXT_TRANSFORMS:
        raise KeyError(f"Unknown text counterfactual: {name}")
    return TEXT_TRANSFORMS[name](question, attribute_type=attribute_type, seed=seed)
