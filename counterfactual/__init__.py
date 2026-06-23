"""Counterfactual explainability pipeline for Mini-STReasoner on ECG-QA.

This package builds counterfactual variants of clinical ECG-QA samples
(question/text edits and ECG signal perturbations), runs the model on every
variant, and measures how much the answer depends on the *text* versus the
*ECG signal*. There is no graph, node, edge or spatial reasoning here: the
only structured modality is the ECG time series ``[time, leads]``.

Shared vocabulary used across the stages
----------------------------------------
- ``ORIGINAL_VARIANT``: the untouched sample.
- ``TEXT_MEANING_PRESERVING``: question rewrites that keep the clinical
  meaning, so the answer *should not* change. A flip here means the model is
  driven by surface text form.
- ``NEUTRAL_VARIANT``: the question is stripped of its clinical cue.
- ``CONFLICT_VARIANTS``: the question is pushed to claim ``normal`` or
  ``abnormal`` against the actual ECG, to see whether the model follows the
  text claim or the signal.
- ``ECG_VARIANTS``: pure signal perturbations, the text is left untouched.
"""

from __future__ import annotations

ORIGINAL_VARIANT = "original"

# Meaning-preserving question rewrites (answer should stay the same).
TEXT_MEANING_PRESERVING = ("question_cf",)

# Question stripped of clinical content.
NEUTRAL_VARIANT = "neutral_question"

# Conflict probes -> claim polarity injected into the question.
CONFLICT_VARIANTS = {
    "conflict_question_normal_ecg_abnormal": "normal",
    "conflict_question_abnormal_ecg_normal": "abnormal",
}

# Pure ECG signal perturbations (text untouched).
ECG_VARIANTS = (
    "ecg_cf_noise",
    "ecg_cf_scaling",
    "ecg_cf_lead_mask",
    "ecg_cf_time_mask",
    "ecg_cf_spike",
    "ecg_cf_shuffle",
)

# Every text-side variant (used by generators / metrics).
TEXT_VARIANTS = TEXT_MEANING_PRESERVING + (NEUTRAL_VARIANT,) + tuple(CONFLICT_VARIANTS)

# All non-original variants, in a stable display order.
ALL_VARIANTS = TEXT_VARIANTS + ECG_VARIANTS

DOMINANCE_CLASSES = (
    "TEXT_DOMINANT",
    "ECG_DOMINANT",
    "BALANCED",
    "UNSTABLE",
    "UNCLEAR",
)

# Thresholds for per-case dominance classification. Tunable in one place.
DOMINANCE_THRESHOLDS = {
    "inactive": 0.25,   # below this a modality is considered "did not move"
    "high": 0.80,       # above this for both -> the model flips on everything
    "margin": 0.30,     # text_score - ecg_score gap that decides dominance
}

__all__ = [
    "ORIGINAL_VARIANT",
    "TEXT_MEANING_PRESERVING",
    "NEUTRAL_VARIANT",
    "CONFLICT_VARIANTS",
    "ECG_VARIANTS",
    "TEXT_VARIANTS",
    "ALL_VARIANTS",
    "DOMINANCE_CLASSES",
    "DOMINANCE_THRESHOLDS",
]
