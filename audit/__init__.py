"""Audit module: linear-probe ledger, calibrated counterfactuals and geometry.

This package implements the auditing framework described in
``brief-implementacion.md``. It is intentionally separate from the existing
``xai`` package to make the shift in methodology visible: ``xai`` measures
representational displacement (delta cosine) between original and intervened
runs; ``audit`` measures *linear accessibility* of the target label at each
stage of the pipeline, which has the same unit in every stage and is defined
where the previous metric was ``n/d``.

Modules
-------
probes
    LinearProbeCascade: register hooks at the 8 stages of the pipeline, fit a
    ridge regression per stage on the balanced set, and report accuracy plus
    a control-task accuracy (permuted labels) for selectivity.

The module is scaffolding: the training loop and CLI entrypoint are in place,
but the wiring to real data loaders and to A_oracle / A_blind lives in the
respective TODO markers.
"""

__all__ = ["probes"]
