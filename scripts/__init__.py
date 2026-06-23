"""Small, controlled ECG-QA experiment scripts for Mini-STReasoner.

These scripts run a reproducible PTB-XL ECG-QA subset end to end (download ->
prepare signals -> baseline inference -> LoRA training -> evaluation ->
counterfactuals) so preliminary, paper-ready numbers can be obtained on a
limited laptop without touching the full ECG-QA / MIMIC-IV-ECG datasets.
"""
