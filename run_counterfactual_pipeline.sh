#!/usr/bin/env bash
set -Eeuo pipefail

# run_counterfactual_pipeline.sh
# End-to-end counterfactual explainability pipeline for Mini-STReasoner on ECG-QA.
#
# Stages:
#   1. prepare ECG-QA            -> data/processed/ecgqa.jsonl
#   2. generate counterfactuals  -> outputs/counterfactuals/
#   3. model inference           -> outputs/predictions/
#   4. counterfactual metrics    -> outputs/metrics/
#   5. dominance classification  -> outputs/metrics/dominance.jsonl
#   6. case selection            -> outputs/reports/selected_cases.jsonl
#   7. aggregation               -> outputs/tables/ + outputs/reports/
#
# Defaults run fully offline with synthetic ECG-QA + a mock predictor so the
# whole chain can be validated without downloads or a GPU.
#
# Configuration via environment variables:
#   PYTHON              Python interpreter            (default: python)
#   MODEL_PATH          Checkpoint dir. If set, runs the real model instead of --mock.
#   MAPPED_DIR          ECG-QA mapped JSON dir (from download_ecgqa_full.bash).
#                       If set, real signals are used instead of synthetic ones.
#   SYNTHETIC_SAMPLES   How many synthetic samples to build when MAPPED_DIR is unset (default: 120).
#   LIMIT               Cap on number of samples processed              (default: 0 = all).
#   NO_QUANTIZATION     Set to 1 to disable 4-bit loading of the real model.
#
# Hardware-safety knobs (only relevant when MODEL_PATH is set, i.e. real GPU work):
#   DEVICE              auto | cuda | cpu. cpu draws NO GPU power (safe, slow).  (default: auto)
#   COOLDOWN            Seconds to sleep between samples to cap power/temp.      (default: 1.0)
#   POWER_LIMIT         If set (watts), runs `sudo nvidia-smi -pl POWER_LIMIT`
#                       before inference to cap the GPU's peak draw. Strongly
#                       recommended on a laptop that has shut down under load.
#   MONITOR             Set to 1 to log GPU temp/power every 5s during inference.
#
# This pipeline runs 11 inferences per sample, so the real-model stage is
# sustained GPU load. The defaults below (mock + synthetic) use NO GPU at all.
#
# Examples:
#   bash run_counterfactual_pipeline.sh                       # offline, CPU-only, safe
#   POWER_LIMIT=60 COOLDOWN=2 MODEL_PATH=checkpoints/mini_streasoner_qwen06 \
#     bash run_counterfactual_pipeline.sh                     # throttled real GPU run
#   DEVICE=cpu MODEL_PATH=checkpoints/mini_streasoner_qwen06 \
#     bash run_counterfactual_pipeline.sh                     # real model, zero GPU power

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-}"
MAPPED_DIR="${MAPPED_DIR:-}"
SYNTHETIC_SAMPLES="${SYNTHETIC_SAMPLES:-120}"
LIMIT="${LIMIT:-0}"
NO_QUANTIZATION="${NO_QUANTIZATION:-0}"
DEVICE="${DEVICE:-auto}"
COOLDOWN="${COOLDOWN:-1.0}"
POWER_LIMIT="${POWER_LIMIT:-}"
MONITOR="${MONITOR:-0}"
MONITOR_PID=""

PROCESSED="data/processed/ecgqa.jsonl"
CF="outputs/counterfactuals/counterfactuals.jsonl"
PRED="outputs/predictions/counterfactual_predictions.jsonl"

log() { printf '\n\033[1;34m[CF-PIPELINE]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[AVISO]\033[0m %s\n' "$*"; }

limit_arg() { [[ "$LIMIT" -gt 0 ]] && printf -- "--limit %s" "$LIMIT" || true; }

apply_power_limit() {
  [[ -z "$POWER_LIMIT" ]] && return 0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "POWER_LIMIT set but nvidia-smi not found; skipping."
    return 0
  fi
  log "Capping GPU power to ${POWER_LIMIT}W (sudo nvidia-smi -pl)"
  sudo nvidia-smi -pl "$POWER_LIMIT" || warn "Could not set power limit (needs sudo / supported GPU)."
}

start_monitor() {
  { [[ "$MONITOR" != "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; } && return 0
  log "GPU monitor on (temp/power every 5s)"
  ( while true; do
      nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu \
        --format=csv,noheader 2>/dev/null | sed 's/^/[GPU] /'
      sleep 5
    done ) &
  MONITOR_PID="$!"
}

stop_monitor() { [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null || true; }
trap stop_monitor EXIT

# ---------------------------------------------------------------------------
log "1/7 Preparing ECG-QA"
if [[ -n "$MAPPED_DIR" ]]; then
  $PYTHON training/prepare_ecgqa.py --mapped-dir "$MAPPED_DIR" --output "$PROCESSED" $(limit_arg)
else
  log "No MAPPED_DIR set -> building $SYNTHETIC_SAMPLES synthetic ECG-QA samples"
  $PYTHON training/prepare_ecgqa.py --synthetic-samples "$SYNTHETIC_SAMPLES" --output "$PROCESSED" $(limit_arg)
fi

# ---------------------------------------------------------------------------
log "2/7 Generating counterfactuals"
$PYTHON counterfactual/generate_counterfactuals.py --input "$PROCESSED" --output "$CF" $(limit_arg)

# ---------------------------------------------------------------------------
log "3/7 Running model inference over variants"
# --resume is always on: an interrupted run never loses finished cases.
EVAL_ARGS=(--originals "$PROCESSED" --counterfactuals "$CF" --output "$PRED" --resume)
if [[ -n "$MODEL_PATH" ]]; then
  log "Using real model at $MODEL_PATH (device=$DEVICE, cooldown=${COOLDOWN}s)"
  warn "Real GPU work ahead: ~11 inferences/sample. Stop anytime with Ctrl-C and rerun to resume."
  apply_power_limit
  start_monitor
  EVAL_ARGS+=(--model-path "$MODEL_PATH" --device "$DEVICE" --cooldown "$COOLDOWN")
  [[ "$NO_QUANTIZATION" == "1" ]] && EVAL_ARGS+=(--no-quantization)
else
  log "No MODEL_PATH set -> deterministic mock predictor (CPU-only, no GPU power)"
  EVAL_ARGS+=(--mock)
fi
$PYTHON counterfactual/run_counterfactual_eval.py "${EVAL_ARGS[@]}"
stop_monitor

# ---------------------------------------------------------------------------
log "4/7 Computing counterfactual metrics"
$PYTHON counterfactual/counterfactual_metrics.py --predictions "$PRED"

# ---------------------------------------------------------------------------
log "5/7 Classifying dominance"
$PYTHON counterfactual/classify_dominance.py

# ---------------------------------------------------------------------------
log "6/7 Selecting representative cases"
$PYTHON counterfactual/select_case_studies.py

# ---------------------------------------------------------------------------
log "7/7 Aggregating results"
$PYTHON counterfactual/aggregate_counterfactual_results.py

log "Done. Key outputs:"
echo "  predictions : $PRED"
echo "  metrics     : outputs/metrics/counterfactual_metrics.json"
echo "  dominance   : outputs/metrics/dominance_summary.json"
echo "  table       : outputs/tables/counterfactual_summary.csv"
echo "  cases       : outputs/reports/selected_cases.jsonl"
echo "  report      : outputs/reports/counterfactual_report.md"
