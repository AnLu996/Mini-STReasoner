#!/usr/bin/env bash
set -Eeuo pipefail

# run_ecgqa_small_pipeline.sh
# Controlled, reproducible ECG-QA (PTB-XL) small run for Mini-STReasoner.
# Each stage stops the pipeline on error (set -e) and tees its console output to
# outputs/ecgqa_small/logs/. Stages are individually re-runnable: download and
# prepare skip already-finished work, so a failed run can simply be relaunched.
#
# Configuration via environment variables (all optional):
#   PYTHON            Python interpreter                         (default: python)
#   OUTPUT_DIR        Dataset dir                                (default: data/ecgqa_small)
#   RESULTS_DIR       Results dir                                (default: outputs/ecgqa_small)
#   CKPT_DIR          Checkpoint dir                             (default: checkpoints/ecgqa_small_lora)
#   SUBSET            train | valid | test | all                 (default: all)
#   MAX_QUESTIONS     Questions to sample                        (default: 300)
#   MAX_UNIQUE_ECGS   Unique ECGs to download                    (default: 100)
#   SEED              Random seed                                (default: 42)
#   TARGET_LENGTH     Resampled ECG length                       (default: 1000)
#   MAX_LEADS         ECG leads                                  (default: 12)
#   INFER_SAMPLES     Stage-3 baseline samples                   (default: 20)
#   TRAIN_SAMPLES     Stage-4 training samples                   (default: 300)
#   VALID_SAMPLES     Stage-4 validation samples per epoch       (default: 50)
#   EVAL_SAMPLES      Stage-5 test samples                       (default: 100)
#   CF_SAMPLES        Stage-6 counterfactual samples             (default: 50)
#   ABL_SAMPLES       Stage-6b modal-ablation samples            (default: 100)
#   ATTR_SAMPLES      Stage-7b real-attribution samples (V3/V4)  (default: 30)
#   TRACE_SAMPLES     Stage-7c representational-tracing samples   (default: 30)
#   TRACE_METRIC      cosine | l2 (representational distance)     (default: cosine)
#   TRACE_SEGMENTS    ECG segment interventions per case (V4)     (default: 6)
#   EPOCHS            Training epochs                            (default: 1)
#   PATIENCE          Early-stopping patience (0 = off)          (default: 3)
#   EARLY_STOP_METRIC valid_loss | token_f1 | exact_match        (default: valid_loss)
#   LR_SCHEDULER      cosine | linear | constant                 (default: cosine)
#   WARMUP_RATIO      Warmup fraction of total steps             (default: 0.06)
#   BATCH_SIZE        Training batch size                        (default: 1)
#   GRAD_ACCUM        Gradient accumulation steps                (default: 8)
#   MAX_SEQ_LEN       Max sequence length                        (default: 512)
#   DEVICE            auto | cuda | cpu (cpu = no GPU power)      (default: auto)
#   NO_QLORA          1 to disable 4-bit training                (default: unset)
#
# NOTE: SUBSET defaults to "all" so Stage 2 emits processed_{train,valid,test}.jsonl,
# which Stages 4-6 consume. Use SUBSET=train for a single-split download.

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/ecgqa_small}"
RESULTS_DIR="${RESULTS_DIR:-outputs/ecgqa_small}"
CKPT_DIR="${CKPT_DIR:-checkpoints/ecgqa_small_lora}"
SUBSET="${SUBSET:-all}"
MAX_QUESTIONS="${MAX_QUESTIONS:-300}"
MAX_UNIQUE_ECGS="${MAX_UNIQUE_ECGS:-100}"
SEED="${SEED:-42}"
TARGET_LENGTH="${TARGET_LENGTH:-1000}"
MAX_LEADS="${MAX_LEADS:-12}"
INFER_SAMPLES="${INFER_SAMPLES:-20}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-300}"
VALID_SAMPLES="${VALID_SAMPLES:-50}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
CF_SAMPLES="${CF_SAMPLES:-50}"
ABL_SAMPLES="${ABL_SAMPLES:-100}"
ATTR_SAMPLES="${ATTR_SAMPLES:-30}"
TRACE_SAMPLES="${TRACE_SAMPLES:-30}"
TRACE_METRIC="${TRACE_METRIC:-cosine}"
TRACE_SEGMENTS="${TRACE_SEGMENTS:-6}"
EPOCHS="${EPOCHS:-1}"
PATIENCE="${PATIENCE:-3}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-valid_loss}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
DEVICE="${DEVICE:-auto}"

QLORA_FLAG=()
[[ "${NO_QLORA:-}" == "1" ]] && QLORA_FLAG=(--no_qlora)

LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

run_stage() {
  local name="$1"; shift
  echo ""
  echo "=============================================================="
  echo ">> $name"
  echo "=============================================================="
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

run_stage "stage1_download" "$PYTHON" scripts/download_ecgqa_small.py \
  --subset "$SUBSET" \
  --max_questions "$MAX_QUESTIONS" \
  --max_unique_ecgs "$MAX_UNIQUE_ECGS" \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"

run_stage "stage2_prepare" "$PYTHON" scripts/prepare_ecg_signals.py \
  --manifest "$OUTPUT_DIR/manifest.jsonl" \
  --output "$OUTPUT_DIR/processed.jsonl" \
  --target_length "$TARGET_LENGTH" \
  --max_leads "$MAX_LEADS"

run_stage "stage3_inference" "$PYTHON" scripts/run_ecgqa_inference_small.py \
  --data "$OUTPUT_DIR/processed.jsonl" \
  --max_samples "$INFER_SAMPLES" \
  --max_leads "$MAX_LEADS" \
  --device "$DEVICE" \
  --output "$RESULTS_DIR/inference_raw.jsonl"

run_stage "stage4_train" "$PYTHON" training/train_ecgqa_lora_small.py \
  --train "$OUTPUT_DIR/processed_train.jsonl" \
  --valid "$OUTPUT_DIR/processed_valid.jsonl" \
  --output_dir "$CKPT_DIR" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --early_stop_metric "$EARLY_STOP_METRIC" \
  --lr_scheduler "$LR_SCHEDULER" \
  --warmup_ratio "$WARMUP_RATIO" \
  --max_samples "$TRAIN_SAMPLES" \
  --valid_max_samples "$VALID_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --max_leads "$MAX_LEADS" \
  --device "$DEVICE" \
  --log_dir "$RESULTS_DIR" \
  "${QLORA_FLAG[@]}"

run_stage "stage4b_curves" "$PYTHON" scripts/plot_training_curves.py \
  --log "$RESULTS_DIR/training_log.jsonl" \
  --output "$RESULTS_DIR/training_curves.png"

run_stage "stage5_evaluate" "$PYTHON" scripts/evaluate_ecgqa_small.py \
  --model_path "$CKPT_DIR" \
  --test "$OUTPUT_DIR/processed_test.jsonl" \
  --max_samples "$EVAL_SAMPLES" \
  --device "$DEVICE" \
  --output "$RESULTS_DIR/evaluation.jsonl"

run_stage "stage6_counterfactual" "$PYTHON" counterfactual/run_ecgqa_counterfactual_small.py \
  --model_path "$CKPT_DIR" \
  --data "$OUTPUT_DIR/processed_test.jsonl" \
  --max_samples "$CF_SAMPLES" \
  --device "$DEVICE" \
  --output "$RESULTS_DIR/counterfactual_results.jsonl"

run_stage "stage7_summary" "$PYTHON" scripts/summarize_ecgqa_small.py \
  --data_dir "$OUTPUT_DIR" \
  --outputs_dir "$RESULTS_DIR"

run_stage "stage6b_ablation" "$PYTHON" scripts/run_ecgqa_ablation_small.py \
  --model_path "$CKPT_DIR" \
  --test "$OUTPUT_DIR/processed_test.jsonl" \
  --max_samples "$ABL_SAMPLES" \
  --device "$DEVICE" \
  --output "$RESULTS_DIR/ablation.jsonl"

run_stage "stage7b_attributions" "$PYTHON" scripts/compute_attributions_small.py \
  --model_path "$CKPT_DIR" \
  --data "$OUTPUT_DIR/processed_test.jsonl" \
  --max_samples "$ATTR_SAMPLES" \
  --device "$DEVICE" \
  --output "$RESULTS_DIR/attributions.jsonl"

run_stage "stage7c_tracing" "$PYTHON" xai/representational_tracing.py \
  --model_path "$CKPT_DIR" \
  --data "$OUTPUT_DIR/processed_test.jsonl" \
  --max_samples "$TRACE_SAMPLES" \
  --metric "$TRACE_METRIC" \
  --ecg_segments "$TRACE_SEGMENTS" \
  --device "$DEVICE" \
  --output outputs/tracing/representational_tracing.jsonl \
  --summary_output outputs/tracing/stage_sensitivity_summary.json \
  --viz_output visualizer/tracing_data.js

run_stage "stage8_export_viz" "$PYTHON" scripts/export_visualizer_data.py \
  --results_dir "$RESULTS_DIR" \
  --processed "$OUTPUT_DIR/processed_test.jsonl" \
  --attributions "$RESULTS_DIR/attributions.jsonl" \
  --ablation "$RESULTS_DIR/ablation.jsonl" \
  --output visualizer/ecgqa_viz_data.js

echo ""
echo "Pipeline finished. Artefacts in $RESULTS_DIR"
echo "Visualizadores (abrir en el navegador; requieren internet para D3 por CDN):"
echo "  visualizer/visualizador_d3.html         (trazado interno · carga tracing_data.js)"
echo "  visualizer/dashboard_dominancia_d3.html (dominancia ECG-QA · carga ecgqa_viz_data.js)"
