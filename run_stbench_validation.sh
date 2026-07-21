#!/usr/bin/env bash
# Validate the scaled-down recipe (Qwen3-0.6B + GRU encoder + LoRA) on the
# original ST-Bench tasks, using the evaluation protocol of the STReasoner paper.
#
# This answers a question the ECG-QA runs never tested: whether the scaling
# itself is sound, or whether the multimodal path was already broken before the
# dataset was swapped. It is a sanity check, not an attempt to match the
# published numbers — those come from Qwen3-8B trained on 8xA100 in three stages.
#
# Prerequisites (both already produce their outputs under data/stbench_small/):
#   python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('Time-HD-Anonymous/ST-Bench', repo_type='dataset', \
#       local_dir='data/stbench_small/raw', allow_patterns=['ST-SFT/*','ST-Test/*'])"
#   python training/prepare_stbench.py --local-dir data/stbench_small/raw/ST-SFT \
#       --output-dir data/stbench_small/train --max-per-task 400 --seed 42
#   python training/prepare_stbench.py --local-dir data/stbench_small/raw/ST-Test \
#       --output-dir data/stbench_small/test --max-per-task 60 --seed 42
#
# Usage: bash run_stbench_validation.sh

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/stbench_small_lora}"
TEST_DIR="${TEST_DIR:-data/stbench_small/test}"
RESULTS_DIR="${RESULTS_DIR:-outputs/stbench_small}"
ABLATION_LIMIT="${ABLATION_LIMIT:-30}"
BASELINE_LIMIT="${BASELINE_LIMIT:-30}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
TASKS="${TASKS:-reasoning_forecasting reasoning_entity reasoning_etiological reasoning_correlation}"

mkdir -p "$RESULTS_DIR/logs"

echo "== Etapa 1: verificar el evaluador contra las cifras publicadas =="
python scripts/verify_scorer_against_paper.py 2>&1 | tee "$RESULTS_DIR/logs/verify_scorer.log"

echo
echo "== Etapa 2: inferencia sobre el conjunto de test =="
for task in $TASKS; do
    echo "-- $task --"
    python inference/run_inference.py \
        --model-path "$CHECKPOINT" \
        --task "$task" \
        --data-dir "$TEST_DIR" \
        --output "$RESULTS_DIR/predictions_$task.jsonl" \
        > "$RESULTS_DIR/logs/inference_$task.log" 2>&1
done

echo
echo "== Etapa 2b: linea base sin entrenar ($BASELINE_LIMIT muestras por tarea) =="
for task in $TASKS; do
    echo "-- $task --"
    python inference/run_inference.py \
        --base-model "$BASE_MODEL" \
        --input-dim 10 \
        --task "$task" \
        --data-dir "$TEST_DIR" \
        --output "$RESULTS_DIR/baseline_predictions_$task.jsonl" \
        --limit "$BASELINE_LIMIT" \
        > "$RESULTS_DIR/logs/baseline_$task.log" 2>&1
done
python scripts/score_stbench.py --predictions-dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/stbench_scores_baseline.json" \
    --prefix baseline_predictions 2>&1 | tee "$RESULTS_DIR/logs/score_baseline.log"

echo
echo "== Etapa 3: puntuacion (protocolo del paper) =="
python scripts/score_stbench.py --predictions-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/logs/score.log"

echo
echo "== Etapa 4: ablacion modal ($ABLATION_LIMIT muestras por tarea) =="
for task in $TASKS; do
    echo "-- $task --"
    python xai/modal_ablation.py \
        --model-path "$CHECKPOINT" \
        --task "$task" \
        --data-dir "$TEST_DIR" \
        --output "$RESULTS_DIR/ablation_$task.jsonl" \
        --limit "$ABLATION_LIMIT" \
        > "$RESULTS_DIR/logs/ablation_$task.log" 2>&1
done

echo
echo "== Etapa 5: puntuacion de la ablacion =="
python scripts/score_stbench_ablation.py --ablation-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/logs/score_ablation.log"

echo
echo "Listo. Artefactos en $RESULTS_DIR/"
