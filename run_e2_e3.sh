#!/usr/bin/env bash
# F05.2 · Lanza E2 y E3 con los hiperparámetros que definen cada configuración
# (brief §6). Reutiliza train_ecgqa_lora_small.py; sólo cambian los args.
#
#   E1 (Corrida A ya ejecutada): BiGRU 128-D, 4 tokens, 1 capa GRU, proyector MLP.
#   E2 : idem con 32 tokens y 2 capas de GRU  -> ¿el cuello es el pooling?
#   E3 : E2 + proyector expresivo (Q-Former)  -> ¿el cuello es el proyector?
#
# Uso (Ubuntu, GPU CUDA):
#   bash run_e2_e3.sh          # corre E2 y E3
#   bash run_e2_e3.sh e2       # sólo E2
#   bash run_e2_e3.sh e3       # sólo E3

set -euo pipefail

TRAIN=${TRAIN:-data/qa_controlled/processed_train.jsonl}
VALID=${VALID:-data/qa_controlled/processed_valid.jsonl}
EPOCHS=${EPOCHS:-7}
PATIENCE=${PATIENCE:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_SAMPLES=${MAX_SAMPLES:-300}
VALID_MAX=${VALID_MAX:-50}
DEVICE=${DEVICE:-auto}

# Escala corregida (F0 del plan): consultas de atención inicializadas para
# igualar la norma de las claves. Sin esto el pooling degenera en promedio
# temporal — la firma H2 del paper.
QUERY_STD=1.0

# ------------------------------------------------------------------------- #
# E2 · 32 tokens, 2 capas GRU, proyector MLP.                                #
# ------------------------------------------------------------------------- #
train_e2() {
  echo "[e2] entrenando 32 tokens · 2 capas GRU · proyector MLP"
  python training/train_ecgqa_lora_small.py \
    --train "$TRAIN" \
    --valid "$VALID" \
    --output_dir checkpoints/e2_32tok \
    --log_dir outputs/e2_32tok \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --max_samples "$MAX_SAMPLES" \
    --valid_max_samples "$VALID_MAX" \
    --num_temporal_tokens 32 \
    --temporal_num_layers 2 \
    --temporal_hidden_dim 256 \
    --query_init_std "$QUERY_STD" \
    --match_embedding_scale \
    --projector_kind mlp \
    --device "$DEVICE"
}

# ------------------------------------------------------------------------- #
# E3 · 32 tokens, 2 capas GRU, proyector expresivo (Q-Former ligero).       #
# ------------------------------------------------------------------------- #
train_e3() {
  echo "[e3] entrenando 32 tokens · 2 capas GRU · proyector expresivo"
  python training/train_ecgqa_lora_small.py \
    --train "$TRAIN" \
    --valid "$VALID" \
    --output_dir checkpoints/e3_expressive \
    --log_dir outputs/e3_expressive \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --max_samples "$MAX_SAMPLES" \
    --valid_max_samples "$VALID_MAX" \
    --num_temporal_tokens 32 \
    --temporal_num_layers 2 \
    --temporal_hidden_dim 256 \
    --query_init_std "$QUERY_STD" \
    --match_embedding_scale \
    --projector_kind expressive \
    --device "$DEVICE"
}

case "${1:-both}" in
  e2)   train_e2 ;;
  e3)   train_e3 ;;
  both) train_e2; train_e3 ;;
  *) echo "Uso: $0 [e2|e3|both]"; exit 1 ;;
esac

echo "[e2/e3] terminado."
echo "Checkpoints:"
echo "  checkpoints/e2_32tok      (si se corrió E2)"
echo "  checkpoints/e3_expressive (si se corrió E3)"
echo ""
echo "Siguiente paso: evaluar y calibrar."
echo "  python scripts/evaluate_ecgqa_small.py --model_path checkpoints/e2_32tok --test data/qa_controlled/processed_test.jsonl"
echo "  python scripts/evaluate_ecgqa_small.py --model_path checkpoints/e3_expressive --test data/qa_controlled/processed_test.jsonl"
echo "  python counterfactual/delta_calibration.py    --model_path checkpoints/e2_32tok"
echo "  python counterfactual/delta_calibration.py    --model_path checkpoints/e3_expressive"
echo "  python counterfactual/generate_dose_response.py --model_path checkpoints/e2_32tok --max_samples 150"
echo "  python counterfactual/generate_dose_response.py --model_path checkpoints/e3_expressive --max_samples 150"
