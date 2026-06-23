# Mini-STReasoner

Minirreplica multimodal de STReasoner para una GPU NVIDIA de 6 GB. Combina Qwen3-0.6B, un encoder GRU bidireccional con atencion temporal, un proyector al espacio latente del LLM e integracion real mediante `inputs_embeds`.

## Alcance

Componentes fieles al trabajo original:

- encoder dedicado para series temporales;
- alineacion de tokens temporales con el espacio oculto del LLM;
- concatenacion latente de tokens temporales y texto;
- SFT multimodal y evaluacion de las cuatro tareas principales de ST-Bench.

Simplificaciones:

- `Qwen/Qwen3-0.6B` en lugar de Qwen3-8B;
- LoRA/QLoRA en lugar de ajustar todos los parametros;
- una sola etapa SFT, sin S-GRPO ni entrenamiento distribuido;
- cuatro tokens temporales producidos por una GRU pequena, no la configuracion de produccion del paper.

Qwen3-0.6B permite mantener el LLM, los adaptadores y los modulos temporales dentro de una RTX 4050 Laptop de 6 GB usando cuantizacion NF4.

## Entorno

```bash
conda create -n mini-str python=3.10 -y
conda activate mini-str
pip install -r requirements.txt
```

El entorno `str` del repositorio original tambien puede utilizarse si contiene estas dependencias.

## Preparar ST-Bench

Desde Hugging Face, en streaming:

```bash
cd Mini-STReasoner
python training/prepare_stbench.py
```

Desde una descarga local completa:

```bash
python training/prepare_stbench.py \
  --local-dir ../data/ST-Bench \
  --output-dir data/processed
```

El script detecta nombres de columnas, conserva metadatos y escribe un JSONL por tarea sin acumular el dataset en memoria. El manifiesto queda en `data/processed/manifest.json`.

Revise el numero maximo de variables del dataset y pase ese valor como `--input-dim`; las series con menos variables se rellenan con ceros.

## Entrenamiento

Configuracion de 6 GB recomendada:

```bash
python training/train_sft_lora.py \
  --input-dim 10 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 512 \
  --epochs 1
```

QLoRA NF4 esta activo por defecto. Use `--no-qlora` solo si hay memoria suficiente. Para una prueba corta agregue `--max-steps 2`. El checkpoint contiene `lora_adapter/`, `ts_encoder.pt`, `temporal_projector.pt`, `tokenizer/` y `config.json`.

## Inferencia y evaluacion

```bash
python inference/run_inference.py \
  --model-path checkpoints/mini_streasoner_qwen06 \
  --task reasoning_forecasting

python inference/evaluate_tasks.py
```

La evaluacion guarda `outputs/evaluation_results.json` y calcula exact match, accuracy cerrada, F1 por tokens y coincidencia textual.

## XAI

```bash
python xai/modal_ablation.py --model-path checkpoints/mini_streasoner_qwen06 --task reasoning_forecasting
python xai/dominance_metrics.py
python xai/attention_export.py --model-path checkpoints/mini_streasoner_qwen06 --task reasoning_forecasting
python xai/temporal_saliency.py --model-path checkpoints/mini_streasoner_qwen06 --task reasoning_forecasting
```

La ablacion compara `full`, `no_text`, `no_series` y `conflict_text`. La dominancia se define como:

```text
(full - no_text) - (full - no_series)
```

Un valor positivo indica mayor dependencia del texto; uno negativo, mayor dependencia de la serie. `conflict_text` es una perturbacion base y debe especializarse segun la semantica de cada tarea para estudios causales rigurosos.

## Prueba controlada ECG-QA (subconjunto pequeno)

Pipeline reproducible para obtener resultados preliminares con senales ECG reales de PTB-XL sin descargar todo ECG-QA ni MIMIC-IV-ECG. Trabaja por etapas, usa `seed=42` y guarda logs y resultados intermedios en `outputs/ecgqa_small/`.

Ejecucion completa (descarga, prepara, infiere baseline, entrena, evalua y mide contrafactuales):

```bash
bash run_ecgqa_small_pipeline.sh
```

El script es configurable por variables de entorno (`DEVICE=cpu` para no usar GPU, `MAX_QUESTIONS`, `MAX_UNIQUE_ECGS`, `SUBSET`, etc.; ver cabecera del script) y cada etapa se puede correr por separado:

```bash
# 1. descarga/muestreo controlado (solo los ECG necesarios)
python scripts/download_ecgqa_small.py --subset all --max_questions 300 --max_unique_ecgs 100 --seed 42 --output data/ecgqa_small
# 2. carga de senales reales -> .npy [1000, 12] (z-score por derivacion)
python scripts/prepare_ecg_signals.py --manifest data/ecgqa_small/manifest.jsonl --output data/ecgqa_small/processed.jsonl --target_length 1000 --max_leads 12
# 3. inferencia baseline sin entrenamiento
python scripts/run_ecgqa_inference_small.py --data data/ecgqa_small/processed.jsonl --max_samples 20 --output outputs/ecgqa_small/inference_raw.jsonl
# 4. entrenamiento pequeno (encoder + proyector + LoRA)
python training/train_ecgqa_lora_small.py --train data/ecgqa_small/processed_train.jsonl --valid data/ecgqa_small/processed_valid.jsonl --output_dir checkpoints/ecgqa_small_lora --epochs 1 --max_samples 300 --batch_size 1 --grad_accum 8 --max_seq_len 512
# 5. evaluacion (EM, Token F1, accuracy yes/no, por question_type y attribute_type)
python scripts/evaluate_ecgqa_small.py --model_path checkpoints/ecgqa_small_lora --test data/ecgqa_small/processed_test.jsonl --max_samples 100 --output outputs/ecgqa_small/evaluation.jsonl
# 6. contrafactuales (QCFR, ECFR, dominancia textual, conflictos)
python counterfactual/run_ecgqa_counterfactual_small.py --model_path checkpoints/ecgqa_small_lora --data data/ecgqa_small/processed_test.jsonl --max_samples 50 --output outputs/ecgqa_small/counterfactual_results.jsonl
```

`download_ecgqa_small.py` clona el repo ECG-QA (`v1.0.2`) y descarga unicamente los registros WFDB de PTB-XL referenciados por el subconjunto elegido. La senal entra al encoder como serie temporal `[tiempo, derivaciones]`, nunca como texto. Requiere `wfdb` (ya incluido en `requirements.txt`).

> Nota: el muestreo `--subset all` reparte `max_questions` entre train/valid/test (70/15/15) para que la Etapa 2 genere `processed_{train,valid,test}.jsonl`, que consumen las Etapas 4-6. Use `--subset train` para una sola particion.

Artefactos finales para redactar la seccion experimental:

```text
outputs/ecgqa_small/inference_raw.jsonl
outputs/ecgqa_small/training_log.jsonl
outputs/ecgqa_small/metrics_train.json
outputs/ecgqa_small/evaluation_summary.json
outputs/ecgqa_small/evaluation_by_question_type.csv
outputs/ecgqa_small/evaluation_by_attribute_type.csv
outputs/ecgqa_small/counterfactual_summary.json
outputs/ecgqa_small/counterfactual_by_question_type.csv
outputs/ecgqa_small/selected_cases.jsonl
outputs/ecgqa_small/run_summary.json
```

## Limitaciones

Esta minirreplica no busca reproducir las cifras del paper: no usa Qwen3-8B, S-GRPO, ocho A100 ni el entrenamiento completo en tres etapas. La atencion del encoder y la saliencia por gradiente son explicaciones del modulo temporal, no pruebas causales por si solas. El objetivo es disponer de una base pequena y auditable para estudiar dominancia modal textual.
