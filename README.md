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

## Limitaciones

Esta minirreplica no busca reproducir las cifras del paper: no usa Qwen3-8B, S-GRPO, ocho A100 ni el entrenamiento completo en tres etapas. La atencion del encoder y la saliencia por gradiente son explicaciones del modulo temporal, no pruebas causales por si solas. El objetivo es disponer de una base pequena y auditable para estudiar dominancia modal textual.
