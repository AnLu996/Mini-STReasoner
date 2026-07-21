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

## Validacion del escalamiento sobre ST-Bench

Antes de confiar en los resultados obtenidos sobre ECG-QA conviene comprobar que la receta reducida (Qwen3-0.6B + encoder GRU + QLoRA) produce un modelo funcional sobre la tarea *original* del paper. Este flujo entrena un subconjunto acotado de ST-Bench y lo mide con el protocolo de evaluacion de STReasoner.

```bash
# 1. descarga (73 MB: solo ST-SFT y ST-Test)
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Time-HD-Anonymous/ST-Bench', repo_type='dataset', \
  local_dir='data/stbench_small/raw', allow_patterns=['ST-SFT/*','ST-Test/*'])"

# 2. subconjuntos con muestreo por reservorio (semilla 42)
python training/prepare_stbench.py --local-dir data/stbench_small/raw/ST-SFT \
  --output-dir data/stbench_small/train --max-per-task 400 --seed 42
python training/prepare_stbench.py --local-dir data/stbench_small/raw/ST-Test \
  --output-dir data/stbench_small/test --max-per-task 60 --seed 42

# 3. entrenamiento (6 GB, ~50 min en una RTX 4050)
python training/train_sft_lora.py \
  --processed-dir data/stbench_small/train \
  --output-dir checkpoints/stbench_small_lora \
  --input-dim 10 --batch-size 1 --gradient-accumulation-steps 8 \
  --max-seq-length 512 --epochs 3

# 4. evaluacion, linea base y ablacion
bash run_stbench_validation.sh
```

`scripts/score_stbench.py` reemplaza a `inference/evaluate_tasks.py` para este dataset: puntua con el protocolo del paper (extraccion de `<answer>`, letra A-D para opcion multiple, MAE/MAPE para forecasting) en lugar de comparar cadenas normalizadas. `scripts/verify_scorer_against_paper.py` re-puntua las generaciones publicadas de STReasoner-8B y reproduce sus cuatro metricas exactamente, de modo que cualquier diferencia posterior sea atribuible al modelo y no al protocolo.

Como el subconjunto de test es pequeno, `scripts/stbench_confidence.py` acompana cada accuracy con su intervalo de Wilson y cada contribucion modal con un intervalo bootstrap pareado.

### Intervenciones sobre el contenido de la serie

La ablacion por eliminacion (`no_series`) no es interpretable en este checkpoint: al retirar los tokens temporales el modelo pierde la senal que lo mantenia en el formato aprendido, revierte al comportamiento del Qwen3 base y agota el presupuesto de tokens sin emitir `<answer>`. Su accuracy mide un colapso de formato, no la contribucion de la modalidad.

`scripts/stbench_series_intervention.py` evita ese problema sustituyendo el *contenido* de la serie y dejando los cuatro tokens temporales en su sitio:

```bash
python scripts/stbench_series_intervention.py \
  --model-path checkpoints/stbench_small_lora \
  --task reasoning_correlation --limit 30
```

```text
original   la serie propia de la muestra
swapped    la serie de otra muestra, misma pregunta
zeroed     una serie de ceros, misma pregunta
```

Si la respuesta no cambia entre las tres condiciones, no depende de la evidencia temporal. Los resultados de la corrida estan en [`RESULTADOS_STBENCH_VALIDACION.txt`](RESULTADOS_STBENCH_VALIDACION.txt).

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
# 6b. ablacion modal (full / no_text / no_series / conflict_text)
python scripts/run_ecgqa_ablation_small.py --model_path checkpoints/ecgqa_small_lora --test data/ecgqa_small/processed_test.jsonl --max_samples 100 --output outputs/ecgqa_small/ablation.jsonl
```

La ablacion evalua el test quitando una modalidad cada vez (`no_text` = solo ECG, `no_series` = solo texto) y con `conflict_text` (nota enganosa contra el ECG). Reporta EM/Token F1/accuracy yes-no por configuracion y la **dominancia modal** del paper:

```text
text_contribution = full - no_text
ecg_contribution  = full - no_series
textual_dominance = text_contribution - ecg_contribution   (positivo = depende mas del texto)
```

global y por `question_type` (`ablation_summary.json`, `ablation_by_config.csv`, `ablation_by_question_type.csv`). Es la Etapa 6b del pipeline maestro y alimenta los paneles V1/V2 del visualizador (selector de Configuracion con `completo / sin ECG / sin texto / texto en conflicto`).

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
outputs/ecgqa_small/ablation.jsonl
outputs/ecgqa_small/ablation_summary.json
outputs/ecgqa_small/ablation_by_config.csv
outputs/ecgqa_small/ablation_by_question_type.csv
outputs/ecgqa_small/run_summary.json
```

### Visualizador D3

El dashboard `visualizer/dashboard_dominancia_d3.html` (5 paneles: flujo multimodal, rendimiento + matriz de confusion, embeddings, relevancia texto/ECG y QA con intervenciones contrafactuales) puede mostrar **los resultados reales** del run. Para conectarlo:

```bash
python scripts/export_visualizer_data.py \
  --results_dir outputs/ecgqa_small \
  --processed data/ecgqa_small/processed_test.jsonl \
  --output visualizer/ecgqa_viz_data.js
```

Esto genera `visualizer/ecgqa_viz_data.js` (un `window.ECGQA_DATA = {...}`). El pipeline maestro ya lo ejecuta como Etapa 8. Despues solo abre `visualizer/dashboard_dominancia_d3.html` en el navegador (doble clic; requiere internet para cargar D3 por CDN). Si el archivo de datos no existe, el visualizador usa sus datos sinteticos de demostracion.

Para que los paneles de embeddings (V3) y relevancia de tokens/ECG (V4) usen **atribuciones reales del modelo** (no proxies), ejecuta antes del export:

```bash
python scripts/compute_attributions_small.py \
  --model_path checkpoints/ecgqa_small_lora \
  --data data/ecgqa_small/processed_test.jsonl \
  --max_samples 30 \
  --output outputs/ecgqa_small/attributions.jsonl
```

Esto calcula, por muestra: saliencia por gradiente sobre la señal ECG (parches temporales), saliencia por gradiente sobre los embeddings de los tokens de la pregunta, y los embeddings reales del espacio del LLM (tokens de texto + tokens temporales proyectados) en tres condiciones (completo / ECG perturbado / sin proyector), reducidos a 16-dim por PCA. El pipeline maestro lo ejecuta como Etapa 7b y el exportador los fusiona automaticamente.

Que muestra con datos reales:
- prediccion del modelo vs respuesta correcta por muestra (panel QA);
- tabla de intervenciones reales (`question_cf`, `neutral`, perturbaciones ECG, conflictos) con marca de *flip*;
- dominancia por muestra `D = QCFR / (QCFR + ECFR)` y veredicto;
- accuracy por tipo de pregunta y matriz de confusion (clic en celda = filtrar);
- la onda ECG real (derivacion configurable con `--lead`) con saliencia por segmento (proxy de energia).

Con `attributions.jsonl` presente, los paneles de embeddings (V3) y relevancia de tokens/ECG (V4) tambien son reales (saliencia por gradiente y embeddings del espacio del LLM). Sin ese archivo, V3/V4 caen a proyecciones/atribuciones aproximadas, pero el resto (rendimiento, dominancia, predicciones e intervenciones) sigue siendo real.

## Trazado representacional interno

`xai/representational_tracing.py` pasa de la auditoria contrafactual *post-hoc* a un **trazado representacional interno** (tambien llamado *trazado contrafactual de sensibilidad modal*). Para cada caso corre cinco condiciones:

```text
Original : ECG + pregunta
Text-CF  : ECG + pregunta reescrita (meaning-preserving)
ECG-CF   : ECG perturbado + pregunta original
Neutral  : ECG + pregunta neutral
Conflict : ECG/pregunta contradictorios
```

y captura resumenes por etapa del flujo multimodal mediante **forward hooks** del modelo (encoder temporal y proyector) mas `output_hidden_states` del LLM:

```text
encoder_output  ·  projector_output  ·  inputs_embeds (fusion)  ·  hidden_states del LLM  ·  logits / respuesta
```

Por etapa calcula distancias representacionales simples y estables (coseno o L2 normalizada) comparando *original vs Text-CF* y *original vs ECG-CF*:

```text
impacto_texto       = dist(original, Text-CF)   (None en encoder/proyector: el texto no atraviesa esas etapas)
impacto_ECG         = dist(original, ECG-CF)
diferencia_texto_ECG = impacto_texto - impacto_ECG
```

No hay backpropagation. La clasificacion por caso es `TEXT_DOMINANT / ECG_DOMINANT / BALANCED / INSENSITIVE / UNCLEAR` (la clase `INSENSITIVE` marca casos que no se mueven ni con texto ni con ECG). El diagnostico usa la formulacion correcta: **el sesgo se manifiesta o se amplifica** en una etapa, nunca que "nace" en ella.

```bash
python xai/representational_tracing.py \
  --model_path checkpoints/ecgqa_small_lora \
  --data data/ecgqa_small/processed_test.jsonl \
  --max_samples 30 \
  --metric cosine \
  --ecg_segments 6
```

Salidas:

```text
outputs/tracing/representational_tracing.jsonl   (un caso por linea, con stages/latent/interventions)
outputs/tracing/stage_sensitivity_summary.json   (agregado por etapa + etapa de amplificacion + class_counts)
visualizer/tracing_data.js                        (window.TRACING_DATA para el visualizador)
```

Si las activaciones internas no estan disponibles (`--output_only` o un fallo al capturar hooks), el modulo **no inventa valores**: cae a modo `contrafactual_global` y solo reporta la sensibilidad de salida (flips de la respuesta); el visualizador lo marca como "contrafactual global". Es la Etapa 7c del pipeline maestro.

### Visualizador (trazado interno)

`visualizer/visualizador_d3.html` carga `visualizer/tracing_data.js` y, si no existe, usa datos sinteticos de demostracion claramente etiquetados. Vistas:

```text
V1  Trazado representacional interno   ECG -> Encoder -> Proyector -> Fusion -> LLM -> Respuesta,
                                       con impacto ECG / impacto texto / diferencia por bloque y color por dominancia.
V2  Sensibilidad modal por etapa       tabla/heatmap (impacto texto | impacto ECG | diferencia | diagnostico).
V3  Espacio latente texto-ECG          proyeccion 2D con flechas original->contrafactual (ECG y pregunta).
V4  Comparación contrafactual local    segun el escenario activo: ECG original vs ECG intervenido (resaltando
                                       el segmento alterado) o pregunta original vs pregunta intervenida
                                       (resaltando palabras modificadas, clicables). Muestra si la respuesta
                                       cambio. No es atribucion causal.
V5  Pregunta-respuesta contrafactual   original/esperada/pregunta-modif/ECG-modif/neutral/conflicto + clase de dominancia.
V6  Atencion del encoder temporal      peso que el encoder pone en cada ventana de la senal, contra la
                                       referencia uniforme. Responde "que parte de la serie mira el modelo".
```

### V6 · atencion del encoder temporal

El encoder devuelve una matriz `[tokens, T]` con el peso que cada consulta aprendida pone en cada paso de la serie. Nunca se guardaba, y es la unica evidencia directa de que tramo lee el modelo. `xai/attention_export.py` la persiste y la agrupa en ventanas:

```bash
python xai/attention_export.py \
  --model-path checkpoints/ecgqa_5k_control \
  --data data/ecgqa_5k/processed_test.jsonl --data-format ecgqa \
  --output outputs/ecgqa_5k_control/encoder_attention.jsonl --bins 50
```

El panel compara el perfil contra la referencia uniforme (`1/bins`) y emite un veredicto explicito: si el pico no llega a 1,5 veces el uniforme y la entropia media supera 0,99, el pooling equivale a promediar toda la senal y el encoder no esta seleccionando nada. Sobre las corridas de ECG-QA el veredicto separa las dos sin ambiguedad: 100 de 100 casos planos en Corrida A, 100 de 100 selectivos en Corrida B.

Para adjuntar la atencion a un trazado ya calculado, sin repetir la inferencia:

```bash
python scripts/rebuild_tracing_viz.py \
  --tracing outputs/tracing/representational_tracing_occlusion.jsonl \
  --encoder-attention outputs/ecgqa_5k_control/encoder_attention.jsonl \
  --viz-output visualizer/tracing_data.js
```

`xai/representational_tracing.py` tambien acepta `--encoder_attention` para hacer la fusion durante la corrida.

Interactividad (todo sobre datos precomputados, sin recalcular el modelo):

- **Selector de escenario** (Original / Pregunta modificada / Pregunta neutral / ECG modificado / Texto contradictorio / Sin ECG): coordina V1–V5 por `case_id` + `scenario`.
- **V1**: hover en cada bloque muestra impacto_texto, impacto_ECG, diferencia y diagnóstico; clic abre un panel de detalle de etapa (Original vs Text-CF, Original vs ECG-CF, distancia usada, diagnóstico). Botón **Reproducir flujo** anima el recorrido etapa por etapa.
- **V4**: en escenarios de texto, las palabras modificadas son clicables y muestran la comparación de frase (sin inventar impacto token-level).
- **Panel lateral**: filtros por `case_dominance` (5 clases), comportamiento (solo cambia con texto / solo con ECG / no cambia / conflicto texto-ECG), `text_dominance`, `question_type`, `attribute_type`, y una **tabla compacta `Caso | D_text`** ordenada de mayor a menor (clic = cargar caso).

Si falta un valor por etapa (modo `contrafactual_global`) o una predicción de escenario, el visualizador muestra **"no disponible"**; nunca inventa valores.

## Limitaciones

Esta minirreplica no busca reproducir las cifras del paper: no usa Qwen3-8B, S-GRPO, ocho A100 ni el entrenamiento completo en tres etapas. La atencion del encoder y la saliencia por gradiente son explicaciones del modulo temporal, no pruebas causales por si solas. El trazado representacional interno mide *sensibilidad* (cuanto se mueve la representacion ante una intervencion), no causalidad; las clases de dominancia describen el comportamiento observado, no su origen. El objetivo es disponer de una base pequena y auditable para estudiar dominancia modal textual.
