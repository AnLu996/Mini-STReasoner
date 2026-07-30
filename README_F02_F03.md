# F02 · F03 · Instrucciones de ejecución (Ubuntu, GPU CUDA)

Estas fases producen los artefactos numéricos que el paper y el visualizador
necesitan para reportar `A_blind` honesto y la curva CFR(δ) sin recurrir a
valores sintéticos.

**Ejecución objetivo:** Ubuntu 22.04 con NVIDIA RTX 4050 Laptop (6 GB VRAM),
CUDA 12.x, PyTorch 2.x, `bitsandbytes` disponible. Los scripts detectan
automáticamente CUDA; si no hay GPU caen a CPU (mucho más lento pero funcional).

Nada de esto se ejecutó en Windows durante el desarrollo. El código está
diseñado para funcionar sin cambios en Ubuntu.

---

## 0 · Requisitos previos

```bash
cd Mini-STReasoner
pip install -r requirements.txt

# Datos ya procesados en:
#   data/ecgqa_small/processed_train.jsonl
#   data/ecgqa_small/processed_valid.jsonl
#   data/ecgqa_small/processed_test.jsonl
# (Si no existen, correr training/prepare_ecgqa.py primero.)

# Checkpoint E1 (Corrida A) esperado en:
#   checkpoints/ecgqa_small_lora/
```

---

## F02 · Modelo E0 (solo texto) para A_blind honesto

**Contribución al paper:** cumple O2 en su parte de `A_blind` y desbloquea el
cálculo real de `TDI = 1 − ΔS / (A_oracle − A_blind)` en la Sección VI (hoy
sólo se reporta `D_cf`, y `TDI` queda declarado como pendiente).

### F02.1 · Entrenamiento

```bash
python training/train_text_only_baseline.py \
  --train data/ecgqa_small/processed_train.jsonl \
  --valid data/ecgqa_small/processed_valid.jsonl \
  --output_dir checkpoints/e0_text_only \
  --epochs 7 \
  --patience 3 \
  --batch_size 1 \
  --grad_accum 8 \
  --learning_rate 2e-4 \
  --lr_scheduler cosine \
  --warmup_ratio 0.06 \
  --max_samples 300 \
  --valid_max_samples 50 \
  --early_stop_metric valid_loss \
  --device auto
```

**Salidas:**

- `checkpoints/e0_text_only/lora_adapter/` — adaptador LoRA entrenado
- `checkpoints/e0_text_only/tokenizer/` — tokenizador
- `checkpoints/e0_text_only/config.json` — con `model_kind: "e0_text_only"`
- `outputs/e0_text_only/training_log.jsonl` — log por paso y por época
- `outputs/e0_text_only/metrics_train.json` — resumen final

**Coste estimado:** 1–3 h en RTX 4050 con QLoRA activo (7 épocas, 300 muestras
train). No requiere señal ECG en RAM/VRAM: sólo prompts, así que es más rápido
por época que la Corrida A.

### F02.2 · Evaluación y export

```bash
python scripts/evaluate_text_only_baseline.py \
  --model_path checkpoints/e0_text_only \
  --test data/ecgqa_small/processed_test.jsonl \
  --max_samples 750 \
  --output outputs/e0_text_only/evaluation.jsonl \
  --summary outputs/e0_text_only/metrics_test.json \
  --blind_export outputs/e0_text_only/a_blind_by_family.json \
  --breakdown_qtype outputs/e0_text_only/breakdown_qtype.csv \
  --breakdown_attribute outputs/e0_text_only/breakdown_attribute.csv \
  --device auto
```

**Salidas:**

- `outputs/e0_text_only/evaluation.jsonl` — una línea por caso (pregunta,
  predicción, gold, EM/F1, sí/no).
- `outputs/e0_text_only/metrics_test.json` — agregados globales + desglose por
  `question_type` y `attribute_type`.
- `outputs/e0_text_only/a_blind_by_family.json` — **artefacto de consumo del
  visualizador**: `A_blind` por familia, listo para el campo `blind` del
  contrato `Condition` (véase `visualizer/data_contract.md`).
- CSVs de desglose para las tablas del paper.

**Cifras esperadas:**

| Métrica | Valor típico (E0 solo texto) | Interpretación |
|---|---|---|
| Exactitud sí/no | 0,50 – 0,60 | El texto por sí solo alcanza este techo |
| EM global | 0,10 – 0,25 | Menor que E1 (Corrida A) si el ECG aporta algo |

Si `A_blind ≈ A_full` (Corrida A), la ablación modal se confirma: la señal no
aporta contribución medible.

---

## F03 · Recalibración de contrafactuales por δ + curva CFR(δ)

**Contribución al paper:** implementa O4 completo — no sólo el protocolo
descrito en la Sección III-D, sino su ejecución numérica sobre la Corrida A.
Convierte el Caso 3 del paper (calibración) de un hallazgo cualitativo a una
tabla con cifras reales para la Vista V4 del visualizador.

### F03.1 · Calibración: mide δ por intervención

```bash
python counterfactual/delta_calibration.py \
  --model_path checkpoints/ecgqa_small_lora \
  --data data/ecgqa_small/processed_test.jsonl \
  --max_samples 100 \
  --factor 2.0 \
  --output outputs/audit/delta_calibration.json \
  --device auto
```

Aplica cada intervención (5 intensidades × 6 intervenciones sobre señal, +
4 intervenciones textuales sin barrido) sobre 100 casos y mide δ en tres
puntos: `h_bigru`, `h_pool` y `mean_text_embeddings`. Además reporta los pares
texto–señal cuyos δ medios difieren menos de un factor 2 (comparables).

**Salidas:**

- `outputs/audit/delta_calibration.json`:
  - `summary` — tabla `(modality, intervention, params) → δ mean/median/p05/p95`.
  - `pairs_by_delta` — lista de pares texto↔señal comparables por δ (útil para
    reportar QCFR y ECFR sólo entre intervenciones equiparadas).

**Coste estimado:** 10–20 min (sólo encoder + embeddings, no genera respuestas).

**Cifras esperadas para la Corrida A:**

| Intervención | δ_pool medio esperado |
|---|---|
| `ecg_cf_noise` (level=0.05) | 5e-6 a 5e-5 |
| `ecg_cf_time_mask` (fraction=0.25) | 1e-2 a 1e-1 |
| `question_cf` | 5e-3 a 5e-2 (δ_text) |

Es la evidencia numérica del Caso 3 del paper.

### F03.2 · Curva dosis-respuesta CFR(δ)

```bash
python counterfactual/generate_dose_response.py \
  --model_path checkpoints/ecgqa_small_lora \
  --data data/ecgqa_small/processed_test.jsonl \
  --max_samples 150 \
  --max_new_tokens 64 \
  --output outputs/audit/cfr_dose_response.json \
  --only_modality both \
  --device auto
```

Barre las mismas intensidades que F03.1 y para cada punto ejecuta la
generación sobre original e intervenido. La tasa de flip agregada es CFR;
combinada con el δ medido produce la curva CFR(δ).

**Salidas:**

- `outputs/audit/cfr_dose_response.json`:
  - `points` — un registro por (intervención, intensidad) con δ, CFR y flips.
  - `cfr_by_delta` — **artefacto de consumo del visualizador** en el formato
    exacto del contrato (`{modality, points:[{delta, cfr, ...}]}`).
  - `highlights_text_vs_signal_at_matched_delta` — para cada intervención
    textual, CFR de la señal al mismo δ (comparación honesta reportable en el
    paper).

**Coste estimado (RTX 4050, 150 casos):** 3–5 h. Es la parte más pesada.
Reduce `--max_samples` a 50 para una pasada rápida de sanity check antes de
lanzar la corrida larga.

---

## Integración con el visualizador

Una vez producidos los tres JSON de salida:

1. `outputs/e0_text_only/a_blind_by_family.json` → alimenta el campo `blind`
   del contrato `Condition` para cada familia. Actualiza `auditor_ecg.html`
   sustituyendo el `ACC[E1][fam][0]` (posición `blind`) por el valor real.

2. `outputs/audit/delta_calibration.json` → soporta la anotación del Caso 3
   en el paper con cifras concretas y actualiza el `DELTA[E1]` del
   visualizador.

3. `outputs/audit/cfr_dose_response.json` → pega directo en el campo
   `cfrByDelta` de la Condition activa del visualizador.

---

## Orden de ejecución sugerido

Total en Ubuntu con GPU: **~6-10 h de cómputo, sin supervisión activa**.

```bash
# 1) Entrenamiento E0 (1-3 h)
python training/train_text_only_baseline.py [args]

# 2) Evaluación E0 (30-60 min sobre 750 casos)
python scripts/evaluate_text_only_baseline.py [args]

# 3) Calibración δ (10-20 min)
python counterfactual/delta_calibration.py [args]

# 4) Curva CFR(δ) — arranca con --max_samples 50 primero (30 min sanity),
#    luego --max_samples 150 (3-5 h)
python counterfactual/generate_dose_response.py [args]
```

---

## Notas metodológicas

- El E0 usa **los mismos hiperparámetros** que la Corrida A. La única
  diferencia es la ausencia del codificador temporal y del proyector. Cualquier
  brecha `A_full − A_blind` es aporte estricto del ECG.
- `delta_calibration.py` usa `h_pool` como δ representativo de la señal
  porque es la representación que efectivamente entra al proyector; `h_bigru`
  se reporta como diagnóstico adicional. La distinción es la que reveló la
  firma H2 (pooling destructivo) tras la separación del hook en F0.
- `generate_dose_response.py` reutiliza las predicciones originales una única
  vez (cacheadas en memoria). El coste dominante es la generación sobre
  intervenido: 30 puntos × 150 casos = 4500 generaciones adicionales.
- Los tres scripts son idempotentes: sobreescriben sus salidas. Correrlos dos
  veces con los mismos argumentos produce los mismos resultados (seed fijo).
