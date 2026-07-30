# Plan de ejecución completo · F02 → F07 (Ubuntu, GPU CUDA)

Todos los scripts de F02–F07 están escritos y comiten al repositorio; ninguno
se ha ejecutado en Windows durante el desarrollo. Este documento es la
secuencia de comandos exacta para ejecutar todo en Ubuntu con GPU y llenar el
paper y el visualizador con datos reales.

**Coste total estimado en RTX 4050:** 10–15 h de cómputo distribuidas, sin
supervisión activa. Puede correr por bloques independientes.

---

## 0 · Dependencias

```bash
cd Mini-STReasoner
pip install -r requirements.txt
pip install pandas wfdb tqdm scikit-learn   # F04 y probes
```

Datos externos (una vez):

| Recurso | Ruta esperada | Cómo obtenerlo |
|---|---|---|
| PTB-XL v1.0.3 | `data/ptbxl/` | `wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/` |
| PTB-XL+ v1.0.1 | `data/ptbxlplus/` | `wget -r -N -c -np https://physionet.org/files/ptb-xl-plus/1.0.1/` |
| ECG-QA processed | `data/ecgqa_small/processed_*.jsonl` | `python training/prepare_ecgqa.py` (ya existe) |
| Checkpoint E1 | `checkpoints/ecgqa_small_lora/` | Corrida A ya ejecutada |

---

## 1 · F02 · Baseline E0 (solo texto) → `A_blind` honesto

```bash
# 1.1  Entrena Qwen3-0.6B + LoRA sin señal (~1–3 h)
python training/train_text_only_baseline.py \
  --train data/ecgqa_small/processed_train.jsonl \
  --valid data/ecgqa_small/processed_valid.jsonl \
  --output_dir checkpoints/e0_text_only \
  --epochs 7 --patience 3 --batch_size 1 --grad_accum 8

# 1.2  Evalúa sobre test y exporta A_blind por familia (~30–60 min)
python scripts/evaluate_text_only_baseline.py \
  --model_path checkpoints/e0_text_only \
  --test data/ecgqa_small/processed_test.jsonl \
  --max_samples 750
```

**Salidas críticas:**
- `outputs/e0_text_only/a_blind_by_family.json` → consumo del visualizador.
- `outputs/e0_text_only/metrics_test.json` → texto de la Sección VI.

---

## 2 · F03 · Calibración por δ + curva CFR(δ)

```bash
# 2.1  Mide δ para 30 puntos de intervención (~10–20 min)
python counterfactual/delta_calibration.py \
  --model_path checkpoints/ecgqa_small_lora \
  --data data/ecgqa_small/processed_test.jsonl \
  --max_samples 100 --factor 2.0

# 2.2  Sanity check rápido (~30 min)
python counterfactual/generate_dose_response.py \
  --model_path checkpoints/ecgqa_small_lora \
  --max_samples 50

# 2.3  Corrida completa CFR(δ) (~3–5 h)
python counterfactual/generate_dose_response.py \
  --model_path checkpoints/ecgqa_small_lora \
  --max_samples 150
```

**Salidas críticas:**
- `outputs/audit/delta_calibration.json` → soporte del Caso 3 del paper.
- `outputs/audit/cfr_dose_response.json` → curva CFR(δ) para el visualizador.

---

## 3 · F04 · Conjunto controlado F1–F5 sobre PTB-XL+ y `A_oracle`

```bash
# 3.1  Construye el JSONL controlado con filtro de concordancia (~30 min–2 h)
python data/build_qa.py \
  --ptbxl_root      data/ptbxl \
  --ptbxlplus_root  data/ptbxlplus \
  --output_dir      data/qa_controlled \
  --tol_ms 10 --tol_mv 0.05 --seed 42

# 3.2  Entrena una CNN 1D por familia (~5–15 min por familia, <1 h total)
python training/train_oracle_probe.py \
  --qa_train data/qa_controlled/processed_train.jsonl \
  --qa_valid data/qa_controlled/processed_valid.jsonl \
  --epochs 30 --patience 6 --batch_size 32

# 3.3  Evalúa A_oracle sobre test (~5 min)
python scripts/evaluate_oracle.py \
  --qa_test data/qa_controlled/processed_test.jsonl \
  --a_blind outputs/e0_text_only/a_blind_by_family.json
```

**Salidas críticas:**
- `data/qa_controlled/processed_{train,valid,test}.jsonl` → conjunto para E2/E3.
- `outputs/oracle/a_oracle_by_family.json` → consumo del visualizador.

---

## 4 · F05 · Configuraciones E2 (32 tokens) y E3 (proyector expresivo)

```bash
# 4.1  Lanza E2 y E3 con los presets del brief (~3–6 h en total)
bash run_e2_e3.sh both

# 4.2  Evalúa las tres configuraciones sobre el conjunto controlado
for cfg in ecgqa_small_lora e2_32tok e3_expressive; do
  python scripts/evaluate_ecgqa_small.py \
    --model_path checkpoints/$cfg \
    --test data/qa_controlled/processed_test.jsonl \
    --output outputs/audit/eval_${cfg}.jsonl
done

# 4.3  Repite F03.2 para E2 y E3 (opcional pero recomendado)
for cfg in e2_32tok e3_expressive; do
  python counterfactual/generate_dose_response.py \
    --model_path checkpoints/$cfg --max_samples 150 \
    --output outputs/audit/cfr_dose_response_${cfg}.json
done
```

**Salidas críticas:**
- `checkpoints/e2_32tok/`, `checkpoints/e3_expressive/`
- `outputs/audit/eval_*.jsonl` — exactitud por config y familia

---

## 5 · F06 · Contratar el visualizador a los datos reales

```bash
python scripts/wire_visualizer_data.py \
  --a_blind     outputs/e0_text_only/a_blind_by_family.json \
  --a_oracle    outputs/oracle/a_oracle_by_family.json \
  --dose        outputs/audit/cfr_dose_response.json \
  --delta_calib outputs/audit/delta_calibration.json \
  --output      visualizer/audit_data.js
```

Después: editar `visualizer/auditor_ecg.html` para reemplazar el bloque de
constantes hardcodeadas (`const ACC = {...}`, `LEDGER = {...}`, `DELTA = {...}`)
por lecturas de `window.AUDIT`:

```html
<script src="audit_data.js"></script>
<script>
  const ACC    = window.AUDIT.ACC    || /* fallback sintético */ {...};
  const LEDGER = window.AUDIT.LEDGER || {...};
  const DELTA  = window.AUDIT.DELTA  || {...};
</script>
```

---

## 6 · F07 · Regenerar los casos de estudio del paper

```bash
python scripts/paper_cases_from_outputs.py \
  --e1_metrics  outputs/ecgqa_small/metrics_test.json \
  --e0_metrics  outputs/e0_text_only/metrics_test.json \
  --dose        outputs/audit/cfr_dose_response.json \
  --delta_calib outputs/audit/delta_calibration.json \
  --output      outputs/paper/casos_estudio.tex
```

Copiar el contenido de `outputs/paper/casos_estudio.tex` sobre la Sección VI de
`a_paper_v3.tex`, reemplazando los tres subsecciones `\subsection{Caso ...}`.

---

## 7 · Orden secuencial recomendado

Las dependencias entre fases son:

```
F02 (E0)  ─────────┐
                   ├───► F04.3 (oracle eval)  ──► F06 (wire)  ──► F07 (paper)
F04.1 (qa)         │
   └──► F04.2 (oracle probe) ─┘
F03.1 (delta) ──► F03.2 (dose) ─┐
                                 └──► F06 (wire)  ──► F07 (paper)
F05 (E2, E3) ──► F03/F04 sobre E2, E3 ──► F06 con múltiples configs
```

Se pueden paralelizar F02 y F04.1 en dos consolas distintas si hay VRAM
suficiente (F04.1 no usa GPU salvo para wfdb).

---

## 8 · Cifras que deberían salir (para sanity check)

Basado en la Corrida A ya conocida y las expectativas del brief:

| Métrica | Esperado | Interpretación |
|---|---|---|
| E0 `A_blind` global | 0.50 – 0.60 (yes/no) | Techo del texto solo |
| E1 `A_full` global   | ≈ E0 `A_blind` + 0.01 | Aporte marginal del ECG con Corrida A |
| Oracle F1 (QRS)      | 0.85 – 0.95 | La señal decide bien esta pregunta |
| Oracle F4 (HR)       | 0.90 – 0.98 | Frecuencia cardiaca extraíble de 10 s |
| δ ruido gaussiano    | 1e-6 – 1e-4 (h_pool) | Firma del pooling destructivo |
| δ oclusión temporal  | 1e-2 – 1e-1 (h_pool) | 3–4 órdenes por encima del ruido |
| CFR(δ=1e-2) señal    | 0.10 – 0.30 | Comparar con CFR texto al mismo δ |

Si algún valor se aparta drásticamente de estas expectativas, revisar
`build_qa.py` (mapeo de columnas de PTB-XL+) o el checkpoint cargado.

---

## 9 · Estructura final de outputs

```
outputs/
├── e0_text_only/
│   ├── evaluation.jsonl
│   ├── metrics_test.json
│   ├── metrics_train.json
│   ├── a_blind_by_family.json      ← F06
│   ├── breakdown_qtype.csv
│   └── breakdown_attribute.csv
├── ecgqa_small/                    (Corrida A, ya existe)
│   └── ...
├── oracle/
│   ├── metrics_train.json
│   ├── metrics_test.json
│   └── a_oracle_by_family.json     ← F06
├── audit/
│   ├── delta_calibration.json      ← F06 · Caso 3 paper
│   ├── cfr_dose_response.json      ← F06 · Sección VI paper
│   ├── ledger_e1.json              (pendiente · F1 del plan)
│   ├── eval_ecgqa_small_lora.jsonl
│   ├── eval_e2_32tok.jsonl
│   └── eval_e3_expressive.jsonl
└── paper/
    └── casos_estudio.tex           ← copiar a a_paper_v3.tex

visualizer/
└── audit_data.js                   ← genera scripts/wire_visualizer_data.py

checkpoints/
├── ecgqa_small_lora/               (E1)
├── e0_text_only/                   (E0, F02)
├── e2_32tok/                       (E2, F05)
├── e3_expressive/                  (E3, F05)
└── oracle/
    ├── oracle_F1.pt … oracle_F5.pt (F04.2)
```

---

## 10 · Notas de troubleshooting

- **OOM en RTX 4050 con QLoRA**: reducir `--batch_size` a 1 (ya está) o
  `--max_seq_len` a 384. Si sigue, considerar `--no_qlora` con `--device cpu`
  (mucho más lento) para al menos completar F02.
- **`bitsandbytes` no encuentra CUDA**: instalar `bitsandbytes>=0.42.0` sobre
  PyTorch 2.x. En Ubuntu 22.04 con CUDA 12.1 basta `pip install bitsandbytes`.
- **`wfdb` falla al leer PTB-XL**: verificar que los `.hea` estén junto a los
  `.dat`. `build_qa.py` produce `.npy` la primera vez y los reutiliza en
  ejecuciones sucesivas.
- **`build_qa.py` descarta demasiados casos por concordancia**: los nombres de
  columna de PTB-XL+ pueden variar entre releases. Editar
  `data/build_qa.py::_normalize_features` para añadir los alias del release
  que se descargó.
- **`train_oracle_probe.py` produce accuracy ≈ 0.5** en todas las familias:
  probable colapso del batch norm con batch_size demasiado pequeño. Subir a
  `--batch_size 64` o `--batch_size 128` si hay VRAM.
