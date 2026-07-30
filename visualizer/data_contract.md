# Contrato de datos del visualizador de auditoría

Documento normativo: fija la interfaz entre el pipeline y el visualizador
`auditor_ecg.html`. Cualquier cambio de esquema debe consensuarse aquí antes
de tocar código, para que las fases del brief que producen datos y el
visualizador que los consume avancen en paralelo sin romperse mutuamente.

Fuente conceptual: sección 8.5 de `brief-implementacion.md`.

---

## 1. Archivo raíz

Nombre esperado por el visualizador: `outputs/audit/AUDIT.json`.

```jsonc
{
  "schemaVersion": "1.0",
  "generatedAt": "2026-07-29T12:00:00Z",
  "families": [ /* FamilyMeta[] */ ],
  "runs":    { /* RunId -> Run */ }
}
```

- `schemaVersion` sube minor con cambios retrocompatibles y major con
  cambios que rompen el visualizador.
- `families` es la lista de familias del conjunto controlado (F1..F5).
- `runs` está indexado por identificador de configuración (E0, E1, E2, E3).

---

## 2. `FamilyMeta`

```jsonc
{
  "id":       "F1",
  "label":    "QRS en V1 > 120 ms",
  "question": "¿La duración del QRS en V1 supera los 120 ms?",
  "gold":     "sí",                      // respuesta de referencia
  "leadIdx":  [6],                       // 0-based, índice de derivaciones citadas por la pregunta
  "span":     [2.05, 2.19],              // ventana de evidencia en segundos
  "locality": "puntual"                  // "puntual" | "regional" | "global"
}
```

`leadIdx` y `span` alimentan la vista V4: el mapa de sensibilidad usa
`leadIdx` para destacar la fila y `span` para dibujar la estrella (★).

---

## 3. `Run`

Un objeto por configuración. Cada `Run` incluye la geometría global de la
configuración y, dentro, una entrada por familia.

```jsonc
{
  "geometry": { /* Geometry */ },
  "perFamily": {
    "F1": {
      "bal": { /* Condition */ },
      "nat": { /* Condition */ }
    }
  }
}
```

`bal` corresponde a la condición balanceada; `nat` a la condición natural.
El visualizador usa `bal` por defecto y `nat` como comparación cuando el
usuario cambia de condición en la barra superior.

---

## 4. `Geometry`

Métricas de geometría medidas típicamente en la etapa del proyector. Todas
son escalares por configuración; alimentan V3 cuando la etapa activa es
`h_proj` o `h_fusion`.

```jsonc
{
  "normRatio":    0.12,   // ||h_proj|| / ||h_text||: 1.0 sería paridad
  "effRank":      3.1,    // rango efectivo (Roy & Vetterli, 2007)
  "attnMass":     0.041,  // masa de atención media sobre tokens de señal en el LLM
  "knnPurity":    0.08,   // fracción de vecinos-K del mismo gold en el espacio del LLM
  "jsDivergence": 0.0     // JS entre distribución de keys de señal y de texto (Zheng et al. 2025)
}
```

`normRatio` cercano a 1 y `attnMass` cercano a `1/L` (con L = longitud de la
secuencia) es la firma sana. `normRatio < 0.2` con `attnMass < 0.05` es la
firma de H4 (desalineación geométrica).

---

## 5. `Condition`

Todos los números que responden las cinco tareas del visualizador para una
familia y una condición.

```jsonc
{
  "blind":  0.50,          // A_blind: exactitud del modelo entrenado solo con texto
  "full":   0.63,          // A_full: exactitud multimodal
  "oracle": 0.96,          // A_oracle: exactitud de la sonda sobre la señal cruda
  "tdi":    0.72,          // 1 - (full-blind)/(oracle-blind); marcado como null si el denominador < 0.25
  "cfr":    0.19,          // Counterfactual Flip Rate ante swap
  "cfr0":   0.06,          // control: swap_null (mismo gold)
  "ls":     0.11,          // Localization Score: Δ(occl_gold) - Δ(occl_rand)
  "las":    0.04,          // Lead-Alignment Score: Δ(gold lead) - Δ(random lead)

  "ledger":        [0.96, 0.94, 0.55, 0.41, 0.40, 0.42, 0.45, 0.44],
  "ledgerControl": [0.50, 0.51, 0.50, 0.49, 0.50, 0.51, 0.50, 0.50],
  "ledgerStages":  ["h_raw","h_bigru","h_pool","h_proj","h_fusion","h_llm_0","h_llm_mid","h_llm_k"],

  "occlusion": [              // matriz derivación (12) x ventana (20) de Δ logit del gold
    [ /* 20 números */ ],
    /* 11 filas más */
  ],
  "occlusionMax": 0.34,       // máximo absoluto de la matriz; imprimir en la leyenda de V4

  "interventions": [ /* Intervention[] */ ],

  "cfrByDelta": [             // curva dosis-respuesta CFR(δ) — dos modalidades
    { "modality": "signal", "points": [ {"delta": 0.01, "cfr": 0.03}, {"delta": 0.05, "cfr": 0.11}, /* ... */ ] },
    { "modality": "text",   "points": [ {"delta": 0.01, "cfr": 0.09}, {"delta": 0.05, "cfr": 0.28}, /* ... */ ] }
  ],

  "cases": [ /* Case[] */ ]   // sólo los seleccionados para el estudio de casos
}
```

Convenciones:
- `blind`, `full`, `oracle`, `cfr`, `cfr0`, `ls`, `las` van en `[0, 1]`.
- `tdi` puede ser `null` cuando `oracle - blind < 0.25`; el visualizador
  entonces muestra `TDI †` y desactiva el color del cuadrante de V1.
- `ledger` y `ledgerControl` tienen la misma longitud; el visualizador la
  usa como ancla para la selección de etapa en V2.
- `occlusion` respeta el orden estándar de PTB-XL (I, II, III, aVR, aVL,
  aVF, V1..V6). El visualizador lo asume; documentar si cambia.
- `interventions` es la lista de las variantes ejecutadas sobre el caso
  representativo de la familia (para V5). Ver §6.

---

## 6. `Intervention`

Una entrada por variante contrafactual aplicada al caso representativo.
Alimenta V5 y las celdas del mapa V4 seleccionadas.

```jsonc
{
  "id":     "swap",
  "label":  "Sustitución por paciente con gold distinto",
  "answer": "no",
  "pGold":  0.61,           // probabilidad asignada al gold original tras la intervención
  "delta":  0.031           // desplazamiento representacional δ en la salida del encoder
}
```

`id` toma valores del catálogo del brief (§4.1):
`orig | zero | shuffle | occl_gold | occl_rand | occl_cell | swap | swap_null | lead_swap | paraphrase | neutral | conflict`.

---

## 7. `Case`

Caso concreto para el estudio de casos. Sólo se exportan los casos
seleccionados por `counterfactual/select_case_studies.py`.

```jsonc
{
  "id":         "case_1024",
  "ecgId":      "records500/00001_hr",
  "family":     "F1",
  "signal":     [ /* [12][T] o [T][12] tras z-score, según convención del visualizador */ ],
  "signalFs":   500,
  "prediction": "sí",
  "expected":   "sí",
  "hit":        true,
  "logitTop":   [ {"label": "sí", "logit": 4.2}, {"label": "no", "logit": 1.1} ],
  "verdict": {
    "dependencia":   "textual",     // "textual" | "ecg" | "ecg_partial" | "insensible"
    "etapa":         "h_pool",      // clave de LEDGER_STAGES donde cae la sonda
    "anclaje":       "espurio",     // "correcto" | "espurio" | "neutra" | "no_aplica"
    "clase":         "TEXT_DOMINANT"
  }
}
```

---

## 8. Notas para el productor

- La condición balanceada es la única sobre la que se pueden leer las
  métricas del ledger como diagnóstico; en `nat` el prior lingüístico
  contamina las cifras.
- Las curvas `cfrByDelta` requieren ejecutar el catálogo de intervenciones
  a varias intensidades (por ejemplo, oclusiones de longitud creciente,
  ruido escalado). El brief lo recomienda como sustituto de los puntos
  únicos QCFR/ECFR (Sec. 4.2).
- Cualquier campo puede omitirse si la fase que lo produce todavía no se
  ejecutó; el visualizador debe degradarse graciosamente y marcar el
  módulo correspondiente como "pendiente".

---

## 9. Compatibilidad hacia adelante

Cambios permitidos sin subir major:
- Añadir claves nuevas a `Geometry`, `Condition` o `Case`.
- Añadir entradas a `interventions` cuando el catálogo crezca.
- Añadir familias nuevas.

Cambios que rompen (suben major):
- Renombrar `ledger`, `ledgerControl`, `blind`, `full`, `oracle` o `tdi`.
- Cambiar el orden estándar de derivaciones sin flag explícita.
- Cambiar la unidad de `delta` en las intervenciones (hoy es distancia
  coseno en la salida del encoder).
