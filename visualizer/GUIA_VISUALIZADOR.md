# Guía del visualizador · Trazado representacional interno (Mini-STReasoner-ECG)

Este visualizador audita **dónde se atenúa o se amplifica la influencia del ECG frente al texto**
durante la inferencia del modelo multimodal (ECG → Encoder temporal → Proyector latente →
Fusión `inputs_embeds` → LLM → Respuesta).

Todo se calcula **offline** con `xai/representational_tracing.py`; el visualizador solo **lee**
los resultados precomputados (`tracing_data.js`). No recalcula el modelo ni hace backpropagation.
Si un valor no existe, muestra **"no disponible"** (no inventa).

---

## 0. Conceptos clave (vocabulario)

| Término | Qué es |
|---|---|
| **Escenario** | Una intervención contrafactual aplicada a la entrada (ver §1). |
| **impacto_texto** | Distancia representacional entre *Original* y *Texto modificado* (Text-CF) en una etapa. Mide cuánto se mueve la representación al editar el texto. |
| **impacto_ECG** | Distancia entre *Original* y *ECG modificado* (ECG-CF) en una etapa. Mide cuánto se mueve al perturbar la señal. |
| **diferencia_texto_ECG (Δ)** | `impacto_texto − impacto_ECG`. Positivo = la etapa es más sensible al texto; negativo = más al ECG. |
| **D_text** | Cuota textual de la salida = `impacto_texto / (impacto_texto + impacto_ECG)`. 1 = todo texto, 0 = todo ECG. |
| **Distancia** | Coseno (por defecto) o L2 normalizada entre vectores de activación. |
| **Clase de dominancia** | Resumen por caso: `TEXT_DOMINANT`, `ECG_DOMINANT`, `BALANCED`, `INSENSITIVE`, `UNCLEAR`. |

**Colores:** naranja = texto · turquesa/azul = ECG · gris = balanceado.

**Lenguaje correcto:** el sesgo **se manifiesta o se amplifica** en una etapa; *no* se afirma que "nace" en ella.
Las comparaciones son **evidencia local por intervención**, *no* atribución causal.

---

## 1. Selector de escenario (panel lateral)

Cambia la entrada del caso y **coordina todas las vistas** (V1–V5) por `case_id` + `scenario`:

| Escenario | Intervención | Foco |
|---|---|---|
| **Original** | entrada sin tocar | ambos |
| **Pregunta modificada** | reescritura que preserva el significado (Text-CF) | texto |
| **Pregunta neutral** | se elimina la pista clínica | texto |
| **ECG modificado** | señal perturbada (ruido) | ECG |
| **Texto contradictorio** | el texto afirma lo contrario al ECG | texto |
| **Sin ECG** | se elimina la modalidad temporal (`use_series=false`) | ECG |

> Solo *Pregunta modificada* (impacto_texto) y *ECG modificado* (impacto_ECG) tienen distancias por
> etapa propias. Neutral / contradictorio / sin-ECG solo tienen predicción de salida; sus columnas
> internas aparecen como "no disponible".

---

## 2. Panel lateral (resto)

- **Badge de modo**: *trazado interno* (hay activaciones internas) o *contrafactual global* (solo salida).
- **D_text (cuota texto)**: barra naranja/turquesa + clase de dominancia del caso.
- **‹ ›**: navegar entre casos del filtro actual.
- **Filtros**:
  - *Dominancia del caso* (5 clases).
  - *Comportamiento*: solo cambia con texto / solo con ECG / no cambia con nada / conflicto texto-ECG.
  - *Dominancia textual*: alta / media / baja.
  - *Tipo de pregunta* y *Tipo de atributo*.
- **Tabla `Caso | D_text`**: casos ordenados de mayor a menor D_text; clic = cargar el caso. Selector rápido de casos críticos.

---

## V1 · Trazado representacional interno

El flujo `ECG → Encoder → Proyector → Fusión → LLM → Respuesta`. En cada bloque con datos:

- **barra naranja** = impacto_texto, **barra turquesa** = impacto_ECG, y abajo la **Δ** (color según quién domina).
- *Encoder* y *Proyector* solo muestran impacto_ECG (el texto no atraviesa esas etapas → impacto_texto = "n/d").
- El escenario **atenúa** la modalidad que no es su foco.

**Interacciones:**
- **Hover** en un bloque → tooltip con impacto_texto, impacto_ECG, diferencia y diagnóstico de la etapa.
- **Clic** en un bloque → panel de detalle (Original vs Text-CF, Original vs ECG-CF, distancia usada, Δ, diagnóstico).
- **▶ Reproducir flujo** → anima paso a paso (ECG/pregunta → encoder → proyector → fusión → LLM → respuesta → diagnóstico), con explicación textual en cada paso. **⟲ Reiniciar** detiene y vuelve al inicio.

---

## V2 · Sensibilidad modal por etapa

Tabla/heatmap con una fila por etapa (Encoder, Proyector, Fusión, LLM, Salida) y columnas:
**Impacto texto · Impacto ECG · Δ texto−ECG · Diagnóstico**.

- Las celdas de impacto se colorean por intensidad (más oscuro = mayor); el texto es blanco sobre fondos fuertes.
- El escenario **resalta** la columna de su modalidad.
- **Clic** en el nombre de una etapa = abre el mismo panel de detalle que V1.
- La nota inferior indica en qué etapa **se amplifica** el desbalance.
- Valores ausentes → "n/d" (no se inventan).

---

## V3 · Espacio latente texto↔ECG

Proyección 2D (PCA) de las representaciones del caso:

- **Puntos**: tokens de pregunta (naranja), opciones, ECG original (turquesa), ECG perturbado (azul).
- **Flechas**: desplazamiento *original → contrafactual* (ECG original→perturbado y pregunta original→modificada).
- El escenario **resalta** la modalidad activa y atenúa la otra.

**Interacciones:**
- **Rueda** = zoom (útil para abrir el cúmulo de tokens de texto, que suelen caer casi en el mismo punto).
- **Arrastrar** = desplazarse. **Doble clic** = reiniciar la vista.
- **Hover** en un punto → su etiqueta y tipo.

---

## V4 · Comparación contrafactual local

Muestra **entrada original vs intervenida** del escenario activo. *No es atribución causal.*

**Columna de texto:**
- *Pregunta original* con el **peso (relevancia) de cada palabra**: tinte por peso y valor al pasar el mouse (hover).
- *Pregunta intervenida*: las **palabras modificadas** van resaltadas; clic en una palabra → comparación de frase
  (original/intervenida, respuesta original/contrafactual, si cambió, D_text del caso). Sin datos por palabra,
  se muestra la frase completa (no se inventa peso token-level).
- Badge: **respuesta cambió / no cambió / no disponible**.

**Columna de serie temporal (ECG):**
- Onda **ECG original** (turquesa) y, en *ECG modificado*, la **intervenida** (azul punteado).
- El **fondo** colorea el **peso por segmento**; hover muestra el valor.
- Marco rojo = **segmento alterado**.
- En *Sin ECG* la onda se atenúa con la nota "ECG removido".

> Los pesos reales vienen de `attributions.jsonl` (saliencia por gradiente, etapa 7b). El trazado los **lee
> de disco**; no recalcula gradientes. Sin ese archivo, el ECG cae a un proxy de energía (etiquetado como tal).

---

## V5 · Pregunta · respuesta contrafactual

Lista las respuestas del modelo en todas las condiciones:
**original · esperada · pregunta modificada · ECG modificado · pregunta neutral · conflicto · sin ECG**.

- La fila del **escenario activo** queda marcada; **clic** en una fila = cambiar a ese escenario.
- Las respuestas que difieren de la original se marcan (flip).
- Abajo, el **veredicto**: clase de dominancia + D_text con una explicación corta.

---

## Reglas de honestidad (importantes para la tesis)

1. No hay backpropagation en el trazado: es un **trazado representacional interno** (distancias entre activaciones).
2. No se afirma que el sesgo "nace" en una capa; **se manifiesta o se amplifica**.
3. Las comparaciones son **evidencia local por intervención / sensibilidad modal**, no causalidad.
4. Si faltan activaciones internas, el visualizador lo marca como **"contrafactual global"** y muestra solo la salida.
5. Cualquier valor ausente se muestra como **"no disponible" / "n/d"**; nunca se rellena con datos inventados.
