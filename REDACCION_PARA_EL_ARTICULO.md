# Redacción para el artículo · resultados y análisis

Texto listo para adaptar a las secciones de Resultados y Discusión, en el registro
y la notación que ya usa el artículo (QCFR, ECFR, `D_texto`, cont_texto, cont_ECG).
Las figuras referidas están en `figures/` en PNG y PDF a 300 dpi, dimensionadas
para columna simple (3,5″) o doble (7,16″).

> **Antes de integrar, leer la sección 7.** Enumera lo que hay que *quitar* del
> artículo actual porque quedó invalidado. Añadir lo nuevo sin retirar aquello
> dejaría el texto contradiciéndose.

---

## 1. Validación del escalamiento (sección nueva, va antes de los resultados actuales)

> **Fig. `fig_stbench_validacion`** — (a) exactitud sobre ST-Bench con IC 95 %
> frente al valor publicado de STReasoner-8B; (b) el efecto de sustituir la serie.

La arquitectura reducida se adaptó a ECG-QA sin comprobar antes que la receta de
escalamiento produjera un modelo funcional sobre la tarea original de STReasoner.
Para cerrar ese hueco se entrenó la misma configuración (Qwen3-0,6 B, encoder GRU,
QLoRA) sobre un subconjunto de ST-Bench de 1.600 muestras de entrenamiento y 60 de
test por tarea, muestreado por reservorio con semilla 42, y se evaluó con el
protocolo del repositorio original: extracción del contenido de `<answer>` y
reducción a una letra A–D. Para descartar que cualquier diferencia proviniera del
protocolo y no del modelo, ese evaluador se validó re-puntuando las generaciones
publicadas de STReasoner-8B, reproduciendo sus cuatro métricas de forma exacta
(0,871231156 · 0,757118928 · 0,956521739 · MAE 65,593473638).

En las tres tareas de opción múltiple el modelo reducido alcanza una exactitud
cuyo intervalo de confianza del 95 % **contiene el valor publicado del modelo de
8 B**: correlación 0,933 [0,841 · 0,974] frente a 0,871; entidad 0,833
[0,720 · 0,907] frente a 0,757; etiológico 0,967 [0,886 · 0,991] frente a 0,957.
La lectura correcta no es que el modelo reducido sea mejor, sino que a este tamaño
de muestra ambos son **estadísticamente indistinguibles**. La línea base sin
ajustar rinde 0,011 y no emite ninguna respuesta con formato válido, de modo que
el desempeño es atribuible al ajuste fino. No hay solapamiento de texto ni de
series entre las particiones de entrenamiento y test.

El resultado relevante, sin embargo, es el de la Fig. `fig_stbench_validacion`(b).
Manteniendo los cuatro tokens temporales en su sitio y sustituyendo únicamente su
contenido —por la serie de otra muestra o por una serie de ceros— **ninguna de las
90 respuestas cambia**: la tasa de cambio es cero exacto en las tres tareas. La
exactitud equivalente a la publicada se obtiene íntegramente a partir del texto.

Esto no demuestra que el modelo de 8 B del paper ignore la señal. Demuestra que
estas exactitudes son *alcanzables* sin leerla, de modo que la exactitud por sí
sola no evidencia razonamiento espacio-temporal. Es exactamente el argumento de
auditoría que sostiene este trabajo, y con ello el fenómeno de dominancia textual
deja de ser un hallazgo circunscrito a la adaptación a ECG-QA: se manifiesta
también en el benchmark de referencia.

---

## 2. La calibración de la intervención determina el índice de dominancia

> **Fig. `fig_recalibracion_ecfr`** — (a) tasa de cambio por intervención de ECG;
> (b) cómo cambia `D_texto` según cuáles se promedien.

El ECFR reportado no medía una intervención sino el promedio de cuatro, con
efectos que difieren en un orden de magnitud entre extremos (Fig.
`fig_recalibracion_ecfr`a): la oclusión temporal voltea el 15,0 % de las
respuestas y la máscara de derivaciones el 9,0 %, mientras el ruido gaussiano
llega al 1,0 % y el spike al 0,5 %.

La explicación es mecánica y se demuestra en la sección 3: el *pooling* del
encoder promedia sobre los 1.000 pasos temporales, de modo que un ruido de media
cero se cancela y un spike puntual se diluye, mientras que un tramo puesto a cero
no puede reconstruirse. Promediar las cuatro deflacta el ECFR e infla `D_texto` en
la misma medida:

| Intervenciones promediadas | QCFR | ECFR | `D_texto` |
|---|---|---|---|
| las cuatro | 0,280 | 0,064 | **0,216** |
| solo las estructuradas | 0,280 | 0,120 | 0,160 |
| solo oclusión temporal | 0,280 | 0,150 | **0,130** |

El valor de 0,150 obtenido por oclusión coincide con el 0,143 del test de ventanas
temporales reportado independientemente en el propio artículo (43 de 300 ventanas
cambian la respuesta): dos medidas de la misma cantidad, calculadas por caminos
distintos, que concuerdan.

El efecto no es uniforme por tipo de pregunta. En `single-query` la dominancia es
robusta a la calibración (0,436 con las cuatro, 0,404 con oclusión), pero en
`single-choose` **cambia de signo**: de +0,029 a −0,077. La afirmación de
dominancia textual en ese subgrupo dependía enteramente de una intervención
demasiado débil.

**Recomendación metodológica.** Reportar las tres filas de la tabla, no una. Que
las métricas de dominancia por perturbación sean sensibles a la calibración de la
intervención es un resultado que la literatura citada no discute, y es más
valioso que cualquiera de los tres números por separado.

---

## 3. Causa raíz: la atención temporal del encoder es un promedio uniforme

> **Fig. `fig_atencion_encoder`** — (a) curva de concentración del peso sobre la
> señal; (b) entropía de la atención por token.

El codificador temporal devuelve una matriz de atención `[tokens, T]` con el peso
que cada consulta aprendida asigna a cada paso de la serie. Es la única evidencia
directa de qué tramo lee el modelo, y no se estaba persistiendo. Al exportarla
sobre 100 muestras de test aparece el hallazgo central de este trabajo:

**en la configuración original la atención no atiende.** La entropía normalizada
por token es 0,9999 [0,9999 · 1,0000], donde 1,0 corresponde a una distribución
uniforme sobre los 1.000 pasos. Verificado sobre la matriz cruda, sin agrupar: el
peso uniforme es 0,001000 y el observado va de 0,000924 a 0,001099, es decir, el
paso más atendido recibe apenas un 10 % más que un promedio plano. Los cuatro
tokens correlacionan entre sí entre 0,72 y 0,91: no se especializan.

El mecanismo es un desajuste de escala. Las consultas se inicializan como
`randn·0,02`, con norma 0,32, y tras entrenar quedan en 0,35. Se multiplican
contra salidas del encoder normalizadas por *LayerNorm*, cuya norma es
√256 = 16, y el producto se divide de nuevo por √256. Los logits resultantes
abarcan un rango de 0,18 cuando un *softmax* sobre 1.000 posiciones necesita un
rango del orden de log(1.000) ≈ 6,9 para apartarse de la uniformidad: **38 veces
menos de lo necesario**. Y como el *softmax* uniforme es una región plana, el
gradiente apenas puede escapar de ella.

Esto explica de forma unificada tres observaciones que hasta ahora parecían
independientes:

1. el ruido gaussiano de media cero se cancela, porque el *pooling* es
   literalmente un promedio sobre 1.000 muestras;
2. el contenido frecuencial se destruye por la misma razón;
3. `cont_ECG` ronda cero porque la representación es prácticamente la media
   temporal de una señal ya normalizada, es decir, casi constante.

Al igualar la escala de las consultas a la de las claves (`query_init_std = 1,0`),
la norma pasa a 15,98 y la atención se concentra: la entropía media baja a
0,9490 [0,9472 · 0,9509] y el token más enfocado alcanza 0,68 [0,67 · 0,70], con
casos individuales en 0,42. Los intervalos de ambas configuraciones **no se
solapan**. En la ventana más atendida el peso pasa de 1,05× a 1,71× el uniforme
(Fig. `fig_atencion_encoder`a).

---

## 4. Fidelidad del encoder

> **Fig. `fig_fidelidad_encoder`** — R² de probes lineales sobre descriptores de
> la señal, con el encoder sin entrenar como referencia.

La revisión de tesis planteó una pregunta que el artículo no respondía: cómo saber
que el codificador refleja la señal de entrada. La exactitud en la tarea no puede
responderla, porque un modelo puede acertar sin leer la señal. Se ajustan probes
ridge con validación cruzada de cinco pliegues desde cada representación hacia
descriptores de la señal, sobre 250 muestras.

La elección de los descriptores es parte del resultado. `TimeSeriesEncoder`
aplica z-score por derivación antes del GRU, de modo que la escala absoluta se
descarta *por diseño*: probar por la media o la desviación mide la normalización
funcionando, no la fidelidad. Los descriptores informativos son los que sobreviven
a la estandarización, y entre ellos el más relevante clínicamente es la frecuencia
dominante, que en un ECG corresponde al ritmo cardíaco.

| Descriptor | Config. original | Sin entrenar | Encoder ampliado |
|---|---|---|---|
| Frecuencia dominante (ritmo) | 0,157 | *0,247* | **0,329** |
| Autocorrelación (lag 1) | 0,855 | *0,864* | 0,855 |
| Rango | 0,346 | *0,290* | 0,411 |

La comparación que importa es contra el encoder **sin entrenar**. En la
configuración original el encoder entrenado representa el ritmo cardíaco *peor*
que un GRU con pesos aleatorios (0,157 frente a 0,247): el entrenamiento degrada
la característica clínicamente más relevante. Con el encoder ampliado y las
escalas corregidas pasa a representarlo claramente mejor (0,329). En
autocorrelación —lo único que el encoder codifica bien— un GRU sin entrenar ya
alcanza 0,864, de modo que el entrenamiento no aporta nada por esa vía.

Conviene señalar que este R² es un valor con validación cruzada al que no se le
calculó intervalo de confianza, de modo que su magnitud debe citarse como
indicativa.

---

## 5. Efecto de corregir la arquitectura

> **Fig. `fig_trazado_etapas`** — trazado representacional interno por etapa, con
> ruido gaussiano y con oclusión temporal.
> **Fig. `fig_intervalos_confianza`** — contribuciones modales con IC 95 %.

### 5.1 El trazado interno con la intervención calibrada

Re-ejecutado sobre los mismos 50 casos, cambiando únicamente la intervención de
ruido a oclusión temporal, **el impacto del ECG deja de ser cero exacto en las
cinco etapas**:

| Etapa | Texto | ECG (ruido) | ECG (oclusión) | Δ (oclusión) |
|---|---|---|---|---|
| Encoder temporal | n/d | 0,0000 | **0,0964** | n/d |
| Proyector latente | n/d | 0,0000 | **0,1238** | n/d |
| Fusión `inputs_embeds` | 0,0001 | 0,0000 | **0,1236** | **−0,1235** |
| LLM | 0,0219 | 0,0000 | 0,0195 | +0,0024 |
| Logits / Respuesta | 0,0165 | 0,0000 | 0,0013 | +0,0152 |

En la fusión el signo se invierte: la representación es mucho más sensible al ECG
que al texto (Δ = −0,1235). La señal entra con fuerza y **se degrada dentro del
LLM** —0,1236 → 0,0195 → 0,0013, casi dos órdenes de magnitud— mientras el impacto
del texto se mantiene estable. La etapa de amplificación sigue siendo el LLM, pero
por el motivo contrario al que se reportaba: no porque crezca el peso del texto,
sino porque colapsa el del ECG. Aparecen además, por primera vez, casos
`ECG_DOMINANT` (3) y `BALANCED` (3); `TEXT_DOMINANT` pasa de 46/50 a 41/50.

### 5.2 Lo que sobrevive a un intervalo de confianza

Con bootstrap pareado de 1.000 remuestreos sobre las mismas muestras
(Fig. `fig_intervalos_confianza`):

| Métrica | Config. original | Encoder ampliado |
|---|---|---|
| cont_texto | 0,386 [0,341 · 0,431] | 0,395 [0,350 · 0,440] |
| cont_ECG | 0,006 [−0,019 · 0,034] | 0,036 [−0,003 · 0,075] |
| Dominancia por ablación | 0,379 [0,332 · 0,426] | 0,358 [0,314 · 0,404] |

**Se puede afirmar:** la contribución del texto es grande y significativa en ambas
configuraciones; la del ECG **no se distingue de cero en ninguna**; la dominancia
textual es significativa por dos vías independientes —ablación, con intervalo
lejos de cero, y contrafactual, con prueba de permutación pareada p = 0,004 y
p = 0,001—; y el modelo supera el azar en preguntas binarias (p = 0,0028 y
p = 0,0197, n = 239).

**No se puede afirmar:** que el encoder ampliado mejore la contribución del ECG.
La diferencia pareada es +0,030 [−0,012 · +0,070] y no es significativa. Tampoco
que `cont_ECG = 0,036` quede por debajo del umbral operativo de 0,05, porque el
intervalo lo contiene. Con n = 300 el dato no decide.

---

## 6. Discusión

**Un diagnóstico mecánico, no una solución.** El trabajo pasa de constatar que el
modelo no usa la señal a explicar por qué, con cada eslabón medido: consultas de
atención fuera de escala → *softmax* uniforme → promedio temporal → contenido
frecuencial destruido → contribución del ECG indistinguible de cero. Corregirlo
cambia de forma inequívoca el comportamiento **interno** del codificador, con
intervalos que no se solapan. Lo que no está demostrado es que ese cambio se
traduzca en más contribución de la señal a la respuesta final.

**El índice `D_texto` está confundido.** Además de depender de la calibración
(sección 2), resta dos cantidades que no son conmensurables. Al ampliar el
encoder, el ECFR por oclusión cae de 0,150 a 0,010 y `D_texto` sube de 0,13 a
0,48, justo cuando la ablación indica que el ECG contribuye más. No es una
contradicción: con cuatro tokens que promedian toda la señal, tapar un cuarto de
ella desplaza el promedio y voltea la respuesta; con treinta y dos tokens
enfocados, los que no cubren la ventana ocluida siguen aportando. **Un ECFR menor
indica mayor robustez a la oclusión parcial, no menor uso de la señal.** Un modelo
puede volverse a la vez más frágil al texto y más anclado en la señal, y el índice
lo reportará como «más textualmente dominante». La contribución por ablación no
comparte este problema, porque elimina la modalidad entera en lugar de perturbarla.
Se recomienda no usar `D_texto` como métrica única y declarar explícitamente que
ambas miden cosas distintas.

**Las intervenciones mal calibradas fallan en las dos direcciones.** El ruido
gaussiano era demasiado débil y subestimaba el papel del ECG. La ablación por
eliminación, en el modelo entrenado sobre ST-Bench, resultó ser demasiado
destructiva: al retirar los tokens temporales el modelo pierde la señal que lo
mantenía en el formato aprendido, revierte al comportamiento del modelo base y
agota el presupuesto de generación sin emitir respuesta, de modo que su exactitud
de 0,000 —por debajo del azar— mide un colapso de formato y no la contribución de
la modalidad. De ahí que toda ablación deba reportar la tasa de respuestas
parseables junto a su métrica.

**Sobre el diseño de los benchmarks.** Cerca de una cuarta parte del conjunto de
alineación de ST-Bench, diseñado precisamente para enseñar al modelo a leer la
señal, consiste en preguntas sobre la estructura del grafo, que está escrita
literalmente en el enunciado. Entrenada esa etapa, el modelo alcanza 0,958 en esas
preguntas y 0,182 en las que exigen leer la señal, emitiendo entre una y cinco
respuestas distintas por tipo de pregunta. El desempeño agregado de 0,637 no
describe un modelo que lee series, sino uno que aprendió la moda de cada tipo.

---

## 7. Qué hay que retirar o corregir del artículo actual

**Obligatorio**

1. **Tabla IV completa** (trazado interno por etapa). Todos los impactos del ECG
   eran cero por artefacto de la intervención. Sustituir por la tabla de la
   sección 5.1.
2. **Sección VI-J.** La afirmación de que el desbalance «se manifiesta desde la
   fusión y se amplifica en el LLM» debe rehacerse: en la fusión el desbalance
   favorece al ECG, y lo que ocurre en el LLM es que la señal se pierde.
3. **Toda afirmación de que «la señal ECG no participa»** o de que «el componente
   multimodal está presente pero no en la decisión». Con ECFR = 0,150 e impacto de
   0,124 en la fusión, participa.
4. **Los valores de ECFR y de `D_texto`** que dependan de la intervención por
   ruido, y las conclusiones derivadas.
5. **La lectura de `cont_ECG` como aporte pequeño pero existente.** El intervalo
   [−0,019 · 0,034] incluye el cero: no hay evidencia de aporte. Esto *refuerza*
   la tesis del artículo, no la debilita.

**Recomendado**

6. Reportar las tres calibraciones de la sección 2 y presentar la sensibilidad a
   la calibración como contribución metodológica propia.
7. Reportar `D_texto` por tipo de pregunta y señalar el cambio de signo de
   `single-choose`.
8. Añadir la advertencia de la sección 6 sobre la no conmensurabilidad de QCFR y
   ECFR.

**Se mantiene sin cambios**

9. La ablación modal: `no_series` elimina la modalidad entera en vez de
   perturbarla, de modo que `cont_ECG` es una medida limpia.
10. El test de oclusión temporal, que ya usaba una intervención estructurada y
    ahora además concuerda con el ECFR recalibrado.
11. QCFR y la prueba de la pregunta neutral: la intervención textual no se modificó.

---

## 8. Limitaciones a declarar

1. La configuración con encoder ampliado combina tres cambios (capacidad, escala
   del proyector y escala de las consultas) y no permite atribuir entre ellos. Hay
   evidencia independiente de cada mecanismo, pero no una ablación por separado.
2. Las tres afirmaciones no concluyentes de la sección 5.2 lo son por tamaño de
   muestra, no por el efecto observado. La ablación usa 300 de las 750 muestras de
   test disponibles.
3. El R² del probe de fidelidad no lleva intervalo de confianza.
4. El análisis por tipo de pregunta tiene grupos de entre 4 y 62 muestras y no
   lleva corrección por comparaciones múltiples; en particular, el cambio de signo
   de `single-choose` se apoya en n = 26.
5. La corrida con encoder ampliado no había dejado de mejorar en su última época,
   de modo que sus cifras no son el techo de esa configuración.
6. El z-score se aplica dos veces —en el preprocesamiento y dentro del encoder—,
   de modo que la amplitud de la señal es irrecuperable desde la representación.

---

## 9. Índice de figuras

| Archivo | Contenido | Ancho sugerido |
|---|---|---|
| `fig_stbench_validacion` | Validación sobre ST-Bench e intervenciones sobre la serie | doble columna |
| `fig_recalibracion_ecfr` | ECFR por intervención y efecto sobre `D_texto` | doble columna |
| `fig_atencion_encoder` | Concentración de la atención y entropía por token | doble columna |
| `fig_trazado_etapas` | Trazado interno: ruido frente a oclusión | doble columna |
| `fig_fidelidad_encoder` | R² de los probes lineales | columna simple |
| `fig_intervalos_confianza` | Contribuciones modales con IC 95 % | columna simple |

Todas se regeneran con `python scripts/make_paper_figures.py`, que lee
exclusivamente los artefactos de `outputs/`; ningún valor está escrito a mano.
