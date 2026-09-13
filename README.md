# Mantenimiento Predictivo — MAI 540

Herramienta de exploración y preparación de datos para predecir **`failure_next_7_days`**
(1 = la máquina falla en los próximos 7 días).

**Autora:** Araceli Castillo · MAI 540: Machine Learning · Prof. Kevin A. Garcia Gallardo
**Institución:** Atlantis University · **Tarea 2.2** · Septiembre de 2026

---

## 1. Cómo ejecutar esto

### Opción A — Google Colab (recomendada, no requiere instalar nada)

1. Abre <https://colab.research.google.com>
2. Pestaña **GitHub** → pega `ACAS0802/mantenimiento-predictivo-mai540`
3. Abre `notebooks/Tarea_2_2_Preprocesamiento.ipynb`
4. Menú **Entorno de ejecución → Ejecutar todas**

La primera celda clona el repositorio y localiza el dataset sola. No hay que configurar rutas ni subir archivos. Tiempo aproximado: 3 a 5 minutos (la sección 8.4 corre 30 validaciones cruzadas).

### Opción B — En tu computadora

```bash
git clone https://github.com/ACAS0802/mantenimiento-predictivo-mai540.git
cd mantenimiento-predictivo-mai540
pip install -r requirements.txt

python3 verificar_fuga.py        # pruebas anti-fuga (~30 s)
jupyter notebook notebooks/Tarea_2_2_Preprocesamiento.ipynb
```

**Requisitos:** Python 3.10 o superior. Las versiones exactas están en `requirements.txt`.

### Reproducibilidad

Todo usa `random_state=42` y `test_size=0.25`. Ejecutar dos veces da exactamente los mismos números. Los resultados de este README se obtuvieron con Python 3.10 y scikit-learn 1.7.2.

---

## 2. Qué hay en el repositorio

| Archivo | Qué contiene |
|---|---|
| `notebooks/Tarea_2_2_Preprocesamiento.ipynb` | **Entregable principal.** El pipeline completo, ejecutado y con salidas guardadas |
| `src/preprocesamiento.py` | El pipeline como módulo reutilizable, importable desde otro código |
| `verificar_fuga.py` | Siete pruebas que demuestran que no hay fuga de información |
| `main.py` | Modelo de la Tarea 1.2 (se conserva como línea base del ANTES) |
| `data/datos.csv` | 5 000 registros, 10 predictoras y la variable objetivo |
| `data/DESCRIPCION.md` | Diccionario de variables |
| `CONTEXT.md` | Reglas de alcance del proyecto (v1.2) |
| `BITACORA_Tarea2.2.md` | Bitácora de desarrollo asistido |
| `reportes/` | Salidas generadas: conjunto final preprocesado |

---

## 3. El orden de las transformaciones y por qué evita la fuga

**Esta es la sección más importante del proyecto.**

```
1. Cargar datos crudos
2. Limpieza SIN estadísticas  ──┐ puede ir antes del split porque
   · duplicados exactos         │ no aprende ningún parámetro
   · reglas físicas del dominio ┘ (y los duplicados DEBEN quitarse aquí)
3. train_test_split  ═══════════════ LA FRONTERA ═══════════════
4. Imputación ────┐
5. Escalamiento   ├─ dentro de un Pipeline: fit() solo con train,
6. Codificación ──┘  transform() se aplica a test
7. Selección de características ── decidida con CV sobre train
8. Evaluación en test ── una sola vez
```

### Por qué cada paso está donde está

**Los duplicados se quitan ANTES del split.** Es la única operación de limpieza que debe ir antes por razones de fuga, no a pesar de ellas: si una fila repetida queda en `train` y su copia idéntica en `test`, el modelo ya vio la respuesta y el test deja de medir generalización. Ningún `Pipeline` corrige eso después. Y como quitar duplicados no calcula ninguna estadística, hacerlo sobre el conjunto completo no filtra nada.

**Las reglas físicas también van antes.** "La presión no puede ser negativa" es conocimiento del dominio, no un parámetro estimado de los datos. No mira la distribución, así que no filtra información.

**La imputación va DESPUÉS del split y dentro del Pipeline.** La mediana es un parámetro *aprendido*. Si se calcula con el conjunto completo, la mediana usada para rellenar una fila de entrenamiento incorpora los valores del test. `Pipeline.fit(X_train)` hace estructuralmente imposible ese error.

**El escalamiento, igual.** `StandardScaler` aprende media y desviación estándar. Ajustarlo con todo el conjunto mete estadísticas del test en cada fila de entrenamiento.

**Los límites de atípicos se calculan solo con train.** Decidir qué filas son extremas usando los cuantiles del conjunto completo es decidir con información del test.

**La selección de características se decide solo con train.** Este es el paso nuevo de la Tarea 2.2 y —según se midió— es donde la fuga realmente muerde. Ver la sección 6.

---

## 4. Decisiones de preprocesamiento, con su evidencia

### 4.1 Datos faltantes

El dataset tiene faltantes en `vibration_mm_s` (2.20 %), `pressure_bar` (3.36 %) y `environment` (2.26 %).

Primero se comprobó si eran aleatorios, porque eso cambia la estrategia:

| Variable | Tasa de falla si FALTA | Tasa de falla si NO falta |
|---|---|---|
| `pressure_bar` | 11.90 % | 8.77 % |
| `vibration_mm_s` | 4.55 % | 8.98 % |
| `environment` | 8.85 % | 8.88 % |

No son completamente aleatorios: en `pressure_bar` y `vibration_mm_s` el hecho de faltar está asociado al resultado. Eso hizo razonable probar un indicador de faltante.

Se compararon **tres** estrategias por validación cruzada de 5 particiones **sobre train**:

| Estrategia | F1 clase 1 | Recall | Precisión |
|---|---|---|---|
| A) Mediana | **0.2650** | 0.6578 | 0.1660 |
| B) KNN (k=5) | 0.2647 | 0.6578 | 0.1659 |
| C) Mediana + indicador de faltante | 0.2621 | 0.6487 | 0.1643 |

**Decisión: mediana.** No porque sea la mejor —las tres empatan dentro del ruido— sino porque cuando hay empate gana la más simple:

- es robusta a valores atípicos, y en 4.2 se decide conservarlos;
- no añade columnas, a diferencia de C, que crearía variables fuera de las autorizadas;
- es mucho más barata que KNN, que calcula distancias entre filas.

El indicador de faltante era una hipótesis razonable y **los datos la descartaron**. Con 2–3 % de faltantes no hay suficiente señal.

### 4.2 Duplicados y valores atípicos

**Duplicados: cero.** Ni filas completas ni filas repetidas solo en las predictoras. El código de limpieza se conserva igual, porque la herramienta debe funcionar con datos que sí los tengan.

**Atípicos: se conservan todos.** Límites IQR calculados solo con train:

| Variable | Atípicos | Falla en atípicos | Falla en normales | Lift |
|---|---|---|---|---|
| `temperature_c` | 32 | **25.00 %** | 8.74 % | **2.86×** |
| `days_since_maintenance` | 81 | **18.52 %** | 8.67 % | **2.14×** |
| `pressure_bar` | 25 | 16.00 % | 8.64 % | 1.85× |
| `operating_hours` | 19 | 15.79 % | 8.84 % | 1.79× |
| `energy_kw` | 30 | 10.00 % | 8.87 % | 1.13× |
| `error_count_30d` | 37 | 8.11 % | 8.89 % | 0.91× |
| `vibration_mm_s` | 17 | 0.00 % | 9.01 % | 0.00× |

La evidencia que respalda conservarlos:

1. **Fallan entre 2 y 3 veces más que las filas normales.** Tiene sentido físico: una máquina fuera de su rango de temperatura, o con 300 días sin mantenimiento, es justo la que está por fallar. **El atípico es la señal, no el ruido.**
2. **Eliminarlos destruiría la clase minoritaria.** Las filas con al menos un atípico son el 6.03 % de train pero concentran el **9.91 % de todas las fallas**: borrarlas quitaría 33 de 333 fallas en un problema donde la clase positiva ya es solo el 8.88 %.
3. **No hay ningún valor físicamente imposible:** ningún negativo, ningún rango absurdo. No hay evidencia de error de medición.

**Excepción anotada:** los 17 atípicos de `vibration_mm_s` tienen 0 % de fallas y sí parecen ruido de sensor. Pero 17 filas sobre 3 750 no bastan para distinguir señal de azar, así que **se conservan y queda registrado como limitación**.

### 4.3 Codificación y escalamiento

El criterio es el algoritmo: `LogisticRegression(max_iter=1000, class_weight="balanced")`.

- **Numéricas → `StandardScaler`.** Obligatorio aquí. La regresión logística optimiza por gradiente y penaliza coeficientes; con `operating_hours` en miles y `vibration_mm_s` entre 0 y 7, sin escalar la penalización castiga injustamente a las variables de escala pequeña.
- **Categóricas → `OneHotEncoder(handle_unknown="ignore")`.** Las tres son nominales: no hay orden entre `A`, `B` y `C`. Codificarlas como enteros le haría creer al modelo que C > B > A. El `handle_unknown` evita que el pipeline se rompa si aparece un nivel nuevo en producción.
- **No se usa target encoding** en ninguna forma: usa la variable objetivo para construir predictoras, y `CONTEXT.md` 2.2 lo prohíbe.

---

## 5. Selección de características

> **Nota de alcance.** `CONTEXT.md` v1.1 prohibía la selección de variables sin autorización escrita de la autora. La Tarea 2.2 la exige. La autorización quedó registrada y `CONTEXT.md` subió a v1.2. La regla no se saltó: se cambió a propósito y quedó documentado.

Cuatro fuentes de evidencia, todas medidas **solo sobre train**.

**Multicollinealidad:** la correlación máxima entre numéricas es 0.222 (`operating_hours` ~ `vibration_mm_s`), muy por debajo de 0.8. **Ninguna variable se descarta por redundancia.**

**Asociación con el objetivo:**

| Variable | Corr. punto-biserial | Información mutua |
|---|---|---|
| `vibration_mm_s` | 0.1860 | 0.0332 |
| `operating_hours` | 0.1009 | 0.0256 |
| `temperature_c` | 0.0579 | 0.0237 |
| `energy_kw` | −0.0357 | 0.0167 |
| `pressure_bar` | −0.0219 | 0.0162 |
| `days_since_maintenance` | 0.1244 | 0.0064 |
| `error_count_30d` | 0.0697 | **0.0016** |

**Ablación con prueba de significancia.** Cinco particiones no bastan para afirmar que una diferencia de +0.007 significa algo, así que se repitió la validación cruzada 6 veces (30 mediciones) y se aplicó una prueba t pareada:

| Subconjunto | F1 | Δ vs. 10 variables | p | Veredicto |
|---|---|---|---|---|
| 10 variables (todas) | 0.2715 ± 0.0205 | — | — | referencia |
| sin `error_count_30d` | 0.2739 ± 0.0210 | +0.0024 | 0.4268 | dentro del ruido |
| **sin `shift`** | **0.2760 ± 0.0208** | **+0.0045** | **0.0093** | **diferencia real** |
| sin ambas | 0.2732 ± 0.0186 | +0.0017 | 0.5601 | dentro del ruido |

### Decisión: se descarta `shift`. Se conservan las otras nueve.

**Se descarta `shift`** por tres razones que apuntan igual:

- **Estadística.** Es el único cambio cuya mejora supera el ruido (p = 0.0093 sobre 30 mediciones pareadas). Los demás dan p > 0.4.
- **Empírica.** Sus niveles tienen tasas de falla casi idénticas — mañana 7.78 %, noche 9.69 %, tarde 9.39 % — pero cuesta dos columnas tras el one-hot.
- **De dominio.** El turno es una etiqueta administrativa, no una condición física. No hay mecanismo por el que el turno de la tarde haga fallar un rodamiento.

**Se conserva `error_count_30d`** pese a tener la información mutua más baja (0.0016). Quitarla da p = 0.4268: está dentro del ruido. El dominio dice que los errores recientes son un indicador legítimo del estado de una máquina. **Sin evidencia para eliminarla, la decisión conservadora es conservarla.** Descartar una variable plausible por una mejora que no supera el azar sería sobreajustar la decisión a esta partición.

**Resultado: 9 variables** (7 numéricas + `machine_type` + `environment`), que tras el one-hot se convierten en 13 columnas.

---

## 6. Verificación de que no hay fuga

`python3 verificar_fuga.py` corre siete pruebas. Las seis primeras comprueban lo estructural: el objetivo no está entre las predictoras, el escalador y el imputador aprendieron de train y no del conjunto completo, cada fila de test se transforma sin mirar a las demás, y ninguna fila aparece a los dos lados de la partición.

La séptima es una prueba de contraste, y el resultado merece leerse con atención:

**6a — Fuga por imputación y escalamiento: medida, no supuesta.** Se construyó a propósito la versión con fuga (ajustar el preprocesamiento con todo el conjunto) y se comparó. **La diferencia en F1 es 0.0000.**

No se maquilla el resultado. La razón es concreta: con 5 000 filas y 2–3 % de faltantes, las medianas de train y las del conjunto completo difieren como máximo un **0.26 %**. La fuga existe conceptualmente pero aquí es numéricamente inocua.

Eso **no** significa que el orden dé igual. Significa que el orden correcto es una póliza de seguro: con menos filas, más faltantes o un escalador sensible a atípicos la diferencia sí aparece, y no se puede saber de antemano si un dataset es benigno sin haberlo medido.

**6b — La fuga que sí importa aquí: la selección de características.** Elegir variables mirando el conjunto completo, contra elegirlas solo con train, **infla el F1**. Es precisamente el paso que introduce la Tarea 2.2, y por eso toda la sección 5 se decidió con validación cruzada sobre train y el conjunto de prueba no se tocó hasta el final.

---

## 7. Resultados

Evaluación única sobre el conjunto de prueba (1 250 filas, 111 fallas reales):

| Métrica | ANTES (Tarea 1.2, 10 var.) | DESPUÉS (Tarea 2.2, 9 var.) |
|---|---|---|
| Accuracy | 0.6856 | 0.6872 |
| Precisión (clase 1) | 0.1527 | 0.1667 |
| **Recall (clase 1)** | 0.5586 | **0.6306** |
| **F1 (clase 1)** | 0.2398 | **0.2637** |
| Fallas detectadas | 62 de 111 | **70 de 111** |

El modelo final detecta **8 fallas más** con una variable menos. En mantenimiento predictivo el recall es lo que importa: una falla no detectada es una máquina que se rompe sin aviso, mientras que un falso positivo solo cuesta una inspección.

**Lo que el modelo todavía no hace bien.** Con precisión de 0.17, de cada 6 máquinas marcadas solo 1 falla de verdad. Sirve para **priorizar inspecciones** —el uso previsto declarado en `CONTEXT.md`— pero no para decisiones automáticas. Y accuracy no debe leerse sola: un modelo que dijera siempre "no falla" acertaría el 91 % sin detectar una sola falla.

---

## 8. Limitaciones conocidas

- Los 17 atípicos de `vibration_mm_s` con 0 % de fallas podrían ser ruido de sensor, pero la muestra es demasiado pequeña para decidirlo. Se conservaron.
- La precisión de 0.17 limita el uso a priorización, no a automatización.
- El umbral de decisión se mantiene en 0.50 sobre `predict_proba`, según `CONTEXT.md`. Moverlo cambiaría el balance precisión/recall y requiere autorización.
- El conjunto de datos es sintético y preparado para el laboratorio: los resultados no se transfieren a maquinaria real sin validación con datos propios.
