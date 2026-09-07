# Plan de mejora — `main.py` (alcance exacto del README)

## Objetivo
Reescribir `main.py` para pasar del modelo limitado (4 numéricas, solo accuracy, sin
balanceo) al pipeline completo con 10 predictores, preprocesamiento por tipo de
variable, `LogisticRegression` balanceada y métricas de la clase `1`, mostrando
comparación ANTES vs DESPUÉS.

## Evidencia del ANTES (ya ejecutada)
```
=== MANTENIMIENTO PREDICTIVO: PUNTO DE PARTIDA ===
Filas: 5,000
Variables usadas por el modelo inicial: 4
Accuracy: 0.9112
Matriz de confusión:
[[1139    0]
 [ 111    0]]
```
El modelo predice siempre `0`: recall de la clase 1 = 0, F1 = 0. La accuracy alta solo
refleja la proporción de la clase mayoritaria.

## Paso 1 — Imports adicionales
Añadir a los ya presentes:
- `from sklearn.preprocessing import StandardScaler, OneHotEncoder`
- `from sklearn.compose import ColumnTransformer`
- `from sklearn.metrics import precision_score, recall_score, f1_score, classification_report`

## Paso 2 — Definición de columnas
```
NUMERICAS   = ['operating_hours','temperature_c','vibration_mm_s','pressure_bar',
               'days_since_maintenance','error_count_30d','energy_kw']
CATEGORICAS = ['machine_type','shift','environment']
FEATURES    = NUMERICAS + CATEGORICAS          # 10 predictores
```
`X = df[FEATURES]`, `y = df['failure_next_7_days'].astype(int)`.

## Paso 3 — Split
Mismo que el ANTES, sin cambios:
`train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)`
Así el conjunto de prueba es idéntico fila a fila y la comparación es justa.

## Paso 4 — ColumnTransformer (dos ramas)
- **Rama numérica** (`Pipeline`): `SimpleImputer(strategy="median")` -> `StandardScaler()`
- **Rama categórica** (`Pipeline`): `SimpleImputer(strategy="most_frequent")` -> `OneHotEncoder(handle_unknown="ignore")`
- Combinar con `ColumnTransformer([...])` aplicando cada rama a su lista de columnas.

## Paso 5 — Pipeline final
```
Pipeline([
    ("prep", preprocesador),
    ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])
```
`model.fit(X_train, y_train)` — el preprocesamiento se ajusta **solo con train** (lo
garantiza el propio Pipeline). Sin tuning, sin CV, sin probar otros algoritmos.

## Paso 6 — Evaluación
Sobre `X_test`:
- `accuracy_score`
- `precision_score`, `recall_score`, `f1_score` con `pos_label=1` (o `classification_report` y leer la fila de la clase `1`)
- `confusion_matrix`

## Paso 7 — Comparación ANTES vs DESPUÉS
Para no perder el ANTES, dentro del alcance:
- Envolver el modelo base (4 numéricas, sin balanceo, `SimpleImputer` mediana) y el
  modelo nuevo en dos bloques dentro del mismo script, con el mismo split.
- Imprimir una tabla comparativa: `accuracy | precision(1) | recall(1) | F1(1)` para
  ambos, más ambas matrices de confusión.
- Alternativa aceptable: dejar la salida del ANTES ya guardada como evidencia y que
  `main.py` imprima solo el DESPUÉS citando los números del ANTES en comentarios.
  Se recomienda la tabla en el script porque el README pide "comparación clara".

## Paso 8 — Texto explicativo (en la salida y/o `BITACORA.md`)
Redactar con palabras propias:
- **Tipos de variable**: por qué numéricas se imputan con mediana + se escalan; por qué
  categóricas se imputan con moda + one-hot; qué hace `handle_unknown="ignore"`.
- **Fuga de datos**: por qué `fit` solo con train.
- **Desbalanceo**: por qué accuracy sola no basta (el ANTES tiene 0.91 y recall 0 en
  clase 1); qué corrige `class_weight="balanced"`; interpretar recall (fallas
  detectadas) y F1 (equilibrio) de la clase 1.

## Entregables afectados
- `main.py` reescrito.
- Evidencia de ejecución DESPUÉS (guardar salida).
- `BITACORA.md` completado con la explicación propia.
- `data/DESCRIPCION.md` dice 1.800 registros pero el CSV tiene 5.000 — **fuera de
  alcance**, solo se menciona; no se toca salvo indicación expresa.

## Fuera de alcance (no se hará)
Selección automática de variables, validación cruzada, búsqueda de hiperparámetros,
feature engineering avanzado, comparación de múltiples algoritmos.
