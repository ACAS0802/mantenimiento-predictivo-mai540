"""
Pipeline de preprocesamiento — Tarea 2.2 (MAI 540).

El orden de las transformaciones es lo que evita la fuga de informacion:

  1. Cargar datos crudos.
  2. Limpieza SIN estadisticas (duplicados exactos, tipos, reglas fisicas).
     Puede ir antes del split porque no aprende nada del conjunto.
     De hecho DEBE ir antes: si una fila duplicada queda en train y su copia
     en test, el modelo ya vio la respuesta -> fuga.
  3. train_test_split estratificado.  <-- frontera. Nada de lo de abajo
     puede mirar el conjunto de prueba.
  4. Imputacion, escalamiento y codificacion DENTRO de un Pipeline que se
     ajusta (fit) solo con train y se aplica (transform) a test.
  5. Seleccion de caracteristicas decidida con evidencia medida solo en train.
  6. Evaluacion en test, una sola vez.

Autora: Araceli Castillo · MAI 540: Machine Learning · Atlantis University
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OBJETIVO = "failure_next_7_days"

# Variables numericas autorizadas (CONTEXT.md 2.1)
NUMERICAS = [
    "operating_hours",
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "days_since_maintenance",
    "error_count_30d",
    "energy_kw",
]

# Variables categoricas autorizadas (CONTEXT.md 2.1)
CATEGORICAS = ["machine_type", "shift", "environment"]

# Descartada en la Tarea 2.2 por evidencia (ver README.md, seccion 5).
# `shift` es la unica variable cuya eliminacion supera el ruido de medicion
# (p = 0.0093 en 30 mediciones pareadas de validacion cruzada sobre train).
DESCARTADAS = ["shift"]

CATEGORICAS_FINALES = [c for c in CATEGORICAS if c not in DESCARTADAS]
NUMERICAS_FINALES = [c for c in NUMERICAS if c not in DESCARTADAS]

SEMILLA = 42
TEST_SIZE = 0.25


def cargar_datos(ruta="data/datos.csv"):
    """Paso 1: carga cruda, sin transformar nada."""
    return pd.read_csv(Path(ruta))


def limpiar_sin_estadisticas(df):
    """
    Paso 2: unica limpieza permitida ANTES del split.

    Solo operaciones que no aprenden ningun parametro del conjunto:
    no usa medias, medianas, cuantiles ni correlaciones. Por eso es
    indiferente que se aplique sobre el conjunto completo.

    Devuelve (df_limpio, reporte).
    """
    reporte = {"filas_iniciales": len(df)}

    # 2a. Duplicados exactos. Se quitan ANTES del split a proposito:
    # una fila repetida a ambos lados de la particion es fuga directa.
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    reporte["duplicados_exactos_eliminados"] = dups

    # 2b. Reglas fisicas del dominio: ninguna de estas magnitudes puede
    # ser negativa. Es conocimiento del dominio, no una estadistica.
    imposibles = 0
    for col in NUMERICAS:
        if col in df.columns:
            malos = (df[col] < 0).sum()
            imposibles += int(malos)
            if malos:
                df.loc[df[col] < 0, col] = pd.NA  # pasa a faltante, se imputa despues

    reporte["valores_imposibles_a_faltante"] = imposibles
    reporte["filas_finales"] = len(df)
    return df, reporte


def separar(df, numericas=None, categoricas=None):
    """Paso 3: la frontera. Todo lo que sigue se ajusta solo con train."""
    numericas = NUMERICAS_FINALES if numericas is None else numericas
    categoricas = CATEGORICAS_FINALES if categoricas is None else categoricas
    X = df[numericas + categoricas]
    y = df[OBJETIVO].astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y
    )


def construir_preprocesador(numericas=None, categoricas=None):
    """
    Paso 4: imputacion + escalamiento + codificacion, todo dentro del
    ColumnTransformer para que `fit` solo pueda ver lo que se le pase.

    Numericas  -> SimpleImputer(median) + StandardScaler
    Categoricas-> SimpleImputer(most_frequent) + OneHotEncoder

    La mediana se eligio tras comparar tres estrategias por validacion
    cruzada sobre train (ver README.md, seccion 3).
    StandardScaler es obligatorio porque el algoritmo final es
    LogisticRegression, que es sensible a la escala de las variables.
    """
    numericas = NUMERICAS_FINALES if numericas is None else numericas
    categoricas = CATEGORICAS_FINALES if categoricas is None else categoricas

    rama_numerica = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
    ])
    rama_categorica = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numericas", rama_numerica, numericas),
        ("categoricas", rama_categorica, categoricas),
    ])


def construir_pipeline(numericas=None, categoricas=None):
    """Preprocesamiento + modelo en un solo objeto: imposible ajustar por separado."""
    return Pipeline([
        ("preprocesamiento", construir_preprocesador(numericas, categoricas)),
        ("modelo", LogisticRegression(
            max_iter=1000, random_state=SEMILLA, class_weight="balanced"
        )),
    ])
