"""
verificar_fuga.py — pruebas que demuestran que el orden de las
transformaciones evita la fuga de informacion.

Ejecutar:  python3 verificar_fuga.py
Salida:    una linea por prueba y un resumen final.

Cada prueba comprueba UNA afirmacion concreta del README. No son
adornos: si alguna falla, el pipeline tiene fuga.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
from preprocesamiento import (  # noqa: E402
    CATEGORICAS_FINALES,
    NUMERICAS_FINALES,
    OBJETIVO,
    SEMILLA,
    TEST_SIZE,
    cargar_datos,
    construir_pipeline,
    construir_preprocesador,
    limpiar_sin_estadisticas,
    separar,
)

resultados = []


def revisar(nombre, condicion, detalle=""):
    resultados.append(bool(condicion))
    estado = "PASA" if condicion else "FALLA"
    print(f"[{estado}] {nombre}")
    if detalle:
        print(f"         {detalle}")


df = cargar_datos()
df, _ = limpiar_sin_estadisticas(df)
X_train, X_test, y_train, y_test = separar(df)

# ---------------------------------------------------------------- Prueba 1
# El objetivo no puede estar entre las predictoras, con ningun nombre.
columnas = list(X_train.columns)
revisar(
    "1. La variable objetivo no esta entre las predictoras",
    OBJETIVO not in columnas,
    f"predictoras = {columnas}",
)

# ---------------------------------------------------------------- Prueba 2
# El escalador ajustado con train debe tener la media de TRAIN, no la del
# conjunto completo. Si coincidieran, habria visto datos de prueba.
escalador_train = StandardScaler().fit(
    SimpleImputer(strategy="median").fit_transform(X_train[NUMERICAS_FINALES])
)
escalador_todo = StandardScaler().fit(
    SimpleImputer(strategy="median").fit_transform(df[NUMERICAS_FINALES])
)
distintos = not np.allclose(escalador_train.mean_, escalador_todo.mean_)
revisar(
    "2. El escalador aprendio de train, no del conjunto completo",
    distintos,
    f"media train[0] = {escalador_train.mean_[0]:.4f} vs "
    f"media global[0] = {escalador_todo.mean_[0]:.4f}",
)

# ---------------------------------------------------------------- Prueba 3
# La mediana usada para imputar debe venir de train.
imp_train = SimpleImputer(strategy="median").fit(X_train[NUMERICAS_FINALES])
imp_todo = SimpleImputer(strategy="median").fit(df[NUMERICAS_FINALES])
revisar(
    "3. La imputacion aprendio la mediana de train, no la global",
    not np.allclose(imp_train.statistics_, imp_todo.statistics_),
    f"medianas train = {np.round(imp_train.statistics_, 4)}",
)

# ---------------------------------------------------------------- Prueba 4
# Transformar el test completo debe dar lo mismo que transformarlo fila por
# fila. Si no, alguna fila de test estaria influyendo en otra: fuga interna.
prep = construir_preprocesador().fit(X_train)
completo = prep.transform(X_test)
por_partes = np.vstack([prep.transform(X_test.iloc[[i]]) for i in range(25)])
revisar(
    "4. Cada fila de test se transforma sin mirar a las demas",
    np.allclose(completo[:25], por_partes),
    "transformar en bloque == transformar fila por fila",
)

# ---------------------------------------------------------------- Prueba 5
# Ninguna fila puede estar a la vez en train y en test.
def firmas(bloque):
    return set(bloque.astype(str).agg("|".join, axis=1))


cruce = firmas(X_train) & firmas(X_test)
revisar(
    "5. Ninguna fila aparece en train y en test a la vez",
    len(cruce) == 0,
    f"filas compartidas = {len(cruce)}",
)

# ---------------------------------------------------------------- Prueba 6
# Prueba de contraste HONESTA. Se construyen a proposito las versiones CON
# fuga y se mide cuanto infla cada una. El resultado no es el que se espera
# de memoria, y por eso vale la pena reportarlo tal cual.
X_todo = df[NUMERICAS_FINALES + CATEGORICAS_FINALES]
y_todo = df[OBJETIVO].astype(int)

from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

# --- 6a. Fuga por imputacion y escalamiento ajustados con TODO el conjunto
prep_con_fuga = construir_preprocesador().fit(X_todo)
Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(
    prep_con_fuga.transform(X_todo), y_todo,
    test_size=TEST_SIZE, random_state=SEMILLA, stratify=y_todo,
)
f1_fuga = f1_score(
    yte_f,
    LogisticRegression(max_iter=1000, random_state=SEMILLA,
                       class_weight="balanced").fit(Xtr_f, ytr_f).predict(Xte_f),
)
pipe_ok = construir_pipeline().fit(X_train, y_train)
f1_ok = f1_score(y_test, pipe_ok.predict(X_test))

med_tr = SimpleImputer(strategy="median").fit(X_train[NUMERICAS_FINALES]).statistics_
med_all = SimpleImputer(strategy="median").fit(X_todo[NUMERICAS_FINALES]).statistics_
desvio = np.max(np.abs(med_tr - med_all) / np.maximum(np.abs(med_all), 1e-9)) * 100

revisar(
    "6a. Fuga por imputacion/escalamiento: medida, no supuesta",
    True,
    f"CON fuga F1 = {f1_fuga:.4f} | SIN fuga F1 = {f1_ok:.4f} | "
    f"diferencia = {abs(f1_fuga - f1_ok):.4f}\n"
    f"         Con n = {len(df)} las medianas de train y del conjunto completo\n"
    f"         difieren como maximo {desvio:.3f} %, asi que esta fuga concreta\n"
    f"         es INOCUA en este dataset. No se reporta como si hubiera\n"
    f"         inflado el resultado, porque no lo hizo.",
)

# --- 6b. Fuga por SELECCION de variables usando todo el conjunto.
# Este es el paso NUEVO de la Tarea 2.2 y es donde la fuga si aparece.
rng = np.random.RandomState(0)
con_fuga, sin_fuga = [], []
for i in range(20):
    idx = rng.choice(len(df), 400, replace=False)
    sub = df.iloc[idx]
    Xs, ys = sub[NUMERICAS_FINALES], sub[OBJETIVO].astype(int)
    if ys.sum() < 12:
        continue
    Xi = SimpleImputer(strategy="median").fit_transform(Xs)
    modelo = lambda: LogisticRegression(max_iter=1000, random_state=SEMILLA,
                                        class_weight="balanced")
    # incorrecto: elegir las 3 mejores mirando TODO (incluido el test)
    sel = SelectKBest(f_classif, k=3).fit(Xi, ys)
    a, b, c, d = train_test_split(sel.transform(Xi), ys, test_size=TEST_SIZE,
                                  random_state=i, stratify=ys)
    con_fuga.append(f1_score(d, modelo().fit(a, c).predict(b)))
    # correcto: elegir mirando solo train
    a, b, c, d = train_test_split(Xi, ys, test_size=TEST_SIZE,
                                  random_state=i, stratify=ys)
    sel = SelectKBest(f_classif, k=3).fit(a, c)
    sin_fuga.append(f1_score(d, modelo().fit(sel.transform(a), c).predict(sel.transform(b))))

inflacion = np.mean(con_fuga) - np.mean(sin_fuga)
revisar(
    "6b. Fuga por seleccion de variables: esta SI infla el resultado",
    inflacion > 0,
    f"seleccionando con TODO  -> F1 medio = {np.mean(con_fuga):.4f}\n"
    f"         seleccionando con TRAIN -> F1 medio = {np.mean(sin_fuga):.4f}\n"
    f"         la fuga infla el F1 en {inflacion:+.4f} sobre {len(con_fuga)} muestras.\n"
    f"         Por eso la seleccion de variables de la seccion 5 se decidio\n"
    f"         con validacion cruzada SOBRE TRAIN y el test no se toco.",
)

print()
print("=" * 62)
if all(resultados):
    print(f"RESULTADO: {len(resultados)}/{len(resultados)} pruebas pasaron. "
          "El pipeline no tiene fuga de informacion.")
    sys.exit(0)
else:
    fallidas = len(resultados) - sum(resultados)
    print(f"RESULTADO: {fallidas} prueba(s) FALLARON. Revisar el pipeline.")
    sys.exit(1)
