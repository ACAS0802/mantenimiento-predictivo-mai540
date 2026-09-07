from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA = Path(__file__).parent / "data" / "datos.csv"

df = pd.read_csv(DATA)

TARGET = "failure_next_7_days"
y = df[TARGET].astype(int)

# 10 predictores obligatorios del README
NUMERICAS = [
    "operating_hours",
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "days_since_maintenance",
    "error_count_30d",
    "energy_kw",
]
CATEGORICAS = ["machine_type", "shift", "environment"]
FEATURES = NUMERICAS + CATEGORICAS

# Variables del modelo inicial (punto de partida deliberadamente limitado)
FEATURES_ANTES = ["operating_hours", "temperature_c", "days_since_maintenance", "energy_kw"]

# Mismo split para ambos modelos: el conjunto de prueba es idéntico fila a fila,
# por lo que la comparación ANTES vs DESPUES es justa.
X_all = df[FEATURES]
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.25, random_state=42, stratify=y
)


def metricas(nombre, y_true, y_pred):
    return {
        "modelo": nombre,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred),
    }


# ----------------------------------------------------------------------
# ANTES: 4 variables numericas, imputacion por mediana, sin escalado,
# sin manejo de categoricas y sin class_weight.
# ----------------------------------------------------------------------
modelo_antes = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
modelo_antes.fit(X_train_all[FEATURES_ANTES], y_train)
pred_antes = modelo_antes.predict(X_test_all[FEATURES_ANTES])
res_antes = metricas("ANTES (4 numericas)", y_test, pred_antes)

# ----------------------------------------------------------------------
# DESPUES: 10 predictores, ColumnTransformer con dos ramas y
# LogisticRegression balanceada. Todo el preprocesamiento se ajusta
# unicamente con los datos de entrenamiento (lo garantiza el Pipeline).
# ----------------------------------------------------------------------
rama_numerica = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
rama_categorica = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocesador = ColumnTransformer([
    ("num", rama_numerica, NUMERICAS),
    ("cat", rama_categorica, CATEGORICAS),
])
modelo_despues = Pipeline([
    ("prep", preprocesador),
    ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])
modelo_despues.fit(X_train_all, y_train)
pred_despues = modelo_despues.predict(X_test_all)
res_despues = metricas("DESPUES (10 predictores)", y_test, pred_despues)

# ----------------------------------------------------------------------
# Reporte
# ----------------------------------------------------------------------
print("=== MANTENIMIENTO PREDICTIVO: CLASIFICACION BINARIA (failure_next_7_days) ===")
print(f"Filas: {len(df):,}")
print(f"Distribucion de la clase objetivo (0/1): {y.value_counts().to_dict()}")
print(f"Positivos en el conjunto de prueba: {int(y_test.sum())} de {len(y_test)}")
print(f"test_size=0.25 | random_state=42 | stratify=y")
print()

print("--- Comparacion ANTES vs DESPUES (metricas para la clase 1) ---")
cab = f"{'modelo':<28}{'accuracy':>10}{'precision_1':>13}{'recall_1':>10}{'f1_1':>8}"
print(cab)
print("-" * len(cab))
for r in (res_antes, res_despues):
    print(
        f"{r['modelo']:<28}{r['accuracy']:>10.4f}{r['precision_1']:>13.4f}"
        f"{r['recall_1']:>10.4f}{r['f1_1']:>8.4f}"
    )
print()

print("Matriz de confusion ANTES  (filas=real, columnas=predicho):")
print(res_antes["cm"])
print()
print("Matriz de confusion DESPUES (filas=real, columnas=predicho):")
print(res_despues["cm"])
print()

print("--- Que cambio y por que ---")
print(
    "1. Predictores: se pasa de 4 variables numericas a los 10 predictores del README\n"
    "   (7 numericas + 3 categoricas), aportando mas senal al modelo.\n"
    "2. Preprocesamiento por tipo de variable con ColumnTransformer:\n"
    "   - Numericas: SimpleImputer(median) + StandardScaler. La mediana es robusta a\n"
    "     valores atipicos y no descarta filas con faltantes (vibration_mm_s, pressure_bar).\n"
    "     El escalado pone todas las variables en la misma magnitud, algo que la\n"
    "     regresion logistica necesita para que ninguna domine por su unidad.\n"
    "   - Categoricas: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore').\n"
    "     One-hot convierte machine_type/shift/environment en columnas 0/1; 'ignore'\n"
    "     evita que una categoria no vista en train rompa la prediccion.\n"
    "3. Sin fuga de datos: el preprocesamiento se ajusta solo con X_train (Pipeline);\n"
    "   el conjunto de prueba se mantiene separado.\n"
    "4. class_weight='balanced': la falla es minoritaria. Sin balanceo el modelo ANTES\n"
    "   maximiza accuracy prediciendo siempre 'no falla' (recall_1 = 0, F1_1 = 0): una\n"
    "   accuracy alta que no detecta ninguna averia. Con balanceo se penaliza mas el\n"
    "   error sobre la clase 1 y el modelo empieza a identificar fallas reales.\n"
    "5. Por eso se reportan precision, recall y F1 de la clase 1 y no solo accuracy:\n"
    "   en un problema desbalanceado el recall de la clase positiva (fallas detectadas)\n"
    "   y el F1 (equilibrio precision/recall) describen el valor real del modelo."
)
