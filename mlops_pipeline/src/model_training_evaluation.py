# ==========================================
# Model Training & Evaluation
# Bootcamp M5 - Credit Risk MLOps
# ==========================================

# ==========================================
# Librerías
# ==========================================

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from ft_engineering import preprocesar_datos

# ==========================================
# Cargar datos preprocesados
# ==========================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
) = preprocesar_datos()

# ==========================================
# Definición de modelos
# ==========================================

modelos = {

    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_estimators=200
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

}

# ==========================================
# Función para entrenar y evaluar modelos
# ==========================================

def build_model(modelos, X_train, X_test, y_train, y_test):

    resultados = {}

    for nombre, modelo in modelos.items():

        # Entrenar
        modelo.fit(X_train, y_train)

        # Predicciones
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        # Métricas
        resultados[nombre] = {
            "modelo": modelo,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
        }

    return resultados

# ==========================================
# Función para resumir resultados
# ==========================================

def summarize_classification(resultados):

    metricas = []

    for nombre, resultado in resultados.items():

        metricas.append({
            "Modelo": nombre,
            "Accuracy": resultado["accuracy"],
            "Precision": resultado["precision"],
            "Recall": resultado["recall"],
            "F1": resultado["f1"],
            "ROC_AUC": resultado["auc"],
        })

    df_metricas = pd.DataFrame(metricas)
    df_metricas = df_metricas.sort_values(
        by="ROC_AUC",
        ascending=False
    )

    return df_metricas

# ==========================================
# Ejecución principal
# ==========================================

if __name__ == "__main__":

    resultados = build_model(
        modelos,
        X_train,
        X_test,
        y_train,
        y_test
    )

    df_metricas = summarize_classification(resultados)

    print("\n========================================")
    print("Resultados de los modelos")
    print("========================================\n")

    print(df_metricas.round(4))

    # ==========================================
    # Gráfico comparativo de modelos
    # ==========================================

    metricas_grafico = df_metricas.set_index("Modelo")

    metricas_grafico.plot(
    kind="bar",
    figsize=(10, 6)
    )

    plt.title("Comparación de desempeño de modelos")
    plt.ylabel("Valor de la métrica")
    plt.xlabel("Modelo")
    plt.xticks(rotation=0)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Guardar imagen para el README
    ruta_imagen = Path(__file__).resolve().parents[2] / "images" / "model_comparison.png"

    plt.tight_layout()

    plt.savefig(ruta_imagen, dpi=300, bbox_inches="tight")

    plt.show()

    # ==========================================
    # Selección del mejor modelo
    # ==========================================

    mejor_modelo_nombre = df_metricas.iloc[0]["Modelo"]

    mejor_modelo = resultados[mejor_modelo_nombre]["modelo"]

    print("\n========================================")
    print(f"Mejor modelo: {mejor_modelo_nombre}")
    print("========================================")

    # ==========================================
    # Guardar mejor modelo y preprocesador
    # ==========================================

    ruta_proyecto = Path(__file__).resolve().parent.parent

    ruta_modelo = ruta_proyecto / "best_model_v1.1.0.joblib"
    ruta_preprocessor = ruta_proyecto / "preprocessor_v1.1.0.joblib"

    joblib.dump(mejor_modelo, ruta_modelo)
    joblib.dump(preprocessor, ruta_preprocessor)

    print("\n========================================")
    print("Modelo y preprocesador guardados")
    print("========================================")
    print(f"Modelo      : {ruta_modelo.name}")
    print(f"Preprocessor: {ruta_preprocessor.name}")