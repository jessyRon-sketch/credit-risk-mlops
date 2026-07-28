# ==========================================
# Feature Engineering
# Bootcamp M5 - Credit Risk MLOps
# ==========================================

# ==========
# Librerías
# ==========

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# Función de limpieza de datos
# ==========================================

def limpiar_datos(df):
    """
    Aplica las reglas de limpieza e ingeniería de características
    al dataset.
    """

    # ==========================================
    # Eliminación de variables
    # ==========================================

    columnas_eliminar = [
        "saldo_mora",
        "saldo_total",
        "saldo_principal",
        "saldo_mora_codeudor",
        "puntaje",
        "fecha_prestamo"
    ]

    df = df.drop(columns=columnas_eliminar)

    # ==========================================
    # Limpieza de tendencia_ingresos
    # ==========================================

    valores_validos = {
        "Creciente",
        "Estable",
        "Decreciente"
    }

    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(valores_validos),
        np.nan
    )

    # ==========================================
    # Corrección de edades atípicas
    # ==========================================

    df.loc[df["edad_cliente"] > 100, "edad_cliente"] = np.nan

    # ==========================================
    # Conversión de variables categóricas
    # ==========================================

    df["tipo_credito"] = df["tipo_credito"].astype(str)

    return df

def preprocesar_datos():
    """
    Realiza el proceso de ingeniería de características y
    preprocesamiento del dataset.
    """

    # ==========================================
    # 1. Cargar datos
    # ==========================================

    # Ruta absoluta del directorio donde está este archivo
    ruta_actual = Path(__file__).resolve().parent

    # Subimos un nivel (mlops_pipeline) y buscamos el Excel
    ruta_datos = ruta_actual.parent / "Base_de_datos.xlsx"

    # Cargar dataset
    df = pd.read_excel(ruta_datos)

    # ==========================================
    # 2. Limpieza de datos
    # ==========================================

    df = limpiar_datos(df)

    # ==========================================
    # 3. Separación de Features y Target
    # ==========================================

    X = df.drop(columns="Pago_atiempo")
    y = df["Pago_atiempo"]

    # ==========================================
    # 4. Identificación de variables
    # ==========================================

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    cat_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    # ==========================================
    # 5. Pipeline para variables numéricas
    # ==========================================

    num_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))])

    # ==========================================
    # 6. Pipeline para variables categóricas
    # ==========================================

    cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    # ==========================================
    # 7. ColumnTransformer
    # ==========================================

    preprocessor = ColumnTransformer(
    transformers=[
        ("numericas", num_transformer, num_features),
        ("categoricas", cat_transformer, cat_features)])

    # ==========================================
    # 8. División Train/Test
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y)

    # ==========================================
    # 9. Aplicar el preprocesamiento
    # ==========================================

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return (
        X_train_preprocessed,
        X_test_preprocessed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, preprocessor = preprocesar_datos()

    print("========================================")
    print("Preprocesamiento completado correctamente")
    print("========================================")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")