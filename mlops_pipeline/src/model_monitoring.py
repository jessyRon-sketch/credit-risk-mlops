# ==========================================
# Model Monitoring
# Bootcamp M5 - Credit Risk MLOps
# ==========================================

# ==========================================
# Librerías
# ==========================================

import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from ft_engineering import limpiar_datos

def cargar_recursos():
    """
    Carga los modelos y datasets necesarios para el monitoreo.
    """

    # Ruta donde está este archivo
    ruta_actual = Path(__file__).resolve().parent

    # Ruta principal del proyecto (mlops_pipeline)
    ruta_proyecto = ruta_actual.parent

    # ===============================
    # Cargar modelo
    # ===============================

    modelo = joblib.load(
        ruta_proyecto / "best_model_v1.1.0.joblib"
    )

    # ===============================
    # Cargar preprocesador
    # ===============================

    preprocessor = joblib.load(
        ruta_proyecto / "preprocessor_v1.1.0.joblib"
    )

    # ===============================
    # Cargar datasets
    # ===============================

    df_historico = pd.read_excel(
        ruta_proyecto / "Base_de_datos.xlsx"
    )

    df_simulado = pd.read_excel(
        ruta_proyecto / "Base_de_datos_simulada.xlsx"
    )

    return (
        modelo,
        preprocessor,
        df_historico,
        df_simulado,
    )

# ==========================================
# Kolmogorov-Smirnov Test
# ==========================================

def calcular_ks(df_historico, df_simulado, variables_numericas):
    """
    Calcula el estadístico de Kolmogorov-Smirnov para detectar Data Drift
    en variables numéricas.
    """

    resultados_ks = []

    for variable in variables_numericas:

        historico = df_historico[variable].dropna()
        simulado = df_simulado[variable].dropna()

        estadistico, p_value = ks_2samp(
           historico,
           simulado
        )
        

        resultados_ks.append({

            "Variable": variable,
            "KS Statistic": estadistico,
            "P-Value": p_value

        })

    return pd.DataFrame(resultados_ks)

# ==========================================
# Population Stability Index (PSI)
# ==========================================

def calcular_psi(df_historico, df_simulado, variables_numericas, bins=10):
    """
    Calcula el Population Stability Index (PSI)
    para detectar Data Drift en variables numéricas.
    """

    resultados_psi = []

    for variable in variables_numericas:

        # Eliminar valores nulos
        historico = df_historico[variable].dropna()
        simulado = df_simulado[variable].dropna()

        # Si alguna variable quedó vacía, continuar
        if historico.empty or simulado.empty:
            resultados_psi.append({
                "Variable": variable,
                "PSI": np.nan
            })
            continue

        # Crear bins usando la distribución histórica
        cortes = np.histogram_bin_edges(historico, bins=bins)

        # Distribuciones
        hist_hist, _ = np.histogram(historico, bins=cortes)
        hist_sim, _ = np.histogram(simulado, bins=cortes)

        # Convertir a porcentajes
        dist_hist = hist_hist / len(historico)
        dist_sim = hist_sim / len(simulado)

        # Evitar división por cero
        dist_hist = np.where(dist_hist == 0, 0.0001, dist_hist)
        dist_sim = np.where(dist_sim == 0, 0.0001, dist_sim)

        # Calcular PSI
        psi = np.sum(
            (dist_hist - dist_sim) *
            np.log(dist_hist / dist_sim)
        )

        resultados_psi.append({
            "Variable": variable,
            "PSI": psi
        })

    return pd.DataFrame(resultados_psi)

# ==========================================
# Jensen-Shannon Divergence
# ==========================================

def calcular_js(df_historico, df_simulado, variables_numericas, bins=10):
    """
    Calcula la distancia de Jensen-Shannon para detectar Data Drift
    en variables numéricas.
    """

    resultados_js = []

    for variable in variables_numericas:

        historico = df_historico[variable].dropna()
        simulado = df_simulado[variable].dropna()

        if historico.empty or simulado.empty:

            resultados_js.append({
                "Variable": variable,
                "JS Distance": np.nan
            })

            continue

        cortes = np.histogram_bin_edges(
            historico,
            bins=bins
        )

        hist_hist, _ = np.histogram(
            historico,
            bins=cortes
        )

        hist_sim, _ = np.histogram(
            simulado,
            bins=cortes
        )

        dist_hist = hist_hist / len(historico)
        dist_sim = hist_sim / len(simulado)

        dist_hist = np.where(dist_hist == 0, 0.0001, dist_hist)
        dist_sim = np.where(dist_sim == 0, 0.0001, dist_sim)

        js = jensenshannon(
            dist_hist,
            dist_sim
        )

        resultados_js.append({

            "Variable": variable,
            "JS Distance": js

        })

    return pd.DataFrame(resultados_js)

# ==========================================
# Chi-Cuadrado para variables categóricas
# ==========================================

def calcular_chi2(df_historico, df_simulado, variables_categoricas):
    """
    Calcula la prueba Chi-cuadrado para detectar
    Data Drift en variables categóricas.
    """

    resultados_chi2 = []

    for variable in variables_categoricas:

        # Frecuencias de ambas bases
        freq_hist = df_historico[variable].value_counts()
        freq_sim = df_simulado[variable].value_counts()

        # Unificar categorías
        tabla = pd.concat(
            [freq_hist, freq_sim],
            axis=1,
            keys=["Historico", "Simulado"]
        ).fillna(0)

        chi2, p_value, _, _ = chi2_contingency(tabla)

        resultados_chi2.append({

            "Variable": variable,
            "Chi2": chi2,
            "P-Value": p_value

        })

    return pd.DataFrame(resultados_chi2)


if __name__ == "__main__":

    (
        modelo,
        preprocessor,
        df_historico,
        df_simulado,
    ) = cargar_recursos()

    # ==========================================
    # Limpieza de las bases
    # ==========================================

    df_historico = limpiar_datos(df_historico)
    df_simulado = limpiar_datos(df_simulado)

    print("======================================")
    print("Recursos cargados y bases procesadas")
    print("======================================")
    print(f"Histórico : {df_historico.shape}")
    print(f"Simulado  : {df_simulado.shape}")

    # ==========================================
    # Generar predicciones sobre la base simulada
    # ==========================================

    # Separar variables independientes y objetivo
    X_simulado = df_simulado.drop(columns="Pago_atiempo")
    y_simulado = df_simulado["Pago_atiempo"]

    # Aplicar el mismo preprocesamiento utilizado en entrenamiento
    X_simulado_preprocessed = preprocessor.transform(X_simulado)

    # Generar predicciones
    predicciones = modelo.predict(X_simulado_preprocessed)

    # Agregar predicciones al dataset
    df_resultados = df_simulado.copy()
    df_resultados["Prediccion_Modelo"] = predicciones

    print("\n========================================")
    print("Predicciones generadas correctamente")
    print("========================================")

    print(df_resultados[
        ["Pago_atiempo", "Prediccion_Modelo"]
    ].head())

    # ==========================================
    # Identificar variables numéricas y categóricas
     # ==========================================

    variable_objetivo = "Pago_atiempo"

    variables_numericas = df_historico.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    variables_categoricas = df_historico.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # Eliminar la variable objetivo de las variables numéricas
    if variable_objetivo in variables_numericas:
        variables_numericas.remove(variable_objetivo)

    print("\n========================================")
    print("Variables identificadas")
    print("========================================")

    print(f"Variables numéricas : {len(variables_numericas)}")
    print(f"Variables categóricas: {len(variables_categoricas)}")

    # ==========================================
    # Calcular KS Test
    # ==========================================

    df_ks = calcular_ks(
    df_historico,
    df_simulado,
    variables_numericas
    )

    print("\n========================================")
    print("Kolmogorov-Smirnov Test")
    print("========================================")

    print(df_ks.round(4))

    # ==========================================
    # Calcular PSI
    # ==========================================

    df_psi = calcular_psi(
        df_historico,
        df_simulado,
        variables_numericas
    )

    print("\n========================================")
    print("Population Stability Index (PSI)")
    print("========================================")

    print(df_psi.round(4))

    # ==========================================
    # Calcular Jensen-Shannon
    # ==========================================

    df_js = calcular_js(
        df_historico,
        df_simulado,
        variables_numericas
    )

    print("\n========================================")
    print("Jensen-Shannon Distance")
    print("========================================")

    print(df_js.round(4))

    # ==========================================
    # Calcular Chi-Cuadrado
    # ==========================================

    df_chi2 = calcular_chi2(
        df_historico,
        df_simulado,
        variables_categoricas
    )

    print("\n========================================")
    print("Chi-Cuadrado")
    print("========================================")

    print(df_chi2.round(4))