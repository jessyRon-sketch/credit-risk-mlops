# ==============================
# Credit Risk Monitoring 
# ==============================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from ft_engineering import limpiar_datos
from model_monitoring import ejecutar_monitoreo


# ==========================================
# Ejecutar monitoreo
# ==========================================

df_ks, df_psi, df_js, df_chi2 = ejecutar_monitoreo()

# ==========================================
# Cargar bases para visualización
# ==========================================

ruta_proyecto = Path(__file__).resolve().parents[1]

df_historico = pd.read_excel(
    ruta_proyecto / "Base_de_datos.xlsx"
)

df_simulado = pd.read_excel(
    ruta_proyecto / "Base_de_datos_simulada.xlsx"
)

# ==========================================
# Aplicar limpieza de datos
# ==========================================

df_historico = limpiar_datos(df_historico)
df_simulado = limpiar_datos(df_simulado)


st.set_page_config(
    page_title="Credit Risk Monitoring",
    layout="wide"
)

st.title("Credit Risk Model Monitoring Dashboard")

st.markdown("---")

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("Navigation")

opcion = st.sidebar.radio(

    "Select an option",

    [
        "Data Drift Metrics",
        "Distribution Analysis",
        "Model Performance",
        "Recommendations"
    ]

)

# ==========================================
# Dashboard Body
# ==========================================

if opcion == "Data Drift Metrics":

    st.header("Data Drift Metrics")

    st.subheader("Kolmogorov-Smirnov Test")
    st.dataframe(df_ks.round(4))

    st.subheader("Population Stability Index (PSI)")
    st.dataframe(df_psi.round(4))

    st.subheader("Jensen-Shannon Distance")
    st.dataframe(df_js.round(4))

    st.subheader("Chi-Square Test")
    st.dataframe(df_chi2.round(4))

elif opcion == "Distribution Analysis":

    st.header("Análisis de Distribuciones")

    st.write(
    "Distribución de variables numéricas entre base historica y simulada.")

    variable = st.selectbox(

    "Seleccione una variable",

    [
        "capital_prestado",
        "edad_cliente",
        "puntaje_datacredito"
    ])

    fig, ax = plt.subplots(figsize=(10,5))

    ax.hist(
        df_historico[variable],
        bins=30,
        alpha=0.6,
        label="Histórico"
    )

    ax.hist(
        df_simulado[variable],
        bins=30,
        alpha=0.6,
        label="Actual"
    )

    ax.set_title(f"Comparación de Distribución. - {variable}")
    ax.set_xlabel(variable.replace("_", " ").title())
    ax.set_ylabel("Frecuencia")

    ax.legend()

    st.pyplot(fig)
    

elif opcion == "Model Performance":

    st.header("Desempeño del Modelo")
    st.write(
        "Resumen de las variables con mayor nivel de drift detectado durante el monitoreo."
    )

    # ==========================================
    # Variables con mayor drift
    # ==========================================

    variable_psi = df_psi.loc[df_psi["PSI"].idxmax()]
    variable_ks = df_ks.loc[df_ks["KS Statistic"].idxmax()]
    variable_js = df_js.loc[df_js["JS Distance"].idxmax()]

    variables_criticas = (df_psi["PSI"] > 0.25).sum()

    # ==========================================
    # KPIs
    # ==========================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Mayor PSI",
        f'{variable_psi["PSI"]:.2f}',
        variable_psi["Variable"]
    )

    col2.metric(
        "Mayor KS",
        f'{variable_ks["KS Statistic"]:.2f}',
        variable_ks["Variable"]
    )

    col3.metric(
        "Mayor JS",
        f'{variable_js["JS Distance"]:.2f}',
        variable_js["Variable"]
    )

    col4.metric(
        "Variables críticas",
        variables_criticas
    )

    st.markdown("---")

    st.subheader("Resumen")

    st.write(
        f"""
Se identificaron **{variables_criticas} variables** con un **PSI superior a 0.25**, lo que indica cambios importantes respecto a la distribución histórica.

La variable con mayor nivel de drift fue **{variable_psi["Variable"]}**, por lo que debería ser priorizada para una revisión del modelo y del comportamiento de los datos.
"""
    )

elif opcion == "Recommendations":

    st.header("Recomendaciones")

    variables_criticas = df_psi[df_psi["PSI"] >= 0.25]

    st.subheader("Estado del monitoreo")

    if len(variables_criticas) == 0:

        st.success(
            "El modelo presenta estabilidad. No se detectaron variables con drift crítico."
        )

    else:

        st.warning(
            f"Se detectaron {len(variables_criticas)} variables con drift significativo."
        )

        st.subheader("Variables que requieren atención")

        st.dataframe(
            variables_criticas.round(4)
        )

        st.subheader("Recomendaciones")

        st.markdown("""
    - Revisar las variables con mayor nivel de drift.
    - Validar la calidad de los datos recibidos.
    - Monitorear el comportamiento del modelo en futuras ejecuciones.
    - Considerar el reentrenamiento del modelo si el drift persiste.
    """)