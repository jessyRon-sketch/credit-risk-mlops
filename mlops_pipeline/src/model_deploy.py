# ==========================================
# Credit Risk Model Deployment
# ==========================================

from pathlib import Path
from typing import List
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ==========================================
# Crear aplicación
# ==========================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API para predicción de riesgo crediticio",
    version="1.1.1"
)

# ==========================================
# Cargar modelo y preprocesador
# ==========================================

ruta_proyecto = Path(__file__).resolve().parents[1]

modelo = joblib.load(
    ruta_proyecto / "best_model_v1.1.0.joblib"
)

preprocessor = joblib.load(
    ruta_proyecto / "preprocessor_v1.1.0.joblib"
)

# ==========================================
# Modelo de entrada
# ==========================================

class Cliente(BaseModel):

    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tipo_credito: str
    tipo_laboral: str
    tendencia_ingresos: str


# ==========================================
# Endpoint de prueba
# ==========================================

@app.get("/")
def home():

    return {
        "message": "La API de predicción de riesgo crediticio se encuentra en ejecución."
    }

# ==========================================
# Endpoint de predicción
# ==========================================

@app.post("/predict")
def predict(clientes: List[Cliente]):

    # Convertir la lista de clientes en DataFrame
    df = pd.DataFrame([cliente.model_dump() for cliente in clientes])

    # Aplicar el preprocesamiento
    X = preprocessor.transform(df)

    # Generar predicciones
    predicciones = modelo.predict(X)

    # Retornar resultados
    return {
        "predicciones": predicciones.tolist()
    }