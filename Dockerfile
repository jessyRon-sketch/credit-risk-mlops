# Imagen base
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Copiar requirements para Docker
COPY requirements_docker.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copiar el proyecto
COPY . .

# Exponer puerto
EXPOSE 8000

# Ejecutar la API
CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]