# Usar una imagen oficial de Python ligera
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requerimientos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del proyecto
COPY . .

# Exponer el puerto que usará FastAPI
EXPOSE 8000

# Comando para arrancar la API
CMD ["uvicorn", "src.serving:app", "--host", "0.0.0.0", "--port", "8000"]