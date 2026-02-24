# MLOps Final Project: Credit Card Fraud Detection API

**Estudiante:** Ricardo Bautista  
**Curso:** MLOps - Maestría en Data Science (UNI)  
**Ciclo:** 3 (2026)

---

## 📌 1. Descripción del Problema (Problem Definition)
El fraude con tarjetas de crédito es un problema crítico en el sector bancario. El objetivo de este proyecto es desarrollar una solución *end-to-end* basada en Machine Learning para detectar transacciones fraudulentas en tiempo real.

Dada la naturaleza desbalanceada de los datos (muy pocos fraudes comparados con transacciones normales) y la falta de etiquetas en escenarios reales, se ha optado por un enfoque de **Detección de Anomalías** utilizando el algoritmo **Isolation Forest**.

### Objetivos:
* Implementar un pipeline de entrenamiento reproducible.
* Serializar el modelo para su persistencia.
* Desplegar el modelo como una **API REST** utilizando **FastAPI**.
* Seguir las mejores prácticas de MLOps (estructura modular, control de versiones).

---

## 📂 2. Estructura del Proyecto
El proyecto sigue una estructura modular estándar para MLOps:

```text
uni_mds_ciclo3_ml_project/
├── data/
│   ├── raw/                # Dataset original (creditcard.csv)
│   └── processed/          # Datos procesados (si aplica)
├── models/                 # Modelos serializados (.pkl)
├── notebooks/              # Experimentos y EDA
├── src/                    # Código fuente
│   ├── train.py            # Script de entrenamiento
│   └── serving.py          # API para inferencia (FastAPI)
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación principal

```

---

## ⚙️ 3. Instalación y Configuración

1. **Clonar el repositorio:**
```bash
git clone [https://github.com/Ricardo-Bautista-H/uni_mds_ciclo3_ml_project.git](https://github.com/Ricardo-Bautista-H/uni_mds_ciclo3_ml_project.git)
cd uni_mds_ciclo3_ml_project

```


2. **Crear un entorno virtual (Recomendado):**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

```


3. **Instalar dependencias:**
```bash
pip install -r requirements.txt

```


4. **Descargar los datos:**
Descarga el dataset `creditcard.csv` desde [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) y colócalo en la carpeta `data/raw/`.

---

## 🚀 4. Ejecución del Pipeline

### A. Entrenamiento del Modelo

Para entrenar el modelo Isolation Forest y generar el archivo `.pkl`:

```bash
python src/train.py

```

*Output esperado:* El script reportará las métricas de evaluación (Precision/Recall/F1) y guardará el modelo en `models/isolation_forest_fraud.pkl`.

### B. Despliegue de la API (Serving)

Para levantar el servidor de predicción localmente:

```bash
uvicorn src.serving:app --reload

```

La API estará disponible en: `http://127.0.0.1:8000`

---

## 📡 5. Uso de la API (Inferencia)

Puedes probar la API directamente desde la documentación interactiva (Swagger UI).


1. Ve a **http://127.0.0.1:8000/docs**
2. Busca el endpoint **POST /predict**.
3. Haz clic en **"Try it out"**.
4. Pega el siguiente JSON de prueba (una transacción normal):

```json
{
  "features": [0, -1.3598, -0.07278, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395, 0.0986, 0.3637, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104, 0.0669, 0.1285, -0.1891, 0.1335, -0.021, 149.62]
}

```

### Respuesta Esperada:


{
  "is_fraud": false,
  "anomaly_score": 0.175,
  "alert": "Normal"
}



---

## 📊 6. Resultados y Evidencia

*Captura de la API funcionando correctamente:*

![Evidencia de API](resources/images/api_prediction_success_1.png)

![Evidencia de API](resources/images/api_prediction_success_2.png)

![Evidencia de API](resources/images/api_prediction_success_3.png)

---