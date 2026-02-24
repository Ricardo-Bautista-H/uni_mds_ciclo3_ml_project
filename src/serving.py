from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Fraud Detection API", version="1.0")

# Cargar modelo al inicio
model_path = 'models/isolation_forest_fraud.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print("ADVERTENCIA: No se encontró el modelo. Ejecuta train.py primero.")

class TransactionInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(transaction: TransactionInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    data = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(data) # -1 fraude, 1 normal
    score = model.decision_function(data)
    
    is_fraud = True if prediction[0] == -1 else False
    
    return {
        "is_fraud": is_fraud,
        "anomaly_score": float(score[0]),
        "alert": "POSIBLE FRAUDE" if is_fraud else "Normal"
    }