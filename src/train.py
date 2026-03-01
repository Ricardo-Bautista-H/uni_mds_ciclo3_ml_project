import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split
import joblib
import os
import mlflow
import mlflow.sklearn

# 1. Cargar Datos
print("Cargando dataset...")
try:
    df = pd.read_csv('data/raw/creditcard.csv')
except FileNotFoundError:
    print("ERROR: No se encuentra data/raw/creditcard.csv")
    exit()

df_sample = df.sample(n=50000, random_state=42)
X = df_sample.drop(columns=['Class'])
y = df_sample['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === INICIO DE MLFLOW ===
# Configuramos el experimento
mlflow.set_experiment("Fraud_Detection_IsolationForest")

with mlflow.start_run():
    print("Entrenando Isolation Forest con MLflow...")
    
    # Definimos y registramos los hiperparámetros
    n_estimators = 100
    contamination = 0.01
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("contamination", contamination)
    mlflow.log_param("sample_size", 50000)

    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X_train)

    # Evaluación
    print("Evaluando modelo...")
    y_pred_raw = model.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

    # Calculamos y registramos las métricas
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("recall", recall)

    print(f"F1 Score: {f1:.4f} | Recall: {recall:.4f}")

    # 5. Guardar el Modelo localmente y en MLflow
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = 'models/isolation_forest_fraud.pkl'
    joblib.dump(model, model_path)
    
    # Registramos el modelo en MLflow
    mlflow.sklearn.log_model(model, "isolation_forest_model")
    print(f"Modelo guardado en: {model_path} y registrado en MLflow.")
# === FIN DE MLFLOW ===