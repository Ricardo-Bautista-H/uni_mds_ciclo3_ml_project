import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. Cargar Datos
print("Cargando dataset...")
try:
    df = pd.read_csv('data/raw/creditcard.csv')
except FileNotFoundError:
    print("ERROR: No se encuentra data/raw/creditcard.csv. Descárgalo de Kaggle.")
    exit()

# Usaremos una muestra para la demo (50k registros)
df_sample = df.sample(n=50000, random_state=42)

X = df_sample.drop(columns=['Class'])
y = df_sample['Class']

# 2. Split (Solo para validación)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar Isolation Forest
# Si bien, es un dataset para clasificación, se probará con un algoritmo de detección de anomalías
print("Entrenando Isolation Forest...")
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
model.fit(X_train)

# 4. Evaluación
print("Evaluando modelo...")
y_pred_raw = model.predict(X_test)
# Isolation Forest: -1 es anomalía, 1 es normal. Mapeamos a 0 y 1.
y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

print("Métricas del Modelo:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))
print(f"F1 Score (Fraude): {f1_score(y_test, y_pred):.4f}")

# 5. Guardar el Modelo
if not os.path.exists('models'):
    os.makedirs('models')

model_path = 'models/isolation_forest_fraud.pkl'
joblib.dump(model, model_path)
print(f"Modelo guardado en: {model_path}")