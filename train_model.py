import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

print("🔍 Loading CSV file...")

data = pd.read_csv("data/system_logs.csv")

print("🔍 Data loaded:")
print(data.head())

if data.empty:
    raise ValueError("CSV file is empty!")

model = IsolationForest(
    contamination=0.3,
    random_state=42
)

model.fit(data)

pickle.dump(model, open("model/anomaly_model.pkl", "wb"))

print("✅ Anomaly Detection Model Trained Successfully")
