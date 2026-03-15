import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Buat folder model kalau belum ada
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Ambil fitur (buang label)
X = df.drop(columns=["Class"])

# Scaling (WAJIB untuk Isolation Forest biar stabil)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Buat model
model = IsolationForest(
    n_estimators=100,
    contamination=0.0017,  # sesuai proporsi fraud dataset
    random_state=42
)

# Training
model.fit(X_scaled)

# Simpan model dan scaler
joblib.dump(model, "model/isolation_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("🔥 Isolation Forest berhasil dilatih dan disimpan!")
