import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from core.feature_engineering import process_features

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("realistic_transactions.csv")

# ======================
# MAP KE FEATURE MODEL
# ======================
df["branch"] = df["merchant"]
df["transaction_type"] = df["category"]
df["account_type"] = df["payment_method"]

# ======================
# FEATURE ENGINEERING
# ======================
# Gunakan process_features, ini pasti nge-dummy semua kolom string
features = process_features(df.copy(), model_features=None)

# ======================
# Pastikan semua kolom numeric
# ======================
# Kadang masih ada sisa object kolom, force convert
features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

# ======================
# SCALING
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ======================
# TRAIN MODEL
# ======================
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled)

# ======================
# SAVE MODEL + SCALER
# ======================
joblib.dump(model, "models/isolation_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features.columns.tolist(), "models/model_features.pkl")

print("🔥 Model, scaler, dan fitur berhasil disimpan!")
