from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

app = FastAPI()

model = joblib.load("models/isolation_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")
model_features = joblib.load("models/model_features.pkl")

# OPTIONAL: mapping kolom fleksibel
COLUMN_MAPPING = {
    "tanggal": "date",
    "nominal": "amount",
    "jenis": "transaction_type",
    "cabang": "branch",
    "akun": "account_type"
}

def preprocess(df):

    # Rename jika ada kolom alternatif
    df = df.rename(columns=COLUMN_MAPPING)

    # Pastikan kolom wajib ada
    required_cols = ["transaction_id", "date", "amount",
                     "transaction_type", "branch", "account_type"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df["log_amount"] = np.log1p(df["amount"])

    branch_freq = df["branch"].value_counts()
    df["branch_freq"] = df["branch"].map(branch_freq)

    account_freq = df["account_type"].value_counts()
    df["account_freq"] = df["account_type"].map(account_freq)

    df = pd.get_dummies(df, columns=["transaction_type", "branch", "account_type"])

    df = df.drop(columns=["transaction_id", "date"])

    # 🔥 PENTING: Samakan struktur dengan training
    df = df.reindex(columns=model_features, fill_value=0)

    return df


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    if file.filename.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(contents))
    else:
        df = pd.read_csv(BytesIO(contents))

    processed = preprocess(df)
    X_scaled = scaler.transform(processed)

    pred = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df["anomaly"] = np.where(pred == -1, 1, 0)
    df["anomaly_score"] = scores

    total_data = len(df)
    total_anomaly = int(df["anomaly"].sum())
    percent_anomaly = round((total_anomaly / total_data) * 100, 2)

    top_anomaly = df[df["anomaly"] == 1] \
        .sort_values("anomaly_score") \
        .head(20)

    return {
        "summary": {
            "total_data": total_data,
            "total_anomaly": total_anomaly,
            "percent_anomaly": percent_anomaly
        },
        "top_anomalies": top_anomaly.to_dict(orient="records")
    }
