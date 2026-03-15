import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from core.feature_engineering import process_features
# =========================
# LOAD DATA & MODEL
# =========================


df = pd.read_csv("realistic_transactions.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

model = joblib.load("models/isolation_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_features = joblib.load("models/model_features.pkl")

# =========================
# FEATURE ENGINEERING (HARUS IDENTIK)
# =========================
df["branch"] = df["merchant"]
df["transaction_type"] = df["category"]
df["account_type"] = df["payment_method"]
features = process_features(df.copy(), model_features=model_features)  # pake model_features
features = features.apply(pd.to_numeric, errors="coerce").fillna(0)
X_scaled = scaler.transform(features)

# =========================
# PREDICTION
# =========================
pred = model.predict(X_scaled)
scores = model.decision_function(X_scaled)

df["anomaly_score"] = scores
df["predicted_anomaly"] = np.where(pred == -1, 1, 0)

# =========================
# INTERPRETASI OTOMATIS
# =========================
def interpret_row(row):
    reasons = []

    if row["amount"] > df["amount"].mean() + 2 * df["amount"].std():
        reasons.append("Nominal transaksi sangat tinggi")

    if row["account_type"] == "Gaji" and row["transaction_type"] == "debit":
        reasons.append("Transaksi Gaji dengan tipe debit tidak wajar")

    if row["date"].month == 12 and row["date"].day >= 28:
        reasons.append("Transaksi akhir tahun (risk pattern)")

    if row["predicted_anomaly"] == 0:
        return "Normal"

    if reasons:
        return " | ".join(reasons)
    else:
        return "Terdeteksi anomali berdasarkan pola statistik model"

df["analysis_explanation"] = df.apply(interpret_row, axis=1)

# =========================
# METRICS
# =========================
y_true = df["is_true_anomaly"]
y_pred = df["predicted_anomaly"]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, -scores)

conf_matrix = confusion_matrix(y_true, y_pred)

summary_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score", "ROC AUC"],
    "Value": [precision, recall, f1, auc]
})

conf_df = pd.DataFrame(conf_matrix,
                       columns=["Predicted Normal", "Predicted Anomaly"],
                       index=["Actual Normal", "Actual Anomaly"])

# =========================
# EXPORT EXCEL
# =========================
with pd.ExcelWriter("model_analysis_report.xlsx") as writer:

    df.sort_values("anomaly_score").to_excel(writer,
                                             sheet_name="All Transactions",
                                             index=False)

    df[df["predicted_anomaly"] == 1].sort_values("anomaly_score") \
        .to_excel(writer,
                  sheet_name="Detected Anomalies",
                  index=False)

    summary_df.to_excel(writer,
                        sheet_name="Model Metrics",
                        index=False)

    conf_df.to_excel(writer,
                     sheet_name="Confusion Matrix")

print("🔥 model_analysis_report.xlsx berhasil dibuat!")
