import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("simulated_transactions.csv")

# =========================
# FEATURE ENGINEERING
# =========================
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

features = df.drop(columns=["transaction_id", "date", "is_true_anomaly"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

y_true = df["is_true_anomaly"]

# =========================
# EXPERIMENT SETTINGS
# =========================
contaminations = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
random_states = [0, 42, 99]

results = []

# =========================
# LOOP EXPERIMENT
# =========================
for c in contaminations:
    for rs in random_states:

        model = IsolationForest(
            n_estimators=200,
            contamination=c,
            random_state=rs,
            n_jobs=-1
        )

        model.fit(X_scaled)
        pred = model.predict(X_scaled)

        y_pred = np.where(pred == -1, 1, 0)

        scores = model.decision_function(X_scaled)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, -scores)

        results.append({
            "contamination": c,
            "random_state": rs,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc
        })

# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame(results)
results_df.to_csv("experiment_results.csv", index=False)

print("🔥 Eksperimen selesai! Hasil disimpan ke experiment_results.csv")
print(results_df)
