import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load model & scaler
model = joblib.load("model/isolation_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

X = df.drop(columns=["Class"])
y_true = df["Class"]

X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)

# Ubah format:
# Isolation Forest: -1 = fraud, 1 = normal
# Dataset: 1 = fraud, 0 = normal

y_pred_converted = [1 if x == -1 else 0 for x in y_pred]

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_converted))

print("\nClassification Report:")
print(classification_report(y_true, y_pred_converted))
