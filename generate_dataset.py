import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

n_samples = 50000
anomaly_ratio = 0.05
n_anomalies = int(n_samples * anomaly_ratio)

# Generate tanggal random dalam 1 tahun
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]

account_types = ['Kas', 'Penjualan', 'Biaya Operasional', 'Pembelian', 'Gaji']
branches = ['Jakarta', 'Bandung', 'Surabaya', 'Medan']
transaction_types = ['debit', 'credit']

data = {
    "transaction_id": range(1, n_samples + 1),
    "date": dates,
    "account_type": np.random.choice(account_types, n_samples),
    "amount": np.random.normal(2000000, 500000, n_samples),
    "transaction_type": np.random.choice(transaction_types, n_samples),
    "branch": np.random.choice(branches, n_samples),
}

df = pd.DataFrame(data)

# Buat amount tidak negatif
df["amount"] = df["amount"].abs()

# Inject anomaly
df["is_true_anomaly"] = 0
anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)

# 1. Amount agak tinggi
df.loc[anomaly_indices[:800], "amount"] = np.random.normal(3200000, 700000, 800)

# 2. Gaji debit (aneh)
df.loc[anomaly_indices[800:1600], "account_type"] = "Gaji"
df.loc[anomaly_indices[800:1600], "transaction_type"] = "debit"

# 3. Transaksi minggu akhir tahun
df.loc[anomaly_indices[1600:], "date"] = pd.to_datetime("2024-12-29")

df.loc[anomaly_indices, "is_true_anomaly"] = 1

df.to_csv("simulated_transactions.csv", index=False)

print("🔥 Dataset simulasi berhasil dibuat!")
