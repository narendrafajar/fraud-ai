import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ================================
# CONFIG
# ================================

n_samples = 50000
anomaly_ratio = 0.02
n_anomalies = int(n_samples * anomaly_ratio)

start_date = datetime(2024, 1, 1)

# ================================
# MERCHANT PROFILE
# ================================

merchant_category_map = {
    "Indomaret": "retail",
    "Alfamart": "retail",
    "Gojek": "transport",
    "GrabFood": "food",
    "Tokopedia": "online shopping",
    "Shopee": "online shopping",
    "Bukalapak": "online shopping",
    "Starbucks": "food",
    "McDonalds": "food",
    "ACE Hardware": "electronics"
}

merchants = list(merchant_category_map.keys())

payment_methods = [
    "cash",
    "credit_card",
    "e-wallet",
    "bank_transfer"
]

locations = [
    "Jakarta",
    "Bandung",
    "Surabaya",
    "Medan",
    "Yogyakarta"
]

# ================================
# CUSTOMER BEHAVIOR PROFILE
# ================================

n_customers = 300

customer_profiles = {}

for cid in range(1000, 1000 + n_customers):

    home_location = random.choice(locations)

    preferred_merchants = random.sample(merchants, k=3)

    avg_amount = random.randint(100000, 400000)

    preferred_payment = random.choice(payment_methods)

    customer_profiles[cid] = {
        "home_location": home_location,
        "preferred_merchants": preferred_merchants,
        "avg_amount": avg_amount,
        "preferred_payment": preferred_payment
    }

# ================================
# GENERATE NORMAL TRANSACTIONS
# ================================

rows = []

for i in range(n_samples):

    customer_id = random.choice(list(customer_profiles.keys()))
    profile = customer_profiles[customer_id]

    merchant = random.choice(profile["preferred_merchants"])

    category = merchant_category_map[merchant]

    amount = np.random.normal(profile["avg_amount"], 50000)
    amount = max(1000, amount)

    payment_method = profile["preferred_payment"]

    location = profile["home_location"]

    # 10% kemungkinan transaksi di kota lain
    if random.random() < 0.1:
        location = random.choice(locations)

    # tanggal random
    date = start_date + timedelta(days=random.randint(0, 364))

    rows.append({
        "transaction_id": i + 1,
        "date": date,
        "merchant": merchant,
        "category": category,
        "amount": round(amount, 2),
        "customer_id": customer_id,
        "payment_method": payment_method,
        "location": location,
        "is_true_anomaly": 0
    })

df = pd.DataFrame(rows)

# ================================
# INJECT FRAUD / ANOMALY
# ================================

anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)

# 1️⃣ Amount Fraud (besar banget)
n_amount_fraud = int(n_anomalies * 0.4)

df.loc[anomaly_indices[:n_amount_fraud], "amount"] = np.random.normal(
    1500000,
    300000,
    n_amount_fraud
).clip(min=500000)

# 2️⃣ Merchant anomaly (merchant tidak biasa)
n_merchant_fraud = int(n_anomalies * 0.3)

for idx in anomaly_indices[n_amount_fraud:n_amount_fraud+n_merchant_fraud]:

    customer = df.loc[idx, "customer_id"]
    profile = customer_profiles[customer]

    unusual_merchants = list(set(merchants) - set(profile["preferred_merchants"]))

    new_merchant = random.choice(unusual_merchants)

    df.loc[idx, "merchant"] = new_merchant
    df.loc[idx, "category"] = merchant_category_map[new_merchant]

# 3️⃣ Location anomaly (transaksi di kota lain)
n_location_fraud = int(n_anomalies * 0.2)

start = n_amount_fraud + n_merchant_fraud
end = start + n_location_fraud

for idx in anomaly_indices[start:end]:

    customer = df.loc[idx, "customer_id"]
    home = customer_profiles[customer]["home_location"]

    other_locations = list(set(locations) - {home})

    df.loc[idx, "location"] = random.choice(other_locations)

# 4️⃣ Payment anomaly
remaining = anomaly_indices[end:]

for idx in remaining:

    current = df.loc[idx, "payment_method"]

    others = list(set(payment_methods) - {current})

    df.loc[idx, "payment_method"] = random.choice(others)

# Tandai anomaly
df.loc[anomaly_indices, "is_true_anomaly"] = 1

# ================================
# SAVE DATASET
# ================================

df = df.sort_values("date")

df.to_csv("realistic_transactions.csv", index=False)

print("🔥 Dataset realistis berhasil dibuat!")
print("Total transaksi:", len(df))
print("Total anomaly:", df["is_true_anomaly"].sum())
print("File: realistic_transactions.csv")