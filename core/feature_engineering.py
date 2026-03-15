import pandas as pd
import numpy as np

def process_features(df, model_features=None):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df["log_amount"] = np.log1p(df["amount"].fillna(0))

    # Frequency features
    branch_freq = df["branch"].value_counts()
    df["branch_freq"] = df["branch"].map(branch_freq)

    account_freq = df["account_type"].value_counts()
    df["account_freq"] = df["account_type"].map(account_freq)

    # Relative features
    df["rel_amount"] = df["amount"] / (df["amount"].mean() + 1e-5)
    df["rel_branch_freq"] = df["branch_freq"] / (df["branch_freq"].max() + 1e-5)

    # Pastikan semua kolom kategorikal ada
    for col in ["transaction_type", "branch", "account_type"]:
        if col not in df.columns:
            df[col] = "unknown"

    # Convert ke dummy
    df_encoded = pd.get_dummies(
        df,
        columns=["transaction_type", "branch", "account_type"],
        dummy_na=False
    )

    # Drop non-numeric columns
    features = df_encoded.drop(
        columns=["transaction_id", "date", "is_true_anomaly"],
        errors="ignore"
    )

    # Reindex ke model_features kalau ada
    if model_features is not None:
        features = features.reindex(columns=model_features, fill_value=0)

    return features

def transform_user_dataset(df_user):
    df = df_user.copy()
    
    df["branch"] = df.get("merchant", "unknown_branch")
    df["transaction_type"] = df.get("category", "other")
    df["account_type"] = df.get("payment_method", "unknown")
    df["amount"] = df.get("amount", 0)
    df["date"] = pd.to_datetime(df.get("date", pd.Timestamp.today()), errors="coerce")

    df = df.fillna({
        "branch": "unknown_branch",
        "transaction_type": "other",
        "account_type": "unknown",
        "amount": 0
    })

    return df