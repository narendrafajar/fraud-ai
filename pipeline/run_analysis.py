import sys, os, json
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model_loader import load_model
from core.feature_engineering import process_features, transform_user_dataset
from core.risk_scoring import calculate_risk_scores, categorize_risk

def compute_feature_contributions(X_scaled, feature_names):
    """Hitung top 3 fitur penyumbang anomali per transaksi"""
    z = (X_scaled - X_scaled.mean(axis=0)) / np.where(X_scaled.std(axis=0)==0, 1, X_scaled.std(axis=0))
    df_contrib = pd.DataFrame(np.abs(z), columns=feature_names)
    df_contrib["top_3_features"] = df_contrib.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
    return df_contrib

def generate_reason(top_features):
    """Ubah top_features jadi reason readable"""
    if not isinstance(top_features, (list, np.ndarray)):
        top_features = []  # amanin kalau NaN / float
    reasons = []
    for f in top_features:
        if f.startswith("amount"):
            reasons.append("Jumlah transaksi tidak biasa")
        elif f.startswith("branch_"):
            branch = f.replace("branch_", "")
            reasons.append(f"Transaksi di cabang/merchant '{branch}' jarang terjadi")
        elif f.startswith("transaction_type_"):
            cat = f.replace("transaction_type_", "")
            reasons.append(f"Transaksi kategori '{cat}' jarang terjadi")
        elif f.startswith("account_type_"):
            acc = f.replace("account_type_", "")
            reasons.append(f"Metode pembayaran '{acc}' jarang digunakan")
        else:
            reasons.append(f"{f} tidak biasa")
    return "; ".join(reasons)

def adaptive_anomaly_flags(scores, n_tx):
    """Flag anomaly adaptif berdasarkan ukuran dataset"""
    if n_tx < 50:
        thresh = np.percentile(scores, 98)
    elif n_tx < 200:
        thresh = np.percentile(scores, 95)
    else:
        thresh = np.percentile(scores, 90)
    return np.where(scores >= thresh, 1, 0)

def generate_decision_recommendation(df):
    high_risk_count = df[df["risk_level"]=="High"].shape[0]
    medium_risk_count = df[df["risk_level"]=="Medium"].shape[0]
    
    if high_risk_count > 0:
        decision = "🚨 Terdapat transaksi berisiko tinggi. Segera lakukan investigasi manual."
    elif medium_risk_count > 10:
        decision = "⚠️ Beberapa transaksi berisiko menengah. Review rekomendasi AI dan lakukan sampling."
    else:
        decision = "✅ Transaksi sebagian besar normal. Monitor rutin cukup."
    return decision

# =================== VERSI FIX UNTUK EXCEL BESAR ===================
def run_analysis(input_file, output_file="analysis_result.xlsx", chunk_size=5000):
    """Support chunking untuk CSV, dan manual chunking untuk Excel besar"""
    all_transactions_json = []
    anomalies_json = []

    model, scaler, model_features = load_model()

    # ===== Baca file =====
    if input_file.endswith(".csv"):
        reader = pd.read_csv(input_file, chunksize=chunk_size)
    elif input_file.endswith((".xlsx",".xls")):
        # ==== PERUBAHAN UTAMA: Excel tidak support chunksize ====
        df_full = pd.read_excel(input_file, engine="openpyxl")  # baca full dulu
        n_rows = len(df_full)
        # split manual per chunk
        reader = [df_full.iloc[i:i+chunk_size] for i in range(0, n_rows, chunk_size)]
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau XLSX")

    total_n_tx = 0
    df_feature_contrib = None  # inisialisasi agar bisa dipakai di Excel writer

    for chunk in reader:
        n_tx = len(chunk)
        total_n_tx += n_tx
        df = transform_user_dataset(chunk)

        df["branch"] = df.get("merchant", "unknown_branch")
        df["transaction_type"] = df.get("category", "other")
        df["account_type"] = df.get("payment_method", "unknown")

        features = process_features(df.copy(), model_features)
        features = features.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_scaled = scaler.transform(features)

        scores = model.decision_function(X_scaled)
        df["predicted_anomaly"] = adaptive_anomaly_flags(scores, n_tx)

        risk_scores = calculate_risk_scores(scores)
        df["risk_score"] = risk_scores
        df["risk_level"] = np.array([categorize_risk(s) for s in risk_scores])

        df_feature_contrib = compute_feature_contributions(X_scaled, features.columns)
        df["top_features"] = df_feature_contrib["top_3_features"]
        df["reason"] = df["top_features"].apply(generate_reason)

        anomalies_json += df[df["predicted_anomaly"]==1].to_dict(orient="records")
        all_transactions_json += df.sort_values("risk_score", ascending=False).to_dict(orient="records")

    # ===== Total risk amount =====
    numeric_cols = df_feature_contrib.select_dtypes(include=[np.number]).columns
    top_impact_threshold = np.percentile(df_feature_contrib[numeric_cols].sum(axis=1), 90)
    total_risk_amount = df[df_feature_contrib[numeric_cols].sum(axis=1) >= top_impact_threshold]["amount"].sum()

    # ===== Excel writer =====
    with pd.ExcelWriter(output_file) as writer:
        pd.DataFrame(all_transactions_json).to_excel(writer, sheet_name="All Transactions", index=False)
        pd.DataFrame(anomalies_json).to_excel(writer, sheet_name="Detected Anomalies", index=False)
        pd.DataFrame({
            "Metric": ["Total Transactions","Detected Anomalies","Total Risk Amount"],
            "Value": [total_n_tx,int(len(anomalies_json)),total_risk_amount]
        }).to_excel(writer, sheet_name="Executive Summary", index=False)
        df_feature_contrib.to_excel(writer, sheet_name="Feature Contribution", index=False)

    df_all = pd.DataFrame(all_transactions_json)
    risk_distribution = df_all["risk_level"].value_counts().to_dict()
    total_anomaly = int(len(anomalies_json))
    perc_anomaly = total_anomaly / total_n_tx * 100 if total_n_tx>0 else 0

    summary_text = ""
    if total_anomaly>0:
        top_merchants = df_all[df_all["predicted_anomaly"]==1]["merchant"].value_counts().head(3).index.tolist()
        top_categories = df_all[df_all["predicted_anomaly"]==1]["category"].value_counts().head(3).index.tolist()
        summary_text = (
            f"Dari {total_n_tx} transaksi, {total_anomaly} transaksi "
            f"({perc_anomaly:.2f}%) terdeteksi berisiko. Total nilai transaksi berisiko Rp{total_risk_amount:,.0f}. "
            f"Top merchant berisiko: {', '.join(top_merchants)}. "
            f"Top kategori berisiko: {', '.join(top_categories)}."
        )
    else:
        summary_text = f"Dari {total_n_tx} transaksi, tidak ada transaksi yang terdeteksi berisiko."
    if total_n_tx < 50:
        summary_text += " ⚠️ Dataset terlalu kecil, hasil bisa kurang akurat."

    result = {
        "summary": {
            "total_transactions": total_n_tx,
            "total_anomaly": total_anomaly,
            "total_risk_amount": float(total_risk_amount)
        },
        "risk_distribution": risk_distribution,
        "all_transactions": all_transactions_json,
        "top_anomalies": anomalies_json,
        "summary_text": summary_text,
        "recom": generate_decision_recommendation(df_all),
        "output_file": output_file
    }

    print(json.dumps(result, default=str))

if __name__ == "__main__":
    input_file = sys.argv[1]
    run_analysis(input_file)