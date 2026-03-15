import pandas as pd

# Load hasil eksperimen
df = pd.read_csv("experiment_results.csv")

# =========================
# Tambah Kategori Precision
# =========================
def categorize_precision(p):
    if p >= 0.8:
        return "Sangat Tinggi"
    elif p >= 0.6:
        return "Tinggi"
    elif p >= 0.4:
        return "Sedang"
    else:
        return "Rendah"

def categorize_recall(r):
    if r >= 0.8:
        return "Sangat Tinggi"
    elif r >= 0.6:
        return "Tinggi"
    elif r >= 0.4:
        return "Sedang"
    else:
        return "Rendah"

def categorize_auc(a):
    if a >= 0.9:
        return "Excellent"
    elif a >= 0.8:
        return "Baik"
    elif a >= 0.7:
        return "Cukup"
    else:
        return "Lemah"

# Tambah kolom interpretasi
df["precision_level"] = df["precision"].apply(categorize_precision)
df["recall_level"] = df["recall"].apply(categorize_recall)
df["auc_level"] = df["roc_auc"].apply(categorize_auc)

interpretations = []

for _, row in df.iterrows():
    behavior = ""
    
    if row["precision"] > row["recall"]:
        behavior = "Model konservatif (sedikit false alarm, banyak anomaly lolos)"
    elif row["precision"] < row["recall"]:
        behavior = "Model agresif (banyak anomaly tertangkap, risiko false alarm)"
    else:
        behavior = "Model seimbang antara precision dan recall"

    explanation = (
        f"Dengan contamination {row['contamination']} dan random_state {row['random_state']}, "
        f"model memiliki precision {row['precision']:.3f} ({row['precision_level']}), "
        f"recall {row['recall']:.3f} ({row['recall_level']}), "
        f"F1-score {row['f1_score']:.3f}, dan ROC-AUC {row['roc_auc']:.3f} ({row['auc_level']}). "
        f"{behavior}."
    )

    interpretations.append({
        "model_behavior": behavior,
        "interpretasi": explanation
    })

interpretation_df = pd.DataFrame(interpretations)

df = pd.concat([df, interpretation_df], axis=1)

# =========================
# Summary per contamination
# =========================
summary = df.groupby("contamination").agg({
    "precision": "mean",
    "recall": "mean",
    "f1_score": "mean",
    "roc_auc": "mean"
}).reset_index()

summary["tradeoff_analysis"] = summary.apply(
    lambda row: "Semakin tinggi contamination → recall naik, precision turun"
    if row["recall"] > row["precision"]
    else "Model cenderung konservatif",
    axis=1
)

# =========================
# Export ke Excel
# =========================
with pd.ExcelWriter("experiment_results_detailed.xlsx") as writer:
    df.to_excel(writer, sheet_name="Detailed Results", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)

print("🔥 experiment_results_detailed.xlsx berhasil dibuat!")
