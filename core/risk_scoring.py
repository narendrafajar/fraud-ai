import numpy as np

def calculate_risk_scores(scores):

    # Invert karena anomaly makin kecil makin berisiko
    risk_raw = -scores

    # Normalisasi ke 0–100
    min_val = risk_raw.min()
    max_val = risk_raw.max()

    normalized = (risk_raw - min_val) / (max_val - min_val)
    risk_score = normalized * 100

    return risk_score


def categorize_risk(score):

    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"
