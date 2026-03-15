import pandas as pd
import numpy as np
from core.feature_engineering import process_features
from core.model_loader import load_model
from core.risk_scoring import calculate_risk_scores
from core.feature_engineering import transform_user_dataset

def validate_pipeline(df):
    model, scaler, model_features = load_model()
    
    # Missing value check
    if df.isna().sum().sum() > 0:
        print("⚠️ Warning: terdapat missing value, akan di-fill 0")
        df = df.fillna(0)
        
    df = transform_user_dataset(df)
    features = process_features(df.copy(), model_features)
    
    X_scaled = scaler.transform(features)
    
    pred = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    risk_scores = calculate_risk_scores(scores)
    
    # Basic sanity check
    assert len(risk_scores) == len(df), "Jumlah risk_score tidak sama dengan jumlah data"
    assert ((risk_scores >= 0) & (risk_scores <= 100)).all(), "Risk score harus di 0-100"
    
    print("✅ Pipeline validasi berhasil, semua normal")
    return pred, risk_scores
