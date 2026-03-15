import joblib

def load_model():

    model = joblib.load("models/isolation_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    model_features = joblib.load("models/model_features.pkl")

    return model, scaler, model_features
