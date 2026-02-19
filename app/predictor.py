import json
import pandas as pd
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("artifacts/model.cbm")

stable_features = json.load(open("artifacts/stable_features.json"))
threshold = json.load(open("artifacts/threshold.json"))["threshold"]
categorical_features = json.load(open("artifacts/categorical_features.json"))

def predict(request):
    data = pd.DataFrame([request.features])
    
    # -------------------------------
    # Keep only stable features
    # -------------------------------
    for f in stable_features:
        if f not in data.columns:
            # Fill missing numerical/categorical features
            if f in categorical_features:
                data[f] = "unknown"
            else:
                data[f] = 0
    
    # Select stable features in correct order
    data = data[stable_features]

    # Cast categorical features
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].astype(str)

    proba = model.predict_proba(data)[0, 1]
    prediction = int(proba >= threshold)

    return {
        "prediction": prediction,
        "probability": float(proba),
    }

