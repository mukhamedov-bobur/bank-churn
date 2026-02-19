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
    data = data[stable_features]

    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].astype(str)

    proba = model.predict_proba(data)[0, 1]
    prediction = int(proba >= threshold)

    return {
        "probability": float(proba),
        "prediction": prediction,
        "threshold": threshold
    }

