import json

stable_features = json.load(open("artifacts/stable_features.json"))
categorical_features = json.load(open("artifacts/categorical_features.json"))
threshold = json.load(open("artifacts/threshold.json"))["threshold"]
importance = json.load(open("artifacts/feature_importance.json"))

def get_model_info():
    return {
        "stable_features": stable_features,
        "categorical_features": categorical_features,
        "threshold": threshold,
        "feature_importance": importance
    }

