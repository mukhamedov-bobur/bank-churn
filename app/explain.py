import json
import pandas as pd
import shap
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("artifacts/model.cbm")

stable_features = json.load(open("artifacts/stable_features.json"))
explainer = shap.TreeExplainer(model)

def explain_instance(request):
    data = pd.DataFrame([request.features])
    data = data[stable_features]

    shap_values = explainer(data)
    print(shap_values)

    return {
        "base_value": float(shap_values.base_values[0]),
        "feature_contributions": dict(
            zip(stable_features,
                shap_values.values[0].tolist())
        )
    }

