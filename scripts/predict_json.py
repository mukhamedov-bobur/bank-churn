"""
Batch prediction from JSON file.

Usage:
    python scripts/predict_json.py path_to_file.json

The script:
1. Loads model artifacts
2. Reads JSON file (list of objects or dict)
3. Filters only stable features
4. Casts categorical features
5. Returns predictions with probabilities
"""

import json
import sys
import pandas as pd
from app.predictor import predict, stable_features, categorical_features

if len(sys.argv) < 2:
    print("Usage: python predict_json.py path_to_file.json")
    sys.exit(1)

json_file = sys.argv[1]

# Load JSON file
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure input is list of dicts (even single row)
if isinstance(data, dict):
    data = [data]

df = pd.DataFrame(data)

# Keep only stable features
missing_features = [f for f in stable_features if f not in df.columns]
if missing_features:
    print("Warning: missing features in input JSON:", missing_features)

df = df[[f for f in stable_features if f in df.columns]]

# Cast categorical features
for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Predict row by row
results = []
for _, row in df.iterrows():
    class DummyRequest:
        def __init__(self, features):
            self.features = features
    req = DummyRequest(row.to_dict())
    result = predict(req)
    results.append(result)

# Print results
for i, r in enumerate(results):
    print(f"Row {i+1}: {r}")

