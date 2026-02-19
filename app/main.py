"""
Main API entrypoint.

Run locally:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from app.schemas import PredictionRequest, BatchPredictionRequest
from app.predictor import predict
from app.model_info import get_model_info
from app.explain import explain_instance

app = FastAPI(title="Bank Churn ML API")

@app.get("/")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict")
def make_prediction(request: PredictionRequest):
    """Return probability and class prediction."""
    return predict(request)

@app.get("/model-info")
def model_info():
    """Return model metadata and required features."""
    return get_model_info()

@app.post("/explain")
def explain(request: PredictionRequest):
    """Return SHAP explanation for a single instance."""
    return explain_instance(request)


# ------------------------------
# New batch prediction endpoint
# ------------------------------
@app.post("/predict-batch")
def predict_batch(request: BatchPredictionRequest):
    results = []
    for row in request.data:
        # Wrap in PredictionRequest schema
        pred_request = PredictionRequest(features=row)
        try:
            result = predict(pred_request)
        except Exception as e:
            # Handle missing feature errors gracefully
            result = {"error": str(e)}
        results.append(result)
    return {"predictions": results}
