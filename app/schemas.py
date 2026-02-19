"""
Pydantic schemas define the API contract.
"""

from pydantic import BaseModel
from typing import Dict, Any, List

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, Any]]  # List of feature dictionaries

