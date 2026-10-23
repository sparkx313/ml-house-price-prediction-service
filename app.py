from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from price_predictor import load_artifact, predict_frame

DEFAULT_MODEL_PATH = "models/house_price_model.joblib"

app = FastAPI(
    title="House Price Valuation API",
    description="Trainable regression model for house price estimates.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        description="Feature values using the column names from the training data."
    )


@lru_cache(maxsize=1)
def _load_model(path: str):
    try:
        return load_artifact(path)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="No trained model is available. Train a model before serving.",
        ) from error


@app.get("/health")
def health():
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    return {
        "status": "ready" if model_path.is_file() else "model_missing",
        "model_path": str(model_path),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    artifact = _load_model(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    frame = pd.DataFrame([request.features])
    try:
        value = float(predict_frame(artifact, frame)[0])
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "predicted_price": value,
        "target_name": artifact["target_name"],
        "target_units": artifact["target_units"],
    }
