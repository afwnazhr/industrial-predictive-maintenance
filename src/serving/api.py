"""ForgeSense model serving API."""

from pathlib import Path
import json

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="ForgeSense API",
    description="Industrial machine failure prediction API",
    version="1.0.0",
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)


THRESHOLD = metadata["threshold"]["value"]


# ============================================================
# INPUT SCHEMA
# ============================================================

class MachineInput(BaseModel):
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data: MachineInput):
    """Convert raw machine data into model features."""

    temperature_difference = (
        data.process_temperature
        - data.air_temperature
    )

    temperature_ratio = (
        data.process_temperature
        / data.air_temperature
    )

    mechanical_power_w = (
        data.rotational_speed
        * data.torque
        * (2 * np.pi / 60)
    )

    mechanical_power_kw = (
        mechanical_power_w / 1000
    )

    type_h = 1 if data.type == "H" else 0
    type_l = 1 if data.type == "L" else 0
    type_m = 1 if data.type == "M" else 0

    return np.array([
        [
            type_h,
            type_l,
            type_m,
            data.air_temperature,
            data.process_temperature,
            data.rotational_speed,
            data.torque,
            data.tool_wear,
            temperature_difference,
            temperature_ratio,
            mechanical_power_w,
            mechanical_power_kw,
        ]
    ])


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """Check whether the API is running."""

    return {
        "status": "healthy",
        "model": metadata["model_name"],
        "model_version": metadata["model_version"],
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: MachineInput):
    """Generate a machine failure prediction."""

    features = create_features(data)

    probability = model.predict_proba(
        features
    )[0, 1]

    prediction = int(
        probability >= THRESHOLD
    )

    status = (
        "FAILURE RISK"
        if prediction == 1
        else "NORMAL"
    )

    return {
        "failure_probability": round(
            float(probability),
            4,
        ),
        "prediction": prediction,
        "status": status,
        "threshold": THRESHOLD,
    }