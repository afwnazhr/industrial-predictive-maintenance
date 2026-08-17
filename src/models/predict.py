"""ForgeSense model inference using persisted model metadata."""

from pathlib import Path
import json

import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the persisted machine-learning model."""

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    """Load model metadata from JSON."""

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# CREATE EXAMPLE INPUT
# ============================================================

def create_example_input():
    """
    Create one example machine observation.

    The input must contain the same 12 features
    used during model training.
    """

    return np.array([
        [
            # Type_H
            0,

            # Type_L
            1,

            # Type_M
            0,

            # Air temperature [K]
            300.0,

            # Process temperature [K]
            310.0,

            # Rotational speed [rpm]
            1500,

            # Torque [Nm]
            40.0,

            # Tool wear [min]
            100,

            # Temperature difference [K]
            10.0,

            # Temperature ratio
            310.0 / 300.0,

            # Mechanical power [W]
            1500 * 40.0 * (2 * np.pi / 60),

            # Mechanical power [kW]
            (
                1500
                * 40.0
                * (2 * np.pi / 60)
            ) / 1000,
        ]
    ])


# ============================================================
# GENERATE PREDICTION
# ============================================================

def predict(model, X, threshold):
    """Generate prediction using metadata threshold."""

    probability = model.predict_proba(X)[0, 1]

    prediction = int(
        probability >= threshold
    )

    return probability, prediction


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FORGESENSE MODEL INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\n[1/4] Loading persisted model...")

    model = load_model()

    print("      Model loaded successfully.")

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print("\n[2/4] Loading model metadata...")

    metadata = load_metadata()

    model_name = metadata["model_name"]
    model_version = metadata["model_version"]
    threshold = metadata["threshold"]["value"]

    print(
        f"      Model: {model_name}"
    )

    print(
        f"      Version: {model_version}"
    )

    print(
        f"      Threshold: {threshold:.2f}"
    )

    # --------------------------------------------------------
    # Create input
    # --------------------------------------------------------

    print("\n[3/4] Creating example machine input...")

    X = create_example_input()

    print(
        f"      Input shape: {X.shape}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\n[4/4] Generating prediction...")

    probability, prediction = predict(
        model,
        X,
        threshold,
    )

    status = (
        "FAILURE RISK"
        if prediction == 1
        else "NORMAL"
    )

    print(
        f"      Failure probability: "
        f"{probability:.4f}"
    )

    print(
        f"      Prediction: {prediction}"
    )

    print(
        f"      Status: {status}"
    )

    print("\n" + "=" * 70)
    print("✓ INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()