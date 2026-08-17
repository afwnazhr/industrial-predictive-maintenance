"""Load the persisted ForgeSense model and perform inference."""

from pathlib import Path

import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the persisted Random Forest model."""

    model = joblib.load(
        MODEL_PATH
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(model, features):
    """Generate prediction and failure probability."""

    probability = model.predict_proba(
        features
    )[:, 1]

    prediction = (
        probability >= 0.50
    ).astype(int)

    return prediction, probability


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FORGESENSE MODEL INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load persisted model
    # --------------------------------------------------------

    print("\n[1/3] Loading persisted model...")

    model = load_model()

    print("      Model loaded successfully.")

    # --------------------------------------------------------
    # Create example input
    # --------------------------------------------------------

    print("\n[2/3] Creating example machine input...")

    example_machine = np.array([
        [
            300.1,   # Air temperature
            310.2,   # Process temperature
            1500,    # Rotational speed
            42.0,    # Torque
            120,     # Tool wear
            10.1,    # Temperature difference
            1.033,   # Temperature ratio
            6597.3,  # Mechanical power
            6.5973,  # Mechanical power kW
            0.0,     # Type_H
            1.0,     # Type_L
            0.0,     # Type_M
        ]
    ])

    print(f"      Input shape: {example_machine.shape}")

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    print("\n[3/3] Generating prediction...")

    prediction, probability = predict(
        model,
        example_machine,
    )

    failure_probability = probability[0]
    failure_prediction = prediction[0]

    print(
        f"      Failure probability: "
        f"{failure_probability:.4f}"
    )

    print(
        f"      Prediction: "
        f"{failure_prediction}"
    )

    if failure_prediction == 1:
        print("      Status: HIGH RISK")
    else:
        print("      Status: NORMAL")

    print("\n" + "=" * 70)
    print("✓ INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()