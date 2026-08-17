"""Train and persist the final ForgeSense Random Forest model."""

from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():
    """Load processed training data."""

    X_train = np.load(
        PROCESSED_DATA_DIR / "X_train.npy"
    )

    y_train = np.load(
        PROCESSED_DATA_DIR / "y_train.npy"
    )

    return X_train, y_train


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):
    """Train the final Random Forest model."""

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """Save trained model to disk."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    return MODEL_PATH


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FORGESENSE MODEL PERSISTENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\n[1/3] Loading training data...")

    X_train, y_train = load_training_data()

    print(f"      X_train: {X_train.shape}")
    print(f"      y_train: {y_train.shape}")

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\n[2/3] Training final Random Forest...")

    model = train_model(
        X_train,
        y_train,
    )

    print("      Training completed.")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print("\n[3/3] Saving model...")

    model_path = save_model(model)

    print(f"      Model saved to:")
    print(f"      {model_path}")

    print("\n" + "=" * 70)
    print("✓ MODEL PERSISTENCE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()