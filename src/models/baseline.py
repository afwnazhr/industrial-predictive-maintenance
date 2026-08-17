"""Baseline machine-failure classification model."""

from pathlib import Path

import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the processed training and test datasets."""

    X_train = np.load(
        PROCESSED_DATA_DIR / "X_train.npy"
    )

    X_test = np.load(
        PROCESSED_DATA_DIR / "X_test.npy"
    )

    y_train = np.load(
        PROCESSED_DATA_DIR / "y_train.npy"
    )

    y_test = np.load(
        PROCESSED_DATA_DIR / "y_test.npy"
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FORGESENSE BASELINE MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    print("\nDataset:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")

    # --------------------------------------------------------
    # Create dummy baseline
    # --------------------------------------------------------

    model = DummyClassifier(
        strategy="most_frequent"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nBaseline performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    print("=" * 60)
    print("Baseline evaluation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()