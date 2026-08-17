"""Logistic Regression model for ForgeSense."""

from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
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
    """Load processed train/test data."""

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
    print("FORGESENSE LOGISTIC REGRESSION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    print("\nDataset:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
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

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nModel performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")

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
    print("Logistic Regression evaluation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()