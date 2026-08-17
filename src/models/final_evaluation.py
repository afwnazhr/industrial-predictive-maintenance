"""Final evaluation of the ForgeSense Random Forest model."""

from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SELECTED_THRESHOLD = 0.45


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load processed training and test data."""

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

    print("=" * 70)
    print("FORGESENSE FINAL MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    print("\nDataset:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")

    print("\nSelected model:")
    print("  Random Forest")

    print("\nSelected threshold:")
    print(f"  {SELECTED_THRESHOLD:.2f}")

    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    print("\nTraining final Random Forest...")

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

    print("Final model training completed.")

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Default threshold = 0.50
    # --------------------------------------------------------

    y_pred_default = (
        y_probability >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # Optimized threshold = 0.45
    # --------------------------------------------------------

    y_pred_optimized = (
        y_probability >= SELECTED_THRESHOLD
    ).astype(int)

    # ========================================================
    # DEFAULT THRESHOLD EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST SET — DEFAULT THRESHOLD (0.50)")
    print("=" * 70)

    accuracy_default = accuracy_score(
        y_test,
        y_pred_default,
    )

    precision_default = precision_score(
        y_test,
        y_pred_default,
        zero_division=0,
    )

    recall_default = recall_score(
        y_test,
        y_pred_default,
        zero_division=0,
    )

    f1_default = f1_score(
        y_test,
        y_pred_default,
        zero_division=0,
    )

    cm_default = confusion_matrix(
        y_test,
        y_pred_default,
    )

    print(f"\nAccuracy:  {accuracy_default:.4f}")
    print(f"Precision: {precision_default:.4f}")
    print(f"Recall:    {recall_default:.4f}")
    print(f"F1-score:  {f1_default:.4f}")

    print("\nConfusion matrix:")
    print(cm_default)

    # ========================================================
    # OPTIMIZED THRESHOLD EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print(
        f"TEST SET — OPTIMIZED THRESHOLD "
        f"({SELECTED_THRESHOLD:.2f})"
    )
    print("=" * 70)

    accuracy = accuracy_score(
        y_test,
        y_pred_optimized,
    )

    precision = precision_score(
        y_test,
        y_pred_optimized,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred_optimized,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred_optimized,
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
        y_pred_optimized,
    )

    # --------------------------------------------------------
    # Extract confusion matrix values
    # --------------------------------------------------------

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nOperational results:")
    print(f"  True negatives:  {tn}")
    print(f"  False positives: {fp}")
    print(f"  False negatives: {fn}")
    print(f"  True positives:  {tp}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred_optimized,
            zero_division=0,
        )
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Metric':<15}"
        f"{'Threshold 0.50':>18}"
        f"{'Threshold 0.45':>18}"
    )

    print("-" * 55)

    print(
        f"{'Accuracy':<15}"
        f"{accuracy_default:>18.4f}"
        f"{accuracy:>18.4f}"
    )

    print(
        f"{'Precision':<15}"
        f"{precision_default:>18.4f}"
        f"{precision:>18.4f}"
    )

    print(
        f"{'Recall':<15}"
        f"{recall_default:>18.4f}"
        f"{recall:>18.4f}"
    )

    print(
        f"{'F1-score':<15}"
        f"{f1_default:>18.4f}"
        f"{f1:>18.4f}"
    )

    print("\n" + "=" * 70)
    print("Final evaluation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()