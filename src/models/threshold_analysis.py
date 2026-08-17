"""Threshold optimization for ForgeSense Random Forest."""

from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


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
    """Load the existing processed training and test data."""

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
    print("FORGESENSE THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    print("\nOriginal data:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")

    # --------------------------------------------------------
    # Create validation split
    # --------------------------------------------------------

    X_model_train, X_validation, y_model_train, y_validation = (
        train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            stratify=y_train,
            random_state=42,
        )
    )

    print("\nTraining / validation split:")
    print(f"  Model training: {X_model_train.shape}")
    print(f"  Validation:     {X_validation.shape}")

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_model_train,
        y_model_train,
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Get validation probabilities
    # --------------------------------------------------------

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    thresholds = np.arange(
        0.05,
        1.00,
        0.05,
    )

    results = []

    for threshold in thresholds:

        y_pred = (
            validation_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            y_pred,
            zero_division=0,
        )

        false_positives = np.sum(
            (y_validation == 0) & (y_pred == 1)
        )

        false_negatives = np.sum(
            (y_validation == 1) & (y_pred == 0)
        )

        results.append(
            (
                threshold,
                precision,
                recall,
                f1,
                false_positives,
                false_negatives,
            )
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nThreshold analysis:")
    print(
        f"{'Threshold':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FP':>8} "
        f"{'FN':>8}"
    )

    print("-" * 65)

    for result in results:

        threshold, precision, recall, f1, fp, fn = result

        print(
            f"{threshold:>10.2f} "
            f"{precision:>10.4f} "
            f"{recall:>10.4f} "
            f"{f1:>10.4f} "
            f"{fp:>8} "
            f"{fn:>8}"
        )

    # --------------------------------------------------------
    # Best F1 threshold
    # --------------------------------------------------------

    best_f1_result = max(
        results,
        key=lambda x: x[3],
    )

    print("\nBest F1 threshold:")
    print(
        f"  Threshold: {best_f1_result[0]:.2f}"
    )
    print(
        f"  Precision: {best_f1_result[1]:.4f}"
    )
    print(
        f"  Recall:    {best_f1_result[2]:.4f}"
    )
    print(
        f"  F1-score:  {best_f1_result[3]:.4f}"
    )

    # --------------------------------------------------------
    # Best threshold with recall >= 90%
    # --------------------------------------------------------

    recall_target = 0.90

    eligible_results = [
        result
        for result in results
        if result[2] >= recall_target
    ]

    if eligible_results:

        best_recall_result = max(
            eligible_results,
            key=lambda x: x[1],
        )

        print(
            "\nBest threshold with "
            "recall >= 90%:"
        )

        print(
            f"  Threshold: "
            f"{best_recall_result[0]:.2f}"
        )

        print(
            f"  Precision: "
            f"{best_recall_result[1]:.4f}"
        )

        print(
            f"  Recall:    "
            f"{best_recall_result[2]:.4f}"
        )

        print(
            f"  F1-score:  "
            f"{best_recall_result[3]:.4f}"
        )

        print(
            f"  False positives: "
            f"{best_recall_result[4]}"
        )

        print(
            f"  False negatives: "
            f"{best_recall_result[5]}"
        )

    else:

        print(
            "\nNo threshold achieved "
            "recall >= 90%."
        )

    print("\n" + "=" * 70)
    print("Threshold optimization completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()