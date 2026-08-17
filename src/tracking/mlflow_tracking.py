"""ForgeSense MLflow experiment tracking."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "ForgeSense - Random Forest"

THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("FORGESENSE MLFLOW EXPERIMENT TRACKING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\n[1/5] Loading processed data...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_data()

    print(
        f"      X_train: {X_train.shape}"
    )

    print(
        f"      X_test:  {X_test.shape}"
    )

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    print("\n[2/5] Configuring MLflow...")

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"      Experiment: {EXPERIMENT_NAME}"
    )

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run():

        print("\n[3/5] Training Random Forest...")

        # Model parameters
        n_estimators = 300
        max_depth = 20
        random_state = 42

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            "      Training completed."
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_test,
            probabilities,
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "threshold": THRESHOLD,
        })

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        })

        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="random_forest",
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\n[4/5] Experiment results...")

        print(
            f"      Accuracy:  {accuracy:.4f}"
        )

        print(
            f"      Precision: {precision:.4f}"
        )

        print(
            f"      Recall:    {recall:.4f}"
        )

        print(
            f"      F1-score:  {f1:.4f}"
        )

        print(
            f"      ROC-AUC:   {roc_auc:.4f}"
        )

        print(
            f"      PR-AUC:    {pr_auc:.4f}"
        )

        # ----------------------------------------------------
        # Save local model
        # ----------------------------------------------------

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            MODEL_PATH,
        )

        print("\n[5/5] Experiment tracked successfully.")

        print(
            f"      Run ID: {mlflow.active_run().info.run_id}"
        )

    print("\n" + "=" * 70)
    print("✓ MLFLOW EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()