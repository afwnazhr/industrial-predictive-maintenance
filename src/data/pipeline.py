"""Final reproducible data pipeline for ForgeSense."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai4i2020.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# DATA CONFIGURATION
# ============================================================

TARGET_COLUMN = "Machine failure"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature difference [K]",
    "Temperature ratio",
    "Mechanical power [W]",
    "Mechanical power [kW]",
]

CATEGORICAL_FEATURES = [
    "Type",
]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature difference [K]",
    "Temperature ratio",
    "Mechanical power [W]",
    "Mechanical power [kW]",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the raw dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def create_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create physically meaningful features."""

    data = df.copy()

    # Temperature difference
    data["Temperature difference [K]"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    # Temperature ratio
    data["Temperature ratio"] = (
        data["Process temperature [K]"]
        / data["Air temperature [K]"]
    )

    # Angular velocity
    angular_velocity = (
        2
        * np.pi
        * data["Rotational speed [rpm]"]
        / 60
    )

    # Mechanical power
    data["Mechanical power [W]"] = (
        data["Torque [Nm]"]
        * angular_velocity
    )

    # Mechanical power in kW
    data["Mechanical power [kW]"] = (
        data["Mechanical power [W]"]
        / 1000
    )

    return data


# ============================================================
# 3. BUILD PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """Build the leakage-safe preprocessing transformer."""

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )


# ============================================================
# 4. SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
) -> None:
    """Save processed datasets and preprocessing artifacts."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        PROCESSED_DATA_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        PROCESSED_DATA_DIR / "X_test.npy",
        X_test,
    )

    np.save(
        PROCESSED_DATA_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        PROCESSED_DATA_DIR / "y_test.npy",
        y_test,
    )

    joblib.dump(
        preprocessor,
        PROCESSED_DATA_DIR / "preprocessor.joblib",
    )


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main() -> None:
    """Execute the complete data engineering pipeline."""

    print("=" * 60)
    print("FORGESENSE FINAL DATA PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load raw data
    # --------------------------------------------------------

    df = load_data()

    print("\n[1/6] Raw data loaded")
    print(f"      Samples: {len(df)}")
    print(f"      Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Step 2: Feature engineering
    # --------------------------------------------------------

    df = create_engineered_features(df)

    print("\n[2/6] Feature engineering completed")
    print(
        f"      Total available columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Step 3: Select features and target
    # --------------------------------------------------------

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print("\n[3/6] Features and target selected")
    print(f"      Features: {X.shape[1]}")
    print(f"      Target: {TARGET_COLUMN}")

    # --------------------------------------------------------
    # Step 4: Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n[4/6] Train/test split completed")
    print(f"      X_train: {X_train.shape}")
    print(f"      X_test:  {X_test.shape}")
    print(f"      y_train: {y_train.shape}")
    print(f"      y_test:  {y_test.shape}")

    # --------------------------------------------------------
    # Step 5: Fit preprocessing ONLY on training data
    # --------------------------------------------------------

    preprocessor = build_preprocessor()

    X_train_transformed = (
        preprocessor.fit_transform(X_train)
    )

    X_test_transformed = (
        preprocessor.transform(X_test)
    )

    print("\n[5/6] Preprocessing completed")
    print(
        f"      X_train transformed: "
        f"{X_train_transformed.shape}"
    )
    print(
        f"      X_test transformed:  "
        f"{X_test_transformed.shape}"
    )

    # --------------------------------------------------------
    # Step 6: Save artifacts
    # --------------------------------------------------------

    save_processed_data(
        X_train_transformed,
        X_test_transformed,
        y_train.to_numpy(),
        y_test.to_numpy(),
        preprocessor,
    )

    print("\n[6/6] Processed data saved")
    print(
        f"      Output directory: "
        f"{PROCESSED_DATA_DIR}"
    )

    print("\n" + "=" * 60)
    print("✓ FINAL DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()