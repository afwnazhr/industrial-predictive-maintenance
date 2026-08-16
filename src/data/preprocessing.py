"""Leakage-safe preprocessing pipeline for ForgeSense."""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai4i2020.csv"
)


TARGET_COLUMN = "Machine failure"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
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
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the raw dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target."""

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing transformer."""

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


def main() -> None:
    """Run the complete preprocessing workflow."""

    print("=" * 60)
    print("FORGESENSE PREPROCESSING PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = load_data()

    print("\nRaw dataset:")
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # 2. Select features and target
    # ---------------------------------------------------------

    X, y = split_features_target(df)

    print("\nModel inputs:")
    print(f"  Features: {X.shape[1]}")

    print("\nTarget:")
    print(f"  {TARGET_COLUMN}")

    # ---------------------------------------------------------
    # 3. Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
    )

    print("\nTrain/test split:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")

    # ---------------------------------------------------------
    # 4. Build preprocessing transformer
    # ---------------------------------------------------------

    preprocessor = build_preprocessor()

    # ---------------------------------------------------------
    # 5. FIT ONLY ON TRAINING DATA
    # ---------------------------------------------------------

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    # ---------------------------------------------------------
    # 6. TRANSFORM TEST DATA
    # ---------------------------------------------------------

    X_test_transformed = preprocessor.transform(
        X_test
    )

    print("\nTransformation:")
    print(
        f"  X_train transformed: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"  X_test transformed:  "
        f"{X_test_transformed.shape}"
    )

    print("\n✓ Preprocessor fitted on training data only.")
    print("✓ Test data transformed without fitting.")
    print("✓ Leakage-safe preprocessing completed.")


if __name__ == "__main__":
    main()