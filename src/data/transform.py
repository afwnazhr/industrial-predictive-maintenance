"""Data transformation pipeline for ForgeSense."""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


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


def load_data() -> pd.DataFrame:
    """Load the raw dataset."""

    return pd.read_csv(RAW_DATA_PATH)


def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessing pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def main() -> None:
    """Build and test the transformation pipeline."""

    df = load_data()

    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]

    preprocessor = build_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    print("=" * 60)
    print("FORGESENSE DATA TRANSFORMATION")
    print("=" * 60)

    print(f"\nOriginal feature count: {X.shape[1]}")
    print(f"Transformed feature count: {X_transformed.shape[1]}")

    print("\nTransformation completed successfully.")


if __name__ == "__main__":
    main()