"""Feature definitions for the ForgeSense predictive maintenance model."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


TARGET_COLUMN = "Machine failure"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_data() -> pd.DataFrame:
    """Load the raw dataset."""

    return pd.read_csv(RAW_DATA_PATH)


def validate_feature_definition(df: pd.DataFrame) -> None:
    """Validate the feature, target, and excluded-column definitions."""

    expected_columns = set(df.columns)

    defined_columns = (
        set(FEATURE_COLUMNS)
        | {TARGET_COLUMN}
        | set(EXCLUDED_COLUMNS)
    )

    if expected_columns != defined_columns:
        missing = expected_columns - defined_columns
        unexpected = defined_columns - expected_columns

        raise ValueError(
            "Feature definition does not match dataset schema.\n"
            f"Missing definitions: {missing}\n"
            f"Unexpected definitions: {unexpected}"
        )

    if TARGET_COLUMN in FEATURE_COLUMNS:
        raise ValueError(
            "Target column cannot also be a feature."
        )

    if set(FEATURE_COLUMNS) & set(EXCLUDED_COLUMNS):
        raise ValueError(
            "A column cannot be both a feature and excluded."
        )


def main() -> None:
    """Run feature definition validation."""

    df = load_data()

    validate_feature_definition(df)

    print("=" * 60)
    print("FORGESENSE FEATURE DEFINITION")
    print("=" * 60)

    print("\nFeatures:")
    for feature in FEATURE_COLUMNS:
        print(f"  ✓ {feature}")

    print(f"\nTarget:")
    print(f"  → {TARGET_COLUMN}")

    print("\nExcluded columns:")
    for column in EXCLUDED_COLUMNS:
        print(f"  ✗ {column}")

    print("\nFeature definition validation passed.")


if __name__ == "__main__":
    main()