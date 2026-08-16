from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_data() -> pd.DataFrame:
    """Load the raw dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that all expected columns exist."""

    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Dataset schema does not match the expected schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )


def validate_missing_values(df: pd.DataFrame) -> None:
    """Validate that the dataset contains no missing values."""

    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        raise ValueError(
            f"Dataset contains missing values:\n{missing_values}"
        )


def validate_duplicates(df: pd.DataFrame) -> None:
    """Validate that the dataset contains no duplicate rows."""

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate rows."
        )


def validate_target(df: pd.DataFrame) -> None:
    """Validate the machine failure target."""

    valid_values = {0, 1}

    actual_values = set(df["Machine failure"].unique())

    if not actual_values.issubset(valid_values):
        raise ValueError(
            "Machine failure contains invalid values: "
            f"{actual_values}"
        )


def validate_product_type(df: pd.DataFrame) -> None:
    """Validate product type values."""

    valid_types = {"L", "M", "H"}

    actual_types = set(df["Type"].unique())

    if not actual_types.issubset(valid_types):
        raise ValueError(
            f"Type contains invalid values: {actual_types}"
        )


def validate_dataset(df: pd.DataFrame) -> None:
    """Run all dataset validation checks."""

    print("Running dataset validation...")

    validate_columns(df)
    print("✓ Schema validation passed")

    validate_missing_values(df)
    print("✓ Missing-value validation passed")

    validate_duplicates(df)
    print("✓ Duplicate validation passed")

    validate_target(df)
    print("✓ Target validation passed")

    validate_product_type(df)
    print("✓ Product-type validation passed")

    print("\nAll validation checks passed.")


def main() -> None:
    """Run the validation pipeline."""

    df = load_data()

    validate_dataset(df)


if __name__ == "__main__":
    main()