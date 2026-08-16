from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


def load_data() -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def profile_target(df: pd.DataFrame) -> None:
    """Analyze the machine failure target."""

    print("=" * 60)
    print("TARGET ANALYSIS")
    print("=" * 60)

    print("\nMachine failure distribution:")
    print(df["Machine failure"].value_counts())

    print("\nMachine failure percentage:")
    print(df["Machine failure"].value_counts(normalize=True) * 100)


def profile_categorical_features(df: pd.DataFrame) -> None:
    """Analyze categorical features."""

    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURES")
    print("=" * 60)

    print("\nProduct type distribution:")
    print(df["Type"].value_counts())


def profile_numeric_features(df: pd.DataFrame) -> None:
    """Analyze numerical features."""

    print("\n" + "=" * 60)
    print("NUMERICAL FEATURES")
    print("=" * 60)

    numeric_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    print(df[numeric_columns].describe())


def main() -> None:
    """Run dataset profiling."""

    df = load_data()

    profile_target(df)
    profile_categorical_features(df)
    profile_numeric_features(df)


if __name__ == "__main__":
    main()