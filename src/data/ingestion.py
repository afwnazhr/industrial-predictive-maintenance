from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_FILENAME = "ai4i2020.csv"


def load_raw_data() -> pd.DataFrame:
    """Load the raw predictive maintenance dataset."""

    dataset_path = RAW_DATA_DIR / DATASET_FILENAME

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return pd.read_csv(dataset_path)


def generate_data_quality_report(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""

    print("=" * 60)
    print("ForgeSense - Data Quality Report")
    print("=" * 60)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(df.isnull().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    print("\nData types:")
    print(df.dtypes)

    print("=" * 60)


def main() -> None:
    """Run the data ingestion pipeline."""

    print("Loading dataset...")

    df = load_raw_data()

    print("Dataset loaded successfully.")

    generate_data_quality_report(df)


if __name__ == "__main__":
    main()