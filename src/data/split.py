"""Train/test split for the ForgeSense predictive maintenance dataset."""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

TARGET_COLUMN = "Machine failure"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the raw dataset."""

    return pd.read_csv(RAW_DATA_PATH)


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into stratified training and test sets."""

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def main() -> None:
    """Run the train/test split."""

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print("=" * 60)
    print("FORGESENSE TRAIN / TEST SPLIT")
    print("=" * 60)

    print("\nDataset:")
    print(f"  Total samples: {len(df)}")

    print("\nTraining set:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")

    print("\nTest set:")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    print("\nTraining target distribution:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True).mul(100).round(2))

    print("\nTest target distribution:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True).mul(100).round(2))

    print("\nTrain/test split completed successfully.")


if __name__ == "__main__":
    main()