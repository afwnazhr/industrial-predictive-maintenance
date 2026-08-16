"""Feature engineering for the ForgeSense predictive maintenance dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai4i2020.csv"
)


TARGET_COLUMN = "Machine failure"


BASE_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


ENGINEERED_FEATURES = [
    "Temperature difference [K]",
    "Temperature ratio",
    "Mechanical power [W]",
    "Mechanical power [kW]",
]


def load_data() -> pd.DataFrame:
    """Load the raw dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


def create_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create physically meaningful machine features."""

    data = df.copy()

    # ---------------------------------------------------------
    # 1. Temperature difference
    # ---------------------------------------------------------

    data["Temperature difference [K]"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    # ---------------------------------------------------------
    # 2. Temperature ratio
    # ---------------------------------------------------------

    data["Temperature ratio"] = (
        data["Process temperature [K]"]
        / data["Air temperature [K]"]
    )

    # ---------------------------------------------------------
    # 3. Mechanical power
    #
    # P = torque × angular velocity
    # angular velocity = 2π × RPM / 60
    # ---------------------------------------------------------

    angular_velocity = (
        2 * np.pi * data["Rotational speed [rpm]"] / 60
    )

    data["Mechanical power [W]"] = (
        data["Torque [Nm]"] * angular_velocity
    )

    # ---------------------------------------------------------
    # 4. Mechanical power in kilowatts
    # ---------------------------------------------------------

    data["Mechanical power [kW]"] = (
        data["Mechanical power [W]"] / 1000
    )

    return data


def main() -> None:
    """Run feature engineering and display the results."""

    df = load_data()

    engineered_df = create_engineered_features(df)

    print("=" * 60)
    print("FORGESENSE FEATURE ENGINEERING")
    print("=" * 60)

    print("\nBase features:")
    for feature in BASE_FEATURES:
        print(f"  ✓ {feature}")

    print("\nEngineered features:")
    for feature in ENGINEERED_FEATURES:
        print(f"  + {feature}")

    print("\nEngineered feature summary:")
    print(
        engineered_df[ENGINEERED_FEATURES].describe().round(4)
    )

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()