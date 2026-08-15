from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_raw_data_path(filename: str) -> Path:
    """Return the path to a file in the raw data directory."""
    return RAW_DATA_DIR / filename


if __name__ == "__main__":
    print(f"Raw data directory: {RAW_DATA_DIR}")