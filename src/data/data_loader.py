from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/energy_consumption.csv")


def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load raw energy consumption data from CSV."""

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    return df