from pathlib import Path

import pandas as pd


PROCESSED_DATA_PATH = Path("data/processed/energy_consumption_processed.csv")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare raw energy consumption data."""

    df = df.copy()

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort data chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def save_processed_data(
    df: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Save processed data to the processed data directory."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)