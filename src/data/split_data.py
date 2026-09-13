from pathlib import Path

import pandas as pd


FEATURE_DATA_PATH = Path(
    "data/processed/energy_consumption_features.csv"
)


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """Split time-series data chronologically."""

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be less than 1."
        )

    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """Save train, validation and test datasets."""

    output_dir = Path("data/processed")

    train_df.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    val_df.to_csv(
        output_dir / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False,
    )