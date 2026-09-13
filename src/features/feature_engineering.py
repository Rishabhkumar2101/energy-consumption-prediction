from pathlib import Path

import pandas as pd


FEATURE_DATA_PATH = Path(
    "data/processed/energy_consumption_features.csv"
)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based and lag features for energy forecasting."""

    df = df.copy()

    # Ensure chronological order
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Time-based features
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical time features
    df["hour_sin"] = (
        __import__("numpy").sin(2 * __import__("numpy").pi * df["hour"] / 24)
    )
    df["hour_cos"] = (
        __import__("numpy").cos(2 * __import__("numpy").pi * df["hour"] / 24)
    )

    df["day_sin"] = (
        __import__("numpy").sin(
            2 * __import__("numpy").pi * df["day_of_week"] / 7
        )
    )
    df["day_cos"] = (
        __import__("numpy").cos(
            2 * __import__("numpy").pi * df["day_of_week"] / 7
        )
    )

    # Lag features
    # Dataset frequency = 10 minutes
    # 1 hour = 6 rows
    # 24 hours = 144 rows
    # 7 days = 1008 rows

    df["lag_1h"] = df["Appliances"].shift(6)
    df["lag_24h"] = df["Appliances"].shift(144)
    df["lag_7d"] = df["Appliances"].shift(1008)

    # Rolling feature
    df["rolling_mean_1h"] = (
        df["Appliances"]
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    # Remove rows created with NaN lag/rolling values
    df = df.dropna().reset_index(drop=True)

    return df


def save_feature_data(
    df: pd.DataFrame,
    output_path: Path = FEATURE_DATA_PATH,
) -> None:
    """Save feature-engineered data."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)