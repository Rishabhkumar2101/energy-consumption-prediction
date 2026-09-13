import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_naive_model(
    df: pd.DataFrame,
    lag_column: str = "lag_1h",
) -> dict:
    """Evaluate a naive forecasting baseline using a lag feature."""

    actual = df["Appliances"]
    predictions = df[lag_column]

    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5

    return {
        "MAE": mae,
        "RMSE": rmse,
    }