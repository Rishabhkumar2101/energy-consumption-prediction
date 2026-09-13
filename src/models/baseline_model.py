from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


TARGET_COLUMN = "Appliances"
MODEL_PATH = Path("models/baseline_model.joblib")


def prepare_features(df: pd.DataFrame):
    """Prepare numerical features and target."""

    df = df.copy()

    X = df.drop(columns=[TARGET_COLUMN, "date"])
    y = df[TARGET_COLUMN]

    return X, y


def train_baseline_model(
    train_df: pd.DataFrame,
) -> LinearRegression:
    """Train Linear Regression baseline model."""

    X_train, y_train = prepare_features(train_df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    df: pd.DataFrame,
) -> dict:
    """Evaluate model using MAE and RMSE."""

    X, y = prepare_features(df)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions) ** 0.5

    return {
        "MAE": mae,
        "RMSE": rmse,
    }


def save_model(
    model,
    output_path: Path = MODEL_PATH,
) -> None:
    """Save trained model to disk."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, output_path)