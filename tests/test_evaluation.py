
import numpy as np

from src.models.evaluation import calculate_metrics


def test_calculate_metrics_returns_expected_values():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])

    metrics = calculate_metrics(y_true, y_pred)

    assert np.isclose(metrics["MAE"], 7 / 3)
    assert np.isclose(metrics["RMSE"], np.sqrt(17 / 3))
    assert "MAPE" in metrics


def test_calculate_metrics_returns_zero_for_perfect_predictions():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 20, 30])

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["MAPE"] == 0