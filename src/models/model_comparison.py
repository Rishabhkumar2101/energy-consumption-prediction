
from pathlib import Path

import pandas as pd

from src.models.baseline_model import train_baseline_model, prepare_features
from src.models.xgboost_model import train_xgboost_model
from src.models.evaluation import calculate_metrics


DATA_DIR = Path("data/processed")
REPORT_PATH = Path("models/model_comparison.csv")


def evaluate_model(model, df):
    X, y = prepare_features(df)
    predictions = model.predict(X)

    metrics = calculate_metrics(y, predictions)
    return metrics


def main():
    # Load the existing datasets
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Train Linear Regression
    print("Training Linear Regression...")
    linear_model = train_baseline_model(train_df)

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = train_xgboost_model(train_df)

    models = {
        "Linear Regression": linear_model,
        "XGBoost": xgb_model,
    }

    # Evaluate both models on validation data
    validation_results = {}

    print("\nValidation Results:")

    for name, model in models.items():
        metrics = evaluate_model(model, val_df)
        validation_results[name] = metrics

        print(
            f"{name}: "
            f"MAE={metrics['MAE']:.3f}, "
            f"RMSE={metrics['RMSE']:.3f}, "
            f"MAPE={metrics['MAPE']:.3f}%"
        )

    # Select the model with the lowest validation MAE
    best_name = min(
        validation_results,
        key=lambda name: validation_results[name]["MAE"],
    )

    best_model = models[best_name]

    print(f"\nSelected Model: {best_name}")

    # Evaluate only the selected model on test data
    print("\nEvaluating selected model on test data...")

    test_metrics = evaluate_model(best_model, test_df)

    print("\nFinal Test Results:")

    for metric, value in test_metrics.items():
        suffix = "%" if metric == "MAPE" else ""
        print(f"{metric}: {value:.3f}{suffix}")

    # Prepare comparison report
    report_rows = []

    for name, metrics in validation_results.items():
        report_rows.append(
            {
                "Model": name,
                "Split": "Validation",
                **metrics,
            }
        )

    report_rows.append(
        {
            "Model": best_name,
            "Split": "Test (selected model)",
            **test_metrics,
        }
    )

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_PATH, index=False)

    print(f"\nComparison report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()