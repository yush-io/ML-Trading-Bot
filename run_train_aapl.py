from pathlib import Path

import pandas as pd

from src.backtest import risk_metrics, simple_long_flat_backtest
from src.feature_engineering import feature_columns
from src.train import classification_metrics, train_baseline_model


# Start with one stock while we validate the pipeline.
SYMBOL = "AAPL"

# Use older data for training and newer data for testing.
TRAIN_END_DATE = "2023-01-01"

# The trading rule only goes long when the model is at least 55% confident.
SIGNAL_THRESHOLD = 0.55


def main() -> None:
    """Train and backtest the first AAPL-only baseline model."""

    # Load the processed feature file created by run_features.py.
    features_path = Path("data/processed") / f"{SYMBOL}_features.csv"

    # Reports are saved here so results do not mix with source data.
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Give a clear error if someone skipped the feature-building step.
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature file: {features_path}. Run python run_features.py first.")

    # Read the processed data and parse Date as an actual datetime column.
    df = pd.read_csv(features_path, parse_dates=["Date"])

    # Use the same feature list defined in src/feature_engineering.py.
    features = feature_columns()

    # Split by date so the model learns from the past and tests on the future.
    train_df = df[df["Date"] < TRAIN_END_DATE].copy()
    test_df = df[df["Date"] >= TRAIN_END_DATE].copy()

    # X columns are model inputs; y is the answer the model learns to predict.
    X_train = train_df[features]
    y_train = train_df["target_next_day_up"]
    X_test = test_df[features]
    y_test = test_df["target_next_day_up"]

    # Train the baseline classifier on the training period.
    model = train_baseline_model(X_train, y_train)

    # Predict the probability of class 1, meaning "next day up".
    probabilities = model.predict_proba(X_test)[:, 1]

    # A 0.50 cutoff is used for basic classifier diagnostics.
    predicted_labels = (probabilities >= 0.5).astype(int)

    # Store predictions beside the test data so the backtest can use them.
    test_df["predicted_probability"] = probabilities

    # Convert model confidence into a trading signal.
    test_df["signal"] = (test_df["predicted_probability"] >= SIGNAL_THRESHOLD).astype(int)

    # Classification metrics describe prediction quality.
    classification = classification_metrics(y_test, predicted_labels, probabilities)

    # Backtest metrics describe whether predictions made money after costs.
    backtest = simple_long_flat_backtest(test_df)
    strategy_metrics = risk_metrics(backtest["net_strategy_return"])

    # Buy-and-hold is the benchmark for comparison.
    benchmark_metrics = risk_metrics(backtest["log_return_1d"])

    # Save the full backtest table for later charts and inspection.
    output_path = reports_dir / f"{SYMBOL}_baseline_backtest.csv"
    backtest.to_csv(output_path, index=False)

    # Print a readable summary to the terminal.
    print(f"{SYMBOL} baseline model")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Signal threshold: {SIGNAL_THRESHOLD}")
    print("\nClassification diagnostics")
    for name, value in classification.items():
        print(f"{name}: {value:.4f}")

    print("\nStrategy risk metrics")
    for name, value in strategy_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nBuy-and-hold risk metrics")
    for name, value in benchmark_metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nSaved backtest results to {output_path}")


# This lets the file run as a script with: python run_train_aapl.py
if __name__ == "__main__":
    main()
