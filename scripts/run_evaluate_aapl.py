from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.backtest import risk_metrics, simple_long_flat_backtest
from src.feature_engineering import feature_columns
from src.train import classification_metrics, train_baseline_model


# Keep the evaluation focused on the first validated ticker.
SYMBOL = "AAPL"

# Match the training split used in run_train_aapl.py.
TRAIN_END_DATE = "2023-01-01"

# Test several confidence cutoffs for turning model probabilities into trades.
SIGNAL_THRESHOLDS = (0.50, 0.55, 0.60, 0.65)


def format_percent(value: float) -> str:
    """Format decimal returns and exposures as readable percentages."""

    return f"{value:.2%}"


def load_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load AAPL features and split them into train/test periods."""

    features_path = Path("data/processed") / f"{SYMBOL}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature file: {features_path}. Run python -m scripts.run_features first.")

    data = pd.read_csv(features_path, parse_dates=["Date"])
    train_data = data[data["Date"] < TRAIN_END_DATE].copy()
    test_data = data[data["Date"] >= TRAIN_END_DATE].copy()
    return train_data, test_data


def evaluate_thresholds(test_data: pd.DataFrame, probabilities) -> pd.DataFrame:
    """Backtest multiple signal thresholds and return one metrics row per threshold."""

    rows = []
    y_test = test_data["target_next_day_up"]

    for threshold in SIGNAL_THRESHOLDS:
        threshold_data = test_data.copy()
        predicted_labels = (probabilities >= threshold).astype(int)
        threshold_data["predicted_probability"] = probabilities
        threshold_data["signal"] = predicted_labels

        backtest = simple_long_flat_backtest(threshold_data)
        strategy_metrics = risk_metrics(backtest["net_strategy_return"])
        classifier_metrics = classification_metrics(y_test, predicted_labels, probabilities)

        rows.append(
            {
                "symbol": SYMBOL,
                "threshold": threshold,
                "exposure": threshold_data["signal"].mean(),
                "trade_count": threshold_data["signal"].diff().abs().fillna(0).sum(),
                **classifier_metrics,
                **strategy_metrics,
            }
        )

    return pd.DataFrame(rows)


def save_signal_exposure_plot(backtest: pd.DataFrame, output_path: Path) -> None:
    """Plot when the strategy is invested versus flat."""

    plt.figure(figsize=(12, 3))
    plt.step(backtest["Date"], backtest["signal"], where="post")
    plt.title(f"{SYMBOL} Signal Exposure")
    plt.xlabel("Date")
    plt.ylabel("Signal")
    plt.yticks([0, 1], ["Flat", "Long"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(y_true, predicted_labels, output_path: Path) -> None:
    """Plot a simple confusion matrix for the baseline 0.55 trading threshold."""

    matrix = confusion_matrix(y_true, predicted_labels, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"{SYMBOL} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Down/Flat", "Up"])
    plt.yticks([0, 1], ["Down/Flat", "Up"])

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(column_index, row_index, matrix[row_index, column_index], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_evaluation_summary(threshold_metrics: pd.DataFrame, output_path: Path) -> None:
    """Save a short written interpretation of the threshold test."""

    best_sharpe = threshold_metrics.loc[threshold_metrics["sharpe_ratio"].idxmax()]
    best_return = threshold_metrics.loc[threshold_metrics["total_return"].idxmax()]
    lowest_drawdown = threshold_metrics.loc[threshold_metrics["max_drawdown"].idxmax()]

    meaningful_thresholds = threshold_metrics[threshold_metrics["exposure"] >= 0.10]
    if meaningful_thresholds.empty:
        recommended = best_sharpe
        recommendation_reason = "No threshold had at least 10% exposure, so the best Sharpe threshold is the fallback."
    else:
        recommended = meaningful_thresholds.loc[meaningful_thresholds["sharpe_ratio"].idxmax()]
        recommendation_reason = "This threshold had the best Sharpe Ratio among thresholds with at least 10% market exposure."

    summary = f"""# {SYMBOL} Evaluation Summary

## Best Thresholds

- Best Sharpe Ratio: `{best_sharpe["threshold"]:.2f}` with Sharpe `{best_sharpe["sharpe_ratio"]:.4f}` and exposure `{format_percent(best_sharpe["exposure"])}`.
- Best total return: `{best_return["threshold"]:.2f}` with total return `{format_percent(best_return["total_return"])}` and exposure `{format_percent(best_return["exposure"])}`.
- Lowest drawdown: `{lowest_drawdown["threshold"]:.2f}` with max drawdown `{format_percent(lowest_drawdown["max_drawdown"])}` and exposure `{format_percent(lowest_drawdown["exposure"])}`.

## Recommendation

Use threshold `{recommended["threshold"]:.2f}` for the next AAPL baseline comparison.

{recommendation_reason}

## Interpretation

The threshold test shows that model confidence cutoffs change the strategy's behavior a lot. Lower thresholds trade more often and capture more upside, while higher thresholds trade less and can look less risky because they spend more time in cash.

Be careful with thresholds that barely trade. A low drawdown is not very meaningful if the strategy is almost always flat.
"""

    output_path.write_text(summary)


def main() -> None:
    """Run AAPL model evaluation beyond the first fixed-threshold backtest."""

    reports_dir = Path("reports/aapl")
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = load_train_test_data()
    features = feature_columns()

    X_train = train_data[features]
    y_train = train_data["target_next_day_up"]
    X_test = test_data[features]
    y_test = test_data["target_next_day_up"]

    model = train_baseline_model(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    threshold_metrics = evaluate_thresholds(test_data, probabilities)
    threshold_metrics_path = reports_dir / f"{SYMBOL}_threshold_metrics.csv"
    threshold_metrics.to_csv(threshold_metrics_path, index=False)
    summary_path = reports_dir / f"{SYMBOL}_evaluation_summary.md"
    write_evaluation_summary(threshold_metrics, summary_path)

    baseline_data = test_data.copy()
    baseline_data["predicted_probability"] = probabilities
    baseline_data["signal"] = (baseline_data["predicted_probability"] >= 0.55).astype(int)
    baseline_backtest = simple_long_flat_backtest(baseline_data)

    exposure_path = figures_dir / f"{SYMBOL}_signal_exposure.png"
    confusion_matrix_path = figures_dir / f"{SYMBOL}_confusion_matrix.png"

    save_signal_exposure_plot(baseline_backtest, exposure_path)
    save_confusion_matrix_plot(y_test, baseline_data["signal"], confusion_matrix_path)

    print(f"Saved threshold metrics to {threshold_metrics_path}")
    print(f"Saved evaluation summary to {summary_path}")
    print(f"Saved signal exposure plot to {exposure_path}")
    print(f"Saved confusion matrix plot to {confusion_matrix_path}")
    print("\nThreshold comparison")
    print(threshold_metrics[["threshold", "exposure", "trade_count", "sharpe_ratio", "max_drawdown", "total_return"]])


# This lets the file run as a module with: python -m scripts.run_evaluate_aapl
if __name__ == "__main__":
    main()
