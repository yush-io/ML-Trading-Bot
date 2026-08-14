from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from src.backtest import risk_metrics, simple_long_flat_backtest
from src.feature_engineering import feature_columns
from src.train import classification_metrics, train_baseline_model


# Keep model comparison scoped to AAPL before expanding to the full stock universe.
SYMBOL = "AAPL"

# Match the same train/test split used by the AAPL baseline scripts.
TRAIN_END_DATE = "2023-01-01"

# Test a wider set of trading thresholds across every model.
SIGNAL_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

# Ignore strategies that barely trade when choosing the "best" configuration.
MIN_MEANINGFUL_EXPOSURE = 0.10


def load_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load AAPL features and split them into train/test periods."""

    features_path = Path("data/processed") / f"{SYMBOL}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature file: {features_path}. Run python run_features.py first.")

    data = pd.read_csv(features_path, parse_dates=["Date"])
    train_data = data[data["Date"] < TRAIN_END_DATE].copy()
    test_data = data[data["Date"] >= TRAIN_END_DATE].copy()
    return train_data, test_data


def build_models() -> dict[str, object]:
    """Create the models we want to compare on the same dataset."""

    models = {
        "logistic_regression": train_baseline_model,
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except Exception as error:
        print(f"Skipping XGBoost because it could not be imported: {error}")

    return models


def fit_model(model_name: str, model, X_train: pd.DataFrame, y_train: pd.Series):
    """Train either the existing baseline function or a scikit-learn style model."""

    if model_name == "logistic_regression":
        return model(X_train, y_train)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model_name: str, probabilities, test_data: pd.DataFrame) -> list[dict[str, float | str]]:
    """Evaluate one model across all signal thresholds."""

    rows = []
    y_test = test_data["target_next_day_up"]

    for threshold in SIGNAL_THRESHOLDS:
        threshold_data = test_data.copy()
        predicted_labels = (probabilities >= threshold).astype(int)
        threshold_data["predicted_probability"] = probabilities
        threshold_data["signal"] = predicted_labels

        backtest = simple_long_flat_backtest(threshold_data)
        classifier_metrics = classification_metrics(y_test, predicted_labels, probabilities)
        strategy_metrics = risk_metrics(backtest["net_strategy_return"])

        rows.append(
            {
                "symbol": SYMBOL,
                "model": model_name,
                "threshold": threshold,
                "exposure": threshold_data["signal"].mean(),
                "trade_count": threshold_data["signal"].diff().abs().fillna(0).sum(),
                **classifier_metrics,
                **strategy_metrics,
            }
        )

    return rows


def write_model_summary(results: pd.DataFrame, output_path: Path) -> None:
    """Write a short model-comparison summary for quick interpretation."""

    meaningful_results = results[results["exposure"] >= MIN_MEANINGFUL_EXPOSURE].copy()
    if meaningful_results.empty:
        meaningful_results = results.copy()

    best_sharpe = meaningful_results.loc[meaningful_results["sharpe_ratio"].idxmax()]
    best_return = meaningful_results.loc[meaningful_results["total_return"].idxmax()]
    best_drawdown = meaningful_results.loc[meaningful_results["max_drawdown"].idxmax()]

    tested_models = ", ".join(results["model"].drop_duplicates())

    summary = f"""# {SYMBOL} Model Comparison Summary

Tested models: {tested_models}

## Best Meaningful Results

These picks only consider rows with at least {MIN_MEANINGFUL_EXPOSURE:.0%} exposure so a model does not look good simply because it almost never trades.

- Best Sharpe Ratio: `{best_sharpe["model"]}` at threshold `{best_sharpe["threshold"]:.2f}` with Sharpe `{best_sharpe["sharpe_ratio"]:.4f}`.
- Best total return: `{best_return["model"]}` at threshold `{best_return["threshold"]:.2f}` with total return `{best_return["total_return"]:.2%}`.
- Lowest drawdown: `{best_drawdown["model"]}` at threshold `{best_drawdown["threshold"]:.2f}` with max drawdown `{best_drawdown["max_drawdown"]:.2%}`.

## Recommendation

Use `{best_sharpe["model"]}` with threshold `{best_sharpe["threshold"]:.2f}` as the next AAPL candidate because it has the strongest risk-adjusted result among strategies that traded meaningfully.

## Caveat

This comparison is still AAPL-only. A setup that works best on AAPL may not generalize to `NVDA`, `AMZN`, `MSFT`, and `META`, so the next validation step should test the same model setup across the full five-stock universe.
"""

    output_path.write_text(summary)


def main() -> None:
    """Compare several model types on the AAPL prediction/backtest task."""

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = load_train_test_data()
    features = feature_columns()

    X_train = train_data[features]
    y_train = train_data["target_next_day_up"]
    X_test = test_data[features]

    rows = []
    for model_name, model in build_models().items():
        trained_model = fit_model(model_name, model, X_train, y_train)
        probabilities = trained_model.predict_proba(X_test)[:, 1]
        rows.extend(evaluate_model(model_name, probabilities, test_data))

    results = pd.DataFrame(rows)
    results_path = reports_dir / f"{SYMBOL}_model_comparison.csv"
    summary_path = reports_dir / f"{SYMBOL}_model_comparison_summary.md"

    results = results.sort_values(["sharpe_ratio", "total_return"], ascending=False).reset_index(drop=True)
    results.to_csv(results_path, index=False)
    write_model_summary(results, summary_path)

    print(f"Saved model comparison results to {results_path}")
    print(f"Saved model comparison summary to {summary_path}")
    print("\nModel comparison")
    print(
        results[
            [
                "model",
                "threshold",
                "exposure",
                "trade_count",
                "accuracy",
                "roc_auc",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "total_return",
            ]
        ].sort_values("sharpe_ratio", ascending=False)
    )


# This lets the file run as a script with: python run_compare_aapl_models.py
if __name__ == "__main__":
    main()
