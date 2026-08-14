from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from src.backtest import risk_metrics, simple_long_flat_backtest
from src.feature_engineering import feature_columns
from src.train import classification_metrics, train_baseline_model


SYMBOL = "AAPL"
SIGNAL_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MIN_MEANINGFUL_EXPOSURE = 0.10
CANDIDATE_MODELS = ("random_forest", "logistic_regression")

# These folds use an expanding training window and a fixed forward test window.
TRAIN_START_DATE = "2018-01-01"
WALK_FORWARD_FOLDS = (
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
)

FEATURE_SETS = {
    "all_features": feature_columns(),
    "returns_only": ["log_return_1d", "log_return_5d", "log_return_20d"],
    "returns_plus_volatility": ["log_return_1d", "log_return_5d", "log_return_20d", "volatility_20d"],
    "no_volume_features": [
        "log_return_1d",
        "log_return_5d",
        "log_return_20d",
        "volatility_20d",
        "close_zscore_20d",
    ],
    "zscore_features": ["close_zscore_20d", "volume_zscore_20d"],
}


def load_data() -> pd.DataFrame:
    """Load the processed AAPL feature file."""

    features_path = Path("data/processed") / f"{SYMBOL}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature file: {features_path}. Run python run_features.py first.")

    return pd.read_csv(features_path, parse_dates=["Date"])


def build_models() -> dict[str, object]:
    """Create the candidate models for walk-forward comparison."""

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
    """Train either the baseline helper or a scikit-learn style model."""

    if model_name == "logistic_regression":
        return model(X_train, y_train)

    model = clone(model)
    model.fit(X_train, y_train)
    return model


def evaluate_always_long(fold_name: str, test_data: pd.DataFrame) -> dict[str, float | str]:
    """Evaluate buy-and-hold as an always-long baseline for one fold."""

    baseline_data = test_data.copy()
    baseline_data["predicted_probability"] = 1.0
    baseline_data["signal"] = 1

    backtest = simple_long_flat_backtest(baseline_data)
    classifier_metrics = classification_metrics(
        baseline_data["target_next_day_up"],
        baseline_data["signal"],
        baseline_data["predicted_probability"],
    )
    strategy_metrics = risk_metrics(backtest["net_strategy_return"])

    return {
        "symbol": SYMBOL,
        "fold": fold_name,
        "model": "always_long",
        "feature_set": "baseline",
        "threshold": 0.0,
        "train_start": pd.NaT,
        "train_end": pd.NaT,
        "test_start": test_data["Date"].min(),
        "test_end": test_data["Date"].max(),
        "exposure": baseline_data["signal"].mean(),
        "trade_count": baseline_data["signal"].diff().abs().fillna(0).sum(),
        **classifier_metrics,
        **strategy_metrics,
    }


def evaluate_fold(model_name: str, model, feature_set_name: str, fold_name: str, train_data: pd.DataFrame, test_data: pd.DataFrame, features: list[str]) -> list[dict[str, float | str]]:
    """Train and evaluate one model on one walk-forward fold."""

    X_train = train_data[features]
    y_train = train_data["target_next_day_up"]
    X_test = test_data[features]
    y_test = test_data["target_next_day_up"]

    trained_model = fit_model(model_name, model, X_train, y_train)
    probabilities = trained_model.predict_proba(X_test)[:, 1]

    rows: list[dict[str, float | str]] = []
    for threshold in SIGNAL_THRESHOLDS:
        fold_test = test_data.copy()
        predicted_labels = (probabilities >= threshold).astype(int)
        fold_test["predicted_probability"] = probabilities
        fold_test["signal"] = predicted_labels

        backtest = simple_long_flat_backtest(fold_test)
        classifier_metrics = classification_metrics(y_test, predicted_labels, probabilities)
        strategy_metrics = risk_metrics(backtest["net_strategy_return"])

        rows.append(
            {
                "symbol": SYMBOL,
                "fold": fold_name,
                "model": model_name,
                "feature_set": feature_set_name,
                "threshold": threshold,
                "train_start": train_data["Date"].min(),
                "train_end": train_data["Date"].max(),
                "test_start": test_data["Date"].min(),
                "test_end": test_data["Date"].max(),
                "exposure": fold_test["signal"].mean(),
                "trade_count": fold_test["signal"].diff().abs().fillna(0).sum(),
                **classifier_metrics,
                **strategy_metrics,
            }
        )

    return rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate walk-forward metrics by model and threshold."""

    summary = (
        results.groupby(["model", "feature_set", "threshold"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            avg_exposure=("exposure", "mean"),
            avg_trade_count=("trade_count", "mean"),
            avg_accuracy=("accuracy", "mean"),
            avg_precision=("precision", "mean"),
            avg_roc_auc=("roc_auc", "mean"),
            avg_sharpe_ratio=("sharpe_ratio", "mean"),
            avg_sortino_ratio=("sortino_ratio", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_total_return=("total_return", "mean"),
        )
        .sort_values(["avg_sharpe_ratio", "avg_total_return"], ascending=False)
        .reset_index(drop=True)
    )
    return summary


def write_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a short markdown report for the best walk-forward configuration."""

    model_results = summary[summary["model"] != "always_long"].copy()
    meaningful = model_results[model_results["avg_exposure"] >= MIN_MEANINGFUL_EXPOSURE].copy()
    if meaningful.empty:
        meaningful = model_results.copy()

    best = meaningful.iloc[0]
    always_long = summary[summary["model"] == "always_long"].iloc[0]

    candidates = meaningful[meaningful["model"].isin(CANDIDATE_MODELS)].copy()
    best_candidate = candidates.iloc[0] if not candidates.empty else best

    feature_set_ranking = (
        meaningful[meaningful["model"].isin(CANDIDATE_MODELS)]
        .sort_values(["avg_sharpe_ratio", "avg_total_return"], ascending=False)
        .head(5)
    )

    tested_models = ", ".join(summary["model"].drop_duplicates())
    feature_lines = "\n".join(
        f"- `{row['model']}` / `{row['feature_set']}` at threshold `{row['threshold']:.2f}`: "
        f"Sharpe `{row['avg_sharpe_ratio']:.4f}`, return `{row['avg_total_return']:.2%}`, exposure `{row['avg_exposure']:.2%}`."
        for _, row in feature_set_ranking.iterrows()
    )

    report = f"""# {SYMBOL} Walk-Forward Summary

Tested models: {tested_models}

## Best Meaningful Configuration

This uses an average exposure filter of at least {MIN_MEANINGFUL_EXPOSURE:.0%}.

- Best model: `{best["model"]}`
- Feature set: `{best["feature_set"]}`
- Threshold: `{best["threshold"]:.2f}`
- Average Sharpe Ratio: `{best["avg_sharpe_ratio"]:.4f}`
- Average Sortino Ratio: `{best["avg_sortino_ratio"]:.4f}`
- Average max drawdown: `{best["avg_max_drawdown"]:.2%}`
- Average total return: `{best["avg_total_return"]:.2%}`
- Average exposure: `{best["avg_exposure"]:.2%}`

## Always-Long Baseline

- Average Sharpe Ratio: `{always_long["avg_sharpe_ratio"]:.4f}`
- Average Sortino Ratio: `{always_long["avg_sortino_ratio"]:.4f}`
- Average max drawdown: `{always_long["avg_max_drawdown"]:.2%}`
- Average total return: `{always_long["avg_total_return"]:.2%}`
- Average exposure: `{always_long["avg_exposure"]:.2%}`

## Feature Set Ranking

{feature_lines}

## AAPL Candidate

Use `{best_candidate["model"]}` with `{best_candidate["feature_set"]}` at threshold `{best_candidate["threshold"]:.2f}` as the AAPL candidate before expanding to all five stocks.
It is ahead of the always-long baseline by `{best_candidate["avg_sharpe_ratio"] - always_long["avg_sharpe_ratio"]:.4f}` average Sharpe points.

## Interpretation

This walk-forward check is stricter than a single train/test split because each fold tests on unseen future data.
The always-long baseline matters because high-exposure strategies can look strong simply by staying invested through a rising market.
"""

    output_path.write_text(report)


def main() -> None:
    """Run walk-forward model comparison for AAPL."""

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    rows: list[dict[str, float | str]] = []
    models = build_models()

    for fold_name, test_end in WALK_FORWARD_FOLDS:
        train_data = data[(data["Date"] >= TRAIN_START_DATE) & (data["Date"] < fold_name)].copy()
        test_data = data[(data["Date"] >= fold_name) & (data["Date"] < test_end)].copy()

        if train_data.empty or test_data.empty:
            print(f"Skipping fold {fold_name} because train or test data is empty.")
            continue

        rows.append(evaluate_always_long(fold_name, test_data))

        for model_name, model in models.items():
            for feature_set_name, feature_set_columns in FEATURE_SETS.items():
                if feature_set_name != "all_features" and model_name not in CANDIDATE_MODELS:
                    continue

                rows.extend(evaluate_fold(model_name, model, feature_set_name, fold_name, train_data, test_data, feature_set_columns))

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No walk-forward results were produced. Check the fold dates and processed data.")

    results_path = reports_dir / f"{SYMBOL}_walk_forward_results.csv"
    summary_path = reports_dir / f"{SYMBOL}_walk_forward_summary.md"

    summary = summarize_results(results)
    results.to_csv(results_path, index=False)
    summary.to_csv(reports_dir / f"{SYMBOL}_walk_forward_summary_table.csv", index=False)
    write_summary(summary, summary_path)

    print(f"Saved walk-forward results to {results_path}")
    print(f"Saved walk-forward summary to {summary_path}")
    print("\nTop walk-forward configurations")
    print(
        summary[
            [
                "model",
                "feature_set",
                "threshold",
                "folds",
                "avg_exposure",
                "avg_accuracy",
                "avg_roc_auc",
                "avg_sharpe_ratio",
                "avg_sortino_ratio",
                "avg_max_drawdown",
                "avg_total_return",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
