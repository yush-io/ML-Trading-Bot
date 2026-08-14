from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from src.backtest import risk_metrics, simple_long_flat_backtest
from src.config import ProjectConfig
from src.train import classification_metrics


SELECTED_MODEL_NAME = "random_forest"
SELECTED_FEATURE_SET_NAME = "returns_plus_volatility"
SELECTED_FEATURES = ["log_return_1d", "log_return_5d", "log_return_20d", "volatility_20d"]
SIGNAL_THRESHOLD = 0.45

TRAIN_START_DATE = "2018-01-01"
WALK_FORWARD_FOLDS = (
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
)


def load_symbol_data(symbol: str) -> pd.DataFrame:
    """Load one processed feature file."""

    features_path = Path("data/processed") / f"{symbol}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature file: {features_path}. Run python run_features.py first.")

    return pd.read_csv(features_path, parse_dates=["Date"])


def build_selected_model() -> RandomForestClassifier:
    """Return the selected AAPL candidate model."""

    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


def fit_model(model: RandomForestClassifier, train_data: pd.DataFrame):
    """Fit a fresh model instance for one symbol/fold."""

    fitted_model = clone(model)
    fitted_model.fit(train_data[SELECTED_FEATURES], train_data["target_next_day_up"])
    return fitted_model


def evaluate_strategy(symbol: str, fold_name: str, train_data: pd.DataFrame, test_data: pd.DataFrame, model: RandomForestClassifier) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Evaluate the selected model strategy for one symbol/fold."""

    fitted_model = fit_model(model, train_data)
    probabilities = fitted_model.predict_proba(test_data[SELECTED_FEATURES])[:, 1]
    predicted_labels = (probabilities >= SIGNAL_THRESHOLD).astype(int)

    strategy_data = test_data.copy()
    strategy_data["predicted_probability"] = probabilities
    strategy_data["signal"] = predicted_labels

    backtest = simple_long_flat_backtest(strategy_data)
    classifier_metrics = classification_metrics(test_data["target_next_day_up"], predicted_labels, probabilities)
    strategy_metrics = risk_metrics(backtest["net_strategy_return"])

    metrics = {
        "symbol": symbol,
        "fold": fold_name,
        "strategy": "model",
        "model": SELECTED_MODEL_NAME,
        "feature_set": SELECTED_FEATURE_SET_NAME,
        "threshold": SIGNAL_THRESHOLD,
        "train_start": train_data["Date"].min(),
        "train_end": train_data["Date"].max(),
        "test_start": test_data["Date"].min(),
        "test_end": test_data["Date"].max(),
        "exposure": strategy_data["signal"].mean(),
        "trade_count": strategy_data["signal"].diff().abs().fillna(0).sum(),
        **classifier_metrics,
        **strategy_metrics,
    }

    daily_returns = backtest[["Date", "net_strategy_return"]].copy()
    daily_returns["symbol"] = symbol
    daily_returns["fold"] = fold_name
    daily_returns["strategy"] = "model"
    return metrics, daily_returns


def evaluate_always_long(symbol: str, fold_name: str, test_data: pd.DataFrame) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Evaluate buy-and-hold as an always-long baseline."""

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

    metrics = {
        "symbol": symbol,
        "fold": fold_name,
        "strategy": "always_long",
        "model": "baseline",
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

    daily_returns = backtest[["Date", "net_strategy_return"]].copy()
    daily_returns["symbol"] = symbol
    daily_returns["fold"] = fold_name
    daily_returns["strategy"] = "always_long"
    return metrics, daily_returns


def summarize_symbol_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics by symbol and strategy."""

    return (
        results.groupby(["symbol", "strategy", "model", "feature_set", "threshold"], as_index=False)
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
        .sort_values(["symbol", "strategy"])
        .reset_index(drop=True)
    )


def summarize_portfolio(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Create an equal-weight portfolio summary from daily symbol returns."""

    rows = []
    for strategy, strategy_returns in daily_returns.groupby("strategy"):
        portfolio_returns = strategy_returns.groupby("Date")["net_strategy_return"].mean().sort_index()
        metrics = risk_metrics(portfolio_returns)
        rows.append(
            {
                "strategy": strategy,
                "symbols": strategy_returns["symbol"].nunique(),
                "start": portfolio_returns.index.min(),
                "end": portfolio_returns.index.max(),
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def write_summary(symbol_summary: pd.DataFrame, portfolio_summary: pd.DataFrame, output_path: Path) -> None:
    """Write a concise markdown summary for the full universe test."""

    model_portfolio = portfolio_summary[portfolio_summary["strategy"] == "model"].iloc[0]
    baseline_portfolio = portfolio_summary[portfolio_summary["strategy"] == "always_long"].iloc[0]

    model_rows = symbol_summary[symbol_summary["strategy"] == "model"].set_index("symbol")
    baseline_rows = symbol_summary[symbol_summary["strategy"] == "always_long"].set_index("symbol")

    symbol_lines = []
    for symbol in model_rows.index:
        model = model_rows.loc[symbol]
        baseline = baseline_rows.loc[symbol]
        symbol_lines.append(
            f"- `{symbol}`: model Sharpe `{model['avg_sharpe_ratio']:.4f}` vs hold `{baseline['avg_sharpe_ratio']:.4f}`, "
            f"return `{model['avg_total_return']:.2%}` vs hold `{baseline['avg_total_return']:.2%}`, "
            f"drawdown `{model['avg_max_drawdown']:.2%}` vs hold `{baseline['avg_max_drawdown']:.2%}`."
        )

    symbol_text = "\n".join(symbol_lines)
    sharpe_delta = model_portfolio["sharpe_ratio"] - baseline_portfolio["sharpe_ratio"]
    return_delta = model_portfolio["total_return"] - baseline_portfolio["total_return"]

    report = f"""# Universe Walk-Forward Summary

Selected setup:

- Model: `{SELECTED_MODEL_NAME}`
- Feature set: `{SELECTED_FEATURE_SET_NAME}`
- Threshold: `{SIGNAL_THRESHOLD:.2f}`

## Equal-Weight Portfolio

- Model Sharpe: `{model_portfolio["sharpe_ratio"]:.4f}`
- Always-long Sharpe: `{baseline_portfolio["sharpe_ratio"]:.4f}`
- Sharpe difference: `{sharpe_delta:.4f}`
- Model total return: `{model_portfolio["total_return"]:.2%}`
- Always-long total return: `{baseline_portfolio["total_return"]:.2%}`
- Return difference: `{return_delta:.2%}`
- Model max drawdown: `{model_portfolio["max_drawdown"]:.2%}`
- Always-long max drawdown: `{baseline_portfolio["max_drawdown"]:.2%}`

## Per-Symbol Results

{symbol_text}

## Interpretation

This applies the AAPL-selected strategy to the full five-stock universe without retuning per ticker.
That makes this a generalization test rather than another optimization pass.
"""

    output_path.write_text(report)


def main() -> None:
    """Run the selected model setup across the configured stock universe."""

    config = ProjectConfig()
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_template = build_selected_model()
    metric_rows: list[dict[str, float | str]] = []
    daily_return_frames: list[pd.DataFrame] = []

    for symbol in config.symbols:
        data = load_symbol_data(symbol)

        for fold_name, test_end in WALK_FORWARD_FOLDS:
            train_data = data[(data["Date"] >= TRAIN_START_DATE) & (data["Date"] < fold_name)].copy()
            test_data = data[(data["Date"] >= fold_name) & (data["Date"] < test_end)].copy()

            if train_data.empty or test_data.empty:
                print(f"Skipping {symbol} fold {fold_name} because train or test data is empty.")
                continue

            model_metrics, model_daily_returns = evaluate_strategy(symbol, fold_name, train_data, test_data, model_template)
            baseline_metrics, baseline_daily_returns = evaluate_always_long(symbol, fold_name, test_data)

            metric_rows.extend([model_metrics, baseline_metrics])
            daily_return_frames.extend([model_daily_returns, baseline_daily_returns])

    results = pd.DataFrame(metric_rows)
    if results.empty:
        raise RuntimeError("No universe walk-forward results were produced.")

    daily_returns = pd.concat(daily_return_frames, ignore_index=True)
    symbol_summary = summarize_symbol_results(results)
    portfolio_summary = summarize_portfolio(daily_returns)

    results_path = reports_dir / "universe_walk_forward_results.csv"
    symbol_summary_path = reports_dir / "universe_walk_forward_symbol_summary.csv"
    portfolio_summary_path = reports_dir / "universe_walk_forward_portfolio_summary.csv"
    daily_returns_path = reports_dir / "universe_walk_forward_daily_returns.csv"
    summary_path = reports_dir / "universe_walk_forward_summary.md"

    results.to_csv(results_path, index=False)
    symbol_summary.to_csv(symbol_summary_path, index=False)
    portfolio_summary.to_csv(portfolio_summary_path, index=False)
    daily_returns.to_csv(daily_returns_path, index=False)
    write_summary(symbol_summary, portfolio_summary, summary_path)

    print(f"Saved fold results to {results_path}")
    print(f"Saved symbol summary to {symbol_summary_path}")
    print(f"Saved portfolio summary to {portfolio_summary_path}")
    print(f"Saved daily returns to {daily_returns_path}")
    print(f"Saved markdown summary to {summary_path}")
    print("\nPortfolio summary")
    print(portfolio_summary)
    print("\nPer-symbol summary")
    print(symbol_summary[["symbol", "strategy", "avg_sharpe_ratio", "avg_max_drawdown", "avg_total_return", "avg_exposure"]])


if __name__ == "__main__":
    main()
