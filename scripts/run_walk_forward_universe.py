from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from src.backtest import position_sized_backtest, risk_metrics, simple_long_flat_backtest
from src.config import ProjectConfig
from src.train import classification_metrics


SELECTED_MODEL_NAME = "random_forest"
SELECTED_FEATURE_SET_NAME = "returns_only"
SELECTED_FEATURES = ["log_return_1d", "log_return_5d", "log_return_20d"]
VOLATILITY_COLUMN = "volatility_20d"
VOLATILITY_ADJUSTED_VARIANTS = {
    "volatility_adjusted_full": {"gross_exposure": 1.00, "max_weight": 0.40},
    "volatility_adjusted_75": {"gross_exposure": 0.75, "max_weight": 0.30},
    "volatility_adjusted_50": {"gross_exposure": 0.50, "max_weight": 0.25},
    "volatility_adjusted_diversified_full": {"gross_exposure": 1.00, "max_weight": 0.20},
    "volatility_adjusted_diversified_75": {"gross_exposure": 0.75, "max_weight": 0.15},
    "volatility_adjusted_diversified_50": {"gross_exposure": 0.50, "max_weight": 0.10},
}
FEE_BPS = 5.0
SIGNAL_THRESHOLDS_BY_SYMBOL = {
    "AAPL": 0.45,
    "AMZN": 0.55,
    "META": 0.60,
    "MSFT": 0.40,
    "NVDA": 0.45,
}
POSITION_SIZE_BUCKETS = {
    "sized_balanced": (
        (0.00, 0.25),
        (0.05, 0.50),
        (0.10, 0.75),
        (0.15, 1.00),
    ),
    "sized_aggressive": (
        (0.00, 0.50),
        (0.05, 0.75),
        (0.10, 1.00),
    ),
    "sized_near_full": (
        (0.00, 0.75),
        (0.05, 1.00),
    ),
}

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
        raise FileNotFoundError(f"Missing feature file: {features_path}. Run python -m scripts.run_features first.")

    return pd.read_csv(features_path, parse_dates=["Date"])


def build_selected_model() -> RandomForestClassifier:
    """Return the optimized shared universe model."""

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


def bucketed_position_size(probabilities, threshold: float, buckets: tuple[tuple[float, float], ...]) -> pd.Series:
    """Convert model confidence into coarse fractional position sizes."""

    confidence_margin = pd.Series(probabilities - threshold)
    position_size = pd.Series(0.0, index=confidence_margin.index)

    for upper_margin, size in buckets:
        position_size = position_size.mask(confidence_margin >= upper_margin, size)

    return position_size


def linear_position_size(probabilities, threshold: float, floor_size: float = 0.25) -> pd.Series:
    """Scale position size smoothly from a floor size at threshold to 100%."""

    probability_series = pd.Series(probabilities)
    confidence_range = max(1 - threshold, 0.01)
    scaled_confidence = ((probability_series - threshold) / confidence_range).clip(lower=0, upper=1)
    return (floor_size + (1 - floor_size) * scaled_confidence).where(probability_series >= threshold, 0.0)


def strategy_metrics_row(
    symbol: str,
    fold_name: str,
    strategy: str,
    threshold: float,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    classifier_metrics: dict[str, float],
    strategy_metrics: dict[str, float],
    exposure: float,
    trade_count: float,
) -> dict[str, float | str]:
    """Build one consistent metrics row for model strategy variants."""

    return {
        "symbol": symbol,
        "fold": fold_name,
        "strategy": strategy,
        "model": SELECTED_MODEL_NAME,
        "feature_set": SELECTED_FEATURE_SET_NAME,
        "threshold": threshold,
        "train_start": train_data["Date"].min(),
        "train_end": train_data["Date"].max(),
        "test_start": test_data["Date"].min(),
        "test_end": test_data["Date"].max(),
        "exposure": exposure,
        "trade_count": trade_count,
        **classifier_metrics,
        **strategy_metrics,
    }


def evaluate_strategy(symbol: str, fold_name: str, train_data: pd.DataFrame, test_data: pd.DataFrame, model: RandomForestClassifier) -> tuple[list[dict[str, float | str]], list[pd.DataFrame], pd.DataFrame]:
    """Evaluate the selected model strategy for one symbol/fold."""

    fitted_model = fit_model(model, train_data)
    probabilities = fitted_model.predict_proba(test_data[SELECTED_FEATURES])[:, 1]
    signal_threshold = SIGNAL_THRESHOLDS_BY_SYMBOL[symbol]
    predicted_labels = (probabilities >= signal_threshold).astype(int)

    strategy_data = test_data.copy()
    strategy_data["predicted_probability"] = probabilities
    strategy_data["signal"] = predicted_labels

    classifier_metrics = classification_metrics(test_data["target_next_day_up"], predicted_labels, probabilities)

    metrics_rows = []
    daily_return_frames = []

    long_flat_backtest = simple_long_flat_backtest(strategy_data)
    metrics_rows.append(
        strategy_metrics_row(
            symbol=symbol,
            fold_name=fold_name,
            strategy="long_flat",
            threshold=signal_threshold,
            train_data=train_data,
            test_data=test_data,
            classifier_metrics=classifier_metrics,
            strategy_metrics=risk_metrics(long_flat_backtest["net_strategy_return"]),
            exposure=strategy_data["signal"].mean(),
            trade_count=strategy_data["signal"].diff().abs().fillna(0).sum(),
        )
    )
    long_flat_daily_returns = long_flat_backtest[["Date", "net_strategy_return"]].copy()
    long_flat_daily_returns["symbol"] = symbol
    long_flat_daily_returns["fold"] = fold_name
    long_flat_daily_returns["strategy"] = "long_flat"
    daily_return_frames.append(long_flat_daily_returns)

    position_size_variants = {
        strategy_name: bucketed_position_size(probabilities, signal_threshold, buckets)
        for strategy_name, buckets in POSITION_SIZE_BUCKETS.items()
    }
    position_size_variants["sized_linear"] = linear_position_size(probabilities, signal_threshold)

    for strategy_name, position_size in position_size_variants.items():
        sized_data = strategy_data.copy()
        sized_data["position_size"] = position_size.to_numpy()
        sized_backtest = position_sized_backtest(sized_data)
        metrics_rows.append(
            strategy_metrics_row(
                symbol=symbol,
                fold_name=fold_name,
                strategy=strategy_name,
                threshold=signal_threshold,
                train_data=train_data,
                test_data=test_data,
                classifier_metrics=classifier_metrics,
                strategy_metrics=risk_metrics(sized_backtest["net_strategy_return"]),
                exposure=sized_data["position_size"].mean(),
                trade_count=sized_data["position_size"].diff().abs().fillna(0).astype(bool).sum(),
            )
        )

        daily_returns = sized_backtest[["Date", "net_strategy_return"]].copy()
        daily_returns["symbol"] = symbol
        daily_returns["fold"] = fold_name
        daily_returns["strategy"] = strategy_name
        daily_return_frames.append(daily_returns)

    volatility_input = test_data[["Date", "log_return_1d", VOLATILITY_COLUMN]].copy()
    volatility_input["symbol"] = symbol
    volatility_input["fold"] = fold_name
    volatility_input["signal"] = predicted_labels

    return metrics_rows, daily_return_frames, volatility_input


def build_volatility_adjusted_strategy(volatility_inputs: pd.DataFrame) -> tuple[list[dict[str, float | str]], pd.DataFrame]:
    """Build portfolio-level inverse-volatility weights for active signals."""

    metric_rows = []
    daily_return_frames = []

    for strategy_name, settings in VOLATILITY_ADJUSTED_VARIANTS.items():
        inputs = volatility_inputs.copy()
        inputs["inverse_volatility"] = 1 / inputs[VOLATILITY_COLUMN].replace(0, np.nan)
        inputs["raw_weight"] = inputs["signal"] * inputs["inverse_volatility"].fillna(0)
        daily_raw_weight = inputs.groupby("Date")["raw_weight"].transform("sum")
        inputs["target_weight"] = (inputs["raw_weight"] / daily_raw_weight.replace(0, np.nan)).fillna(0)
        inputs["target_weight"] = (inputs["target_weight"] * settings["gross_exposure"]).clip(upper=settings["max_weight"])

        inputs = inputs.sort_values(["symbol", "Date"]).reset_index(drop=True)
        inputs["position_weight"] = inputs.groupby("symbol")["target_weight"].shift(1).fillna(0)
        inputs["cost"] = inputs.groupby("symbol")["target_weight"].diff().abs().fillna(0) * (FEE_BPS / 10000.0)
        inputs["net_strategy_return"] = inputs["position_weight"] * inputs["log_return_1d"] - inputs["cost"]
        inputs["strategy"] = strategy_name

        for symbol, symbol_returns in inputs.groupby("symbol"):
            metrics = risk_metrics(symbol_returns["net_strategy_return"])
            metric_rows.append(
                {
                    "symbol": symbol,
                    "fold": "all",
                    "strategy": strategy_name,
                    "model": SELECTED_MODEL_NAME,
                    "feature_set": f"{SELECTED_FEATURE_SET_NAME}_inverse_volatility",
                    "threshold": SIGNAL_THRESHOLDS_BY_SYMBOL[symbol],
                    "train_start": pd.NaT,
                    "train_end": pd.NaT,
                    "test_start": symbol_returns["Date"].min(),
                    "test_end": symbol_returns["Date"].max(),
                    "exposure": symbol_returns["target_weight"].mean(),
                    "trade_count": symbol_returns["target_weight"].diff().abs().fillna(0).astype(bool).sum(),
                    "accuracy": pd.NA,
                    "precision": pd.NA,
                    "roc_auc": pd.NA,
                    **metrics,
                }
            )

        daily_return_frames.append(inputs[["Date", "symbol", "fold", "strategy", "net_strategy_return"]].copy())

    return metric_rows, pd.concat(daily_return_frames, ignore_index=True)


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
        if strategy.startswith("volatility_adjusted"):
            portfolio_returns = strategy_returns.groupby("Date")["net_strategy_return"].sum().sort_index()
        else:
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

    baseline_portfolio = portfolio_summary[portfolio_summary["strategy"] == "always_long"].iloc[0]
    model_portfolios = portfolio_summary[portfolio_summary["strategy"] != "always_long"].copy()
    best_model_portfolio = model_portfolios.sort_values(["sharpe_ratio", "total_return"], ascending=False).iloc[0]

    baseline_rows = symbol_summary[symbol_summary["strategy"] == "always_long"].set_index("symbol")

    portfolio_lines = "\n".join(
        f"- `{row['strategy']}`: Sharpe `{row['sharpe_ratio']:.4f}`, return `{row['total_return']:.2%}`, "
        f"drawdown `{row['max_drawdown']:.2%}`."
        for _, row in portfolio_summary.sort_values(["strategy"]).iterrows()
    )

    symbol_lines = []
    for symbol in baseline_rows.index:
        baseline = baseline_rows.loc[symbol]
        variants = symbol_summary[(symbol_summary["symbol"] == symbol) & (symbol_summary["strategy"] != "always_long")]
        variant_text = "; ".join(
            f"{row['strategy']} Sharpe `{row['avg_sharpe_ratio']:.4f}`, return `{row['avg_total_return']:.2%}`, drawdown `{row['avg_max_drawdown']:.2%}`"
            for _, row in variants.sort_values("strategy").iterrows()
        )
        symbol_lines.append(
            f"- `{symbol}`: {variant_text}; always_long Sharpe `{baseline['avg_sharpe_ratio']:.4f}`, "
            f"return `{baseline['avg_total_return']:.2%}`, drawdown `{baseline['avg_max_drawdown']:.2%}`."
        )

    symbol_text = "\n".join(symbol_lines)
    sharpe_delta = best_model_portfolio["sharpe_ratio"] - baseline_portfolio["sharpe_ratio"]
    return_delta = best_model_portfolio["total_return"] - baseline_portfolio["total_return"]

    report = f"""# Universe Walk-Forward Summary

Selected setup:

- Model: `{SELECTED_MODEL_NAME}`
- Feature set: `{SELECTED_FEATURE_SET_NAME}`
- Thresholds: `{", ".join(f"{symbol}={threshold:.2f}" for symbol, threshold in SIGNAL_THRESHOLDS_BY_SYMBOL.items())}`

## Equal-Weight Portfolio

- Best model variant: `{best_model_portfolio["strategy"]}`
- Best model Sharpe: `{best_model_portfolio["sharpe_ratio"]:.4f}`
- Always-long Sharpe: `{baseline_portfolio["sharpe_ratio"]:.4f}`
- Sharpe difference: `{sharpe_delta:.4f}`
- Best model total return: `{best_model_portfolio["total_return"]:.2%}`
- Always-long total return: `{baseline_portfolio["total_return"]:.2%}`
- Return difference: `{return_delta:.2%}`
- Best model max drawdown: `{best_model_portfolio["max_drawdown"]:.2%}`
- Always-long max drawdown: `{baseline_portfolio["max_drawdown"]:.2%}`

## Strategy Variants

{portfolio_lines}

## Per-Symbol Results

{symbol_text}

## Interpretation

This applies the optimized per-ticker threshold strategy to the full five-stock universe.
The `long_flat` variant keeps the original all-in/all-out behavior, while `sized_*` variants scale exposure by model confidence.
The `volatility_adjusted_*` variants weight active signals by inverse 20-day volatility, with diversified versions capped closer to equal-weight portfolio limits.
"""

    output_path.write_text(report)


def main() -> None:
    """Run the selected model setup across the configured stock universe."""

    config = ProjectConfig()
    reports_dir = Path("reports/universe")
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_template = build_selected_model()
    metric_rows: list[dict[str, float | str]] = []
    daily_return_frames: list[pd.DataFrame] = []
    volatility_input_frames: list[pd.DataFrame] = []

    for symbol in config.symbols:
        data = load_symbol_data(symbol)

        for fold_name, test_end in WALK_FORWARD_FOLDS:
            train_data = data[(data["Date"] >= TRAIN_START_DATE) & (data["Date"] < fold_name)].copy()
            test_data = data[(data["Date"] >= fold_name) & (data["Date"] < test_end)].copy()

            if train_data.empty or test_data.empty:
                print(f"Skipping {symbol} fold {fold_name} because train or test data is empty.")
                continue

            model_metrics, model_daily_returns, volatility_inputs = evaluate_strategy(symbol, fold_name, train_data, test_data, model_template)
            baseline_metrics, baseline_daily_returns = evaluate_always_long(symbol, fold_name, test_data)

            metric_rows.extend([*model_metrics, baseline_metrics])
            daily_return_frames.extend([*model_daily_returns, baseline_daily_returns])
            volatility_input_frames.append(volatility_inputs)

    volatility_metric_rows, volatility_daily_returns = build_volatility_adjusted_strategy(pd.concat(volatility_input_frames, ignore_index=True))
    metric_rows.extend(volatility_metric_rows)
    daily_return_frames.append(volatility_daily_returns)

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
