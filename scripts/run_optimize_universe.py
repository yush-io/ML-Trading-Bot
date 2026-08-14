from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from src.backtest import risk_metrics, simple_long_flat_backtest
from src.config import ProjectConfig
from src.train import classification_metrics, train_baseline_model


SIGNAL_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MIN_MEANINGFUL_EXPOSURE = 0.10
MAX_MEANINGFUL_EXPOSURE = 0.95
MAX_DRAWDOWN_LIMIT = -0.25

TRAIN_START_DATE = "2018-01-01"
WALK_FORWARD_FOLDS = (
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
)

FEATURE_SETS = {
    "all_features": [
        "log_return_1d",
        "log_return_5d",
        "log_return_20d",
        "volatility_20d",
        "volume_log_change",
        "close_zscore_20d",
        "volume_zscore_20d",
    ],
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


def load_symbol_data(symbol: str) -> pd.DataFrame:
    """Load one processed symbol feature file."""

    features_path = Path("data/processed") / f"{symbol}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature file: {features_path}. Run python -m scripts.run_features first.")

    return pd.read_csv(features_path, parse_dates=["Date"])


def build_models() -> dict[str, object]:
    """Create the model candidates for the universe optimization pass."""

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


def fit_model(model_name: str, model, train_data: pd.DataFrame, feature_columns: list[str]):
    """Fit a fresh model for one symbol/fold/configuration."""

    if model_name == "logistic_regression":
        return model(train_data[feature_columns], train_data["target_next_day_up"])

    fitted_model = clone(model)
    fitted_model.fit(train_data[feature_columns], train_data["target_next_day_up"])
    return fitted_model


def evaluate_model_config(
    symbol: str,
    fold_name: str,
    model_name: str,
    model,
    feature_set_name: str,
    feature_columns: list[str],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[list[dict[str, float | str]], list[pd.DataFrame]]:
    """Evaluate one model/feature setup across all thresholds."""

    fitted_model = fit_model(model_name, model, train_data, feature_columns)
    probabilities = fitted_model.predict_proba(test_data[feature_columns])[:, 1]

    metric_rows: list[dict[str, float | str]] = []
    daily_return_frames: list[pd.DataFrame] = []

    for threshold in SIGNAL_THRESHOLDS:
        config_id = f"{model_name}|{feature_set_name}|{threshold:.2f}"
        predicted_labels = (probabilities >= threshold).astype(int)

        strategy_data = test_data.copy()
        strategy_data["predicted_probability"] = probabilities
        strategy_data["signal"] = predicted_labels

        backtest = simple_long_flat_backtest(strategy_data)
        classifier_metrics = classification_metrics(test_data["target_next_day_up"], predicted_labels, probabilities)
        strategy_metrics = risk_metrics(backtest["net_strategy_return"])

        metric_rows.append(
            {
                "config_id": config_id,
                "symbol": symbol,
                "fold": fold_name,
                "strategy": "model",
                "model": model_name,
                "feature_set": feature_set_name,
                "threshold": threshold,
                "train_start": train_data["Date"].min(),
                "train_end": train_data["Date"].max(),
                "test_start": test_data["Date"].min(),
                "test_end": test_data["Date"].max(),
                "exposure": strategy_data["signal"].mean(),
                "trade_count": strategy_data["signal"].diff().abs().fillna(0).sum(),
                **classifier_metrics,
                **strategy_metrics,
            }
        )

        daily_returns = backtest[["Date", "net_strategy_return"]].copy()
        daily_returns["config_id"] = config_id
        daily_returns["symbol"] = symbol
        daily_returns["fold"] = fold_name
        daily_returns["strategy"] = "model"
        daily_return_frames.append(daily_returns)

    return metric_rows, daily_return_frames


def evaluate_always_long(symbol: str, fold_name: str, test_data: pd.DataFrame) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Evaluate buy-and-hold for one symbol/fold."""

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
        "config_id": "always_long|baseline|0.00",
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
    daily_returns["config_id"] = "always_long|baseline|0.00"
    daily_returns["symbol"] = symbol
    daily_returns["fold"] = fold_name
    daily_returns["strategy"] = "always_long"
    return metrics, daily_returns


def summarize_symbol_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold metrics by symbol and configuration."""

    return (
        results.groupby(["config_id", "symbol", "strategy", "model", "feature_set", "threshold"], as_index=False)
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
        .reset_index(drop=True)
    )


def summarize_portfolio(daily_returns: pd.DataFrame, symbol_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations by equal-weight portfolio performance."""

    rows = []
    exposure = symbol_summary.groupby("config_id")["avg_exposure"].mean()

    for config_id, config_returns in daily_returns.groupby("config_id"):
        portfolio_returns = config_returns.groupby("Date")["net_strategy_return"].mean().sort_index()
        first_row = config_returns.iloc[0]
        metrics = risk_metrics(portfolio_returns)

        rows.append(
            {
                "config_id": config_id,
                "strategy": first_row["strategy"],
                "symbols": config_returns["symbol"].nunique(),
                "avg_exposure": exposure.loc[config_id],
                "start": portfolio_returns.index.min(),
                "end": portfolio_returns.index.max(),
                **metrics,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["sharpe_ratio", "total_return"], ascending=False)
        .reset_index(drop=True)
    )


def summarize_inverse_volatility_portfolio(daily_returns: pd.DataFrame, symbol_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations using inverse-volatility symbol weights."""

    rows = []
    exposure = symbol_summary.groupby("config_id")["avg_exposure"].mean()

    for config_id, config_returns in daily_returns.groupby("config_id"):
        pivot_returns = config_returns.pivot_table(index="Date", columns="symbol", values="net_strategy_return")
        inverse_volatility = 1 / pivot_returns.std().replace(0, np.nan)
        weights = inverse_volatility / inverse_volatility.sum()
        portfolio_returns = pivot_returns.mul(weights, axis=1).sum(axis=1).astype(float).sort_index()
        first_row = config_returns.iloc[0]
        metrics = risk_metrics(portfolio_returns)

        rows.append(
            {
                "config_id": config_id,
                "strategy": first_row["strategy"],
                "symbols": config_returns["symbol"].nunique(),
                "avg_exposure": exposure.loc[config_id],
                "weighting": "inverse_volatility",
                "start": portfolio_returns.index.min(),
                "end": portfolio_returns.index.max(),
                **metrics,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["sharpe_ratio", "total_return"], ascending=False)
        .reset_index(drop=True)
    )


def eligible_model_configs(portfolio_summary: pd.DataFrame) -> pd.DataFrame:
    """Filter out always-long and configurations with extreme exposure."""

    return portfolio_summary[
        (portfolio_summary["strategy"] == "model")
        & (portfolio_summary["avg_exposure"] >= MIN_MEANINGFUL_EXPOSURE)
        & (portfolio_summary["avg_exposure"] <= MAX_MEANINGFUL_EXPOSURE)
    ].copy()


def select_shared_strategy_modes(portfolio_summary: pd.DataFrame) -> pd.DataFrame:
    """Pick representative shared configurations for different risk preferences."""

    model_configs = eligible_model_configs(portfolio_summary)
    if model_configs.empty:
        model_configs = portfolio_summary[portfolio_summary["strategy"] == "model"].copy()

    selections = []

    best_sharpe = model_configs.sort_values(["sharpe_ratio", "total_return"], ascending=False).iloc[0]
    selections.append(("best_sharpe", best_sharpe))

    best_return = model_configs.sort_values(["total_return", "sharpe_ratio"], ascending=False).iloc[0]
    selections.append(("best_return", best_return))

    drawdown_limited = model_configs[model_configs["max_drawdown"] >= MAX_DRAWDOWN_LIMIT].copy()
    if not drawdown_limited.empty:
        best_return_with_drawdown_limit = drawdown_limited.sort_values(["total_return", "sharpe_ratio"], ascending=False).iloc[0]
        selections.append(("best_return_with_drawdown_limit", best_return_with_drawdown_limit))

    rows = []
    for selection_mode, row in selections:
        out = row.to_dict()
        out["selection_mode"] = selection_mode
        out["min_exposure_filter"] = MIN_MEANINGFUL_EXPOSURE
        out["max_exposure_filter"] = MAX_MEANINGFUL_EXPOSURE
        out["max_drawdown_limit"] = MAX_DRAWDOWN_LIMIT if selection_mode == "best_return_with_drawdown_limit" else pd.NA
        rows.append(out)

    return pd.DataFrame(rows)


def summarize_per_ticker_thresholds(daily_returns: pd.DataFrame, symbol_summary: pd.DataFrame) -> pd.DataFrame:
    """Allow each symbol to use its own threshold for the same model/feature set."""

    model_rows = symbol_summary[
        (symbol_summary["strategy"] == "model")
        & (symbol_summary["avg_exposure"] >= MIN_MEANINGFUL_EXPOSURE)
        & (symbol_summary["avg_exposure"] <= MAX_MEANINGFUL_EXPOSURE)
    ].copy()

    rows = []
    for (model_name, feature_set_name), group in model_rows.groupby(["model", "feature_set"]):
        selected_by_symbol = (
            group.sort_values(["symbol", "avg_sharpe_ratio", "avg_total_return"], ascending=[True, False, False])
            .groupby("symbol", as_index=False)
            .head(1)
        )

        selected_returns = []
        for _, selected in selected_by_symbol.iterrows():
            mask = (daily_returns["symbol"] == selected["symbol"]) & (daily_returns["config_id"] == selected["config_id"])
            selected_returns.append(daily_returns.loc[mask])

        if not selected_returns:
            continue

        combined_returns = pd.concat(selected_returns, ignore_index=True)
        portfolio_returns = combined_returns.groupby("Date")["net_strategy_return"].mean().sort_index()
        metrics = risk_metrics(portfolio_returns)
        threshold_summary = "; ".join(
            f"{row['symbol']}={row['threshold']:.2f}" for _, row in selected_by_symbol.sort_values("symbol").iterrows()
        )

        rows.append(
            {
                "selection_mode": "per_ticker_thresholds",
                "config_id": f"per_ticker_thresholds|{model_name}|{feature_set_name}",
                "strategy": "model",
                "model": model_name,
                "feature_set": feature_set_name,
                "thresholds": threshold_summary,
                "symbols": selected_by_symbol["symbol"].nunique(),
                "avg_exposure": selected_by_symbol["avg_exposure"].mean(),
                "start": portfolio_returns.index.min(),
                "end": portfolio_returns.index.max(),
                **metrics,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["sharpe_ratio", "total_return"], ascending=False)
        .reset_index(drop=True)
    )


def write_summary(
    portfolio_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    strategy_selections: pd.DataFrame,
    per_ticker_threshold_summary: pd.DataFrame,
    inverse_volatility_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a readable optimization summary."""

    model_configs = eligible_model_configs(portfolio_summary)
    if model_configs.empty:
        model_configs = portfolio_summary[portfolio_summary["strategy"] == "model"].copy()

    best = model_configs.iloc[0]
    baseline = portfolio_summary[portfolio_summary["strategy"] == "always_long"].iloc[0]
    best_symbols = symbol_summary[symbol_summary["config_id"] == best["config_id"]].copy()

    model_name, feature_set_name, threshold = best["config_id"].split("|")
    symbol_lines = "\n".join(
        f"- `{row['symbol']}`: Sharpe `{row['avg_sharpe_ratio']:.4f}`, return `{row['avg_total_return']:.2%}`, "
        f"drawdown `{row['avg_max_drawdown']:.2%}`, exposure `{row['avg_exposure']:.2%}`."
        for _, row in best_symbols.sort_values("symbol").iterrows()
    )

    top_lines = "\n".join(
        f"- `{row['config_id']}`: Sharpe `{row['sharpe_ratio']:.4f}`, return `{row['total_return']:.2%}`, "
        f"drawdown `{row['max_drawdown']:.2%}`, exposure `{row['avg_exposure']:.2%}`."
        for _, row in model_configs.head(5).iterrows()
    )
    selection_lines = "\n".join(
        f"- `{row['selection_mode']}`: `{row['config_id']}` with Sharpe `{row['sharpe_ratio']:.4f}`, "
        f"return `{row['total_return']:.2%}`, drawdown `{row['max_drawdown']:.2%}`, exposure `{row['avg_exposure']:.2%}`."
        for _, row in strategy_selections.iterrows()
    )

    best_per_ticker = per_ticker_threshold_summary.iloc[0]
    best_inverse_volatility = inverse_volatility_summary[
        (inverse_volatility_summary["strategy"] == "model")
        & (inverse_volatility_summary["avg_exposure"] >= MIN_MEANINGFUL_EXPOSURE)
        & (inverse_volatility_summary["avg_exposure"] <= MAX_MEANINGFUL_EXPOSURE)
    ].iloc[0]

    report = f"""# Universe Optimization Summary

## Best Shared Configuration

- Model: `{model_name}`
- Feature set: `{feature_set_name}`
- Threshold: `{threshold}`
- Portfolio Sharpe: `{best["sharpe_ratio"]:.4f}`
- Portfolio total return: `{best["total_return"]:.2%}`
- Portfolio max drawdown: `{best["max_drawdown"]:.2%}`
- Average exposure: `{best["avg_exposure"]:.2%}`

## Always-Long Baseline

- Portfolio Sharpe: `{baseline["sharpe_ratio"]:.4f}`
- Portfolio total return: `{baseline["total_return"]:.2%}`
- Portfolio max drawdown: `{baseline["max_drawdown"]:.2%}`

## Top Shared Configurations

{top_lines}

## Strategy Selection Modes

These selections use exposure between {MIN_MEANINGFUL_EXPOSURE:.0%} and {MAX_MEANINGFUL_EXPOSURE:.0%}.

{selection_lines}

## Per-Ticker Threshold Candidate

- Config: `{best_per_ticker["config_id"]}`
- Thresholds: `{best_per_ticker["thresholds"]}`
- Portfolio Sharpe: `{best_per_ticker["sharpe_ratio"]:.4f}`
- Portfolio total return: `{best_per_ticker["total_return"]:.2%}`
- Portfolio max drawdown: `{best_per_ticker["max_drawdown"]:.2%}`
- Average exposure: `{best_per_ticker["avg_exposure"]:.2%}`

## Inverse-Volatility Weighting Candidate

- Config: `{best_inverse_volatility["config_id"]}`
- Portfolio Sharpe: `{best_inverse_volatility["sharpe_ratio"]:.4f}`
- Portfolio total return: `{best_inverse_volatility["total_return"]:.2%}`
- Portfolio max drawdown: `{best_inverse_volatility["max_drawdown"]:.2%}`
- Average exposure: `{best_inverse_volatility["avg_exposure"]:.2%}`

## Best Configuration By Symbol

{symbol_lines}

## Interpretation

This optimization ranks shared configurations by equal-weight portfolio performance across all five stocks.
Use this result as the next price-only benchmark before adding sentiment features.
"""

    output_path.write_text(report)


def main() -> None:
    """Optimize a shared model setup across the full stock universe."""

    config = ProjectConfig()
    reports_dir = Path("reports/universe")
    reports_dir.mkdir(parents=True, exist_ok=True)

    models = build_models()
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

            baseline_metrics, baseline_daily_returns = evaluate_always_long(symbol, fold_name, test_data)
            metric_rows.append(baseline_metrics)
            daily_return_frames.append(baseline_daily_returns)

            for model_name, model in models.items():
                for feature_set_name, feature_columns in FEATURE_SETS.items():
                    model_metrics, model_daily_returns = evaluate_model_config(
                        symbol=symbol,
                        fold_name=fold_name,
                        model_name=model_name,
                        model=model,
                        feature_set_name=feature_set_name,
                        feature_columns=feature_columns,
                        train_data=train_data,
                        test_data=test_data,
                    )
                    metric_rows.extend(model_metrics)
                    daily_return_frames.extend(model_daily_returns)

    results = pd.DataFrame(metric_rows)
    if results.empty:
        raise RuntimeError("No universe optimization results were produced.")

    daily_returns = pd.concat(daily_return_frames, ignore_index=True)
    symbol_summary = summarize_symbol_results(results)
    portfolio_summary = summarize_portfolio(daily_returns, symbol_summary)
    strategy_selections = select_shared_strategy_modes(portfolio_summary)
    per_ticker_threshold_summary = summarize_per_ticker_thresholds(daily_returns, symbol_summary)
    inverse_volatility_summary = summarize_inverse_volatility_portfolio(daily_returns, symbol_summary)

    results_path = reports_dir / "optimization_fold_results.csv"
    symbol_summary_path = reports_dir / "optimization_symbol_summary.csv"
    portfolio_summary_path = reports_dir / "optimization_portfolio_summary.csv"
    daily_returns_path = reports_dir / "optimization_daily_returns.csv"
    strategy_selections_path = reports_dir / "optimization_strategy_selections.csv"
    per_ticker_threshold_path = reports_dir / "optimization_per_ticker_threshold_summary.csv"
    inverse_volatility_path = reports_dir / "optimization_inverse_volatility_summary.csv"
    markdown_summary_path = reports_dir / "optimization_summary.md"

    results.to_csv(results_path, index=False)
    symbol_summary.to_csv(symbol_summary_path, index=False)
    portfolio_summary.to_csv(portfolio_summary_path, index=False)
    daily_returns.to_csv(daily_returns_path, index=False)
    strategy_selections.to_csv(strategy_selections_path, index=False)
    per_ticker_threshold_summary.to_csv(per_ticker_threshold_path, index=False)
    inverse_volatility_summary.to_csv(inverse_volatility_path, index=False)
    write_summary(
        portfolio_summary,
        symbol_summary,
        strategy_selections,
        per_ticker_threshold_summary,
        inverse_volatility_summary,
        markdown_summary_path,
    )

    print(f"Saved fold results to {results_path}")
    print(f"Saved symbol summary to {symbol_summary_path}")
    print(f"Saved portfolio summary to {portfolio_summary_path}")
    print(f"Saved daily returns to {daily_returns_path}")
    print(f"Saved strategy selections to {strategy_selections_path}")
    print(f"Saved per-ticker threshold summary to {per_ticker_threshold_path}")
    print(f"Saved inverse-volatility summary to {inverse_volatility_path}")
    print(f"Saved markdown summary to {markdown_summary_path}")
    print("\nTop portfolio configurations")
    print(portfolio_summary.head(10))
    print("\nStrategy selections")
    print(strategy_selections)


if __name__ == "__main__":
    main()
