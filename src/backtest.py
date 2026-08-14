from __future__ import annotations

import numpy as np
import pandas as pd


def simple_long_flat_backtest(df: pd.DataFrame, signal_col: str = "signal", return_col: str = "log_return_1d", fee_bps: float = 5.0) -> pd.DataFrame:
    """Simulate a simple long/flat trading strategy.

    Signal meaning:
        1 = hold the stock
        0 = stay in cash

    The strategy uses yesterday's signal for today's return. That avoids
    lookahead bias because today's return would not be known when deciding.
    """

    # Work on a copy so the caller's DataFrame stays unchanged.
    out = df.copy()

    # Use yesterday's signal to calculate today's strategy return.
    out["strategy_return"] = out[signal_col].shift(1).fillna(0) * out[return_col]

    # Charge a transaction cost only when the position changes.
    out["cost"] = out[signal_col].diff().abs().fillna(0) * (fee_bps / 10000.0)

    # Net return is the strategy return after trading costs.
    out["net_strategy_return"] = out["strategy_return"] - out["cost"]

    # Convert cumulative log returns into a $1 growth curve.
    out["equity_curve"] = np.exp(out["net_strategy_return"].cumsum())

    # Buy-and-hold is the benchmark: buy the stock once and keep holding it.
    out["buy_and_hold"] = np.exp(out[return_col].cumsum())

    # Drawdown shows how far the strategy is below its previous peak.
    out["drawdown"] = out["equity_curve"] / out["equity_curve"].cummax() - 1
    return out


def position_sized_backtest(df: pd.DataFrame, position_col: str = "position_size", return_col: str = "log_return_1d", fee_bps: float = 5.0) -> pd.DataFrame:
    """Simulate a long/cash strategy with fractional position sizes."""

    out = df.copy()
    out["strategy_return"] = out[position_col].shift(1).fillna(0) * out[return_col]
    out["cost"] = out[position_col].diff().abs().fillna(0) * (fee_bps / 10000.0)
    out["net_strategy_return"] = out["strategy_return"] - out["cost"]
    out["equity_curve"] = np.exp(out["net_strategy_return"].cumsum())
    out["buy_and_hold"] = np.exp(out[return_col].cumsum())
    out["drawdown"] = out["equity_curve"] / out["equity_curve"].cummax() - 1
    return out


def risk_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Calculate trading-focused risk and return metrics."""

    # Remove missing returns before computing statistics.
    returns = returns.dropna()

    # Sortino only penalizes downside volatility, so isolate negative returns.
    downside_returns = returns[returns < 0]

    # Daily mean/std are annualized using 252 trading days per year.
    annualized_return = returns.mean() * periods_per_year
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)

    # Rebuild an equity curve so max drawdown can be calculated from returns.
    equity_curve = np.exp(returns.cumsum())
    max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": annualized_return / annualized_volatility if annualized_volatility else 0.0,
        "sortino_ratio": annualized_return / downside_volatility if downside_volatility else 0.0,
        "max_drawdown": max_drawdown,
        "total_return": equity_curve.iloc[-1] - 1 if not equity_curve.empty else 0.0,
    }
