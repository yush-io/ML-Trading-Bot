from __future__ import annotations

import numpy as np
import pandas as pd


def load_raw_price_data(path: str) -> pd.DataFrame:
    """Load and clean one raw Yahoo Finance CSV file.

    The downloaded CSV has a couple of Yahoo metadata rows near the top.
    This function removes those rows, converts dates/numbers to proper types,
    and returns a clean table sorted by date.
    """

    # Read the CSV into a pandas DataFrame.
    df = pd.read_csv(path)

    # yfinance labels the first column "Price"; change it to the date.
    if "Price" in df.columns:
        df = df.rename(columns={"Price": "Date"})

    # Remove metadata rows like "Ticker,AAPL,AAPL,..." and "Date,,,,,".
    df = df[df["Date"].ne("Ticker") & df["Date"].ne("Date")].copy()

    # Convert the Date column from text into real datetime values.
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    # Convert price and volume columns from text into numeric columns.
    price_columns = ["Close", "High", "Low", "Open", "Volume"]
    df[price_columns] = df[price_columns].apply(pd.to_numeric, errors="coerce")

    # Keep rows ordered from oldest to newest.
    return df.sort_values("Date").reset_index(drop=True)


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from cleaned price data.

    Most features are based on returns or rolling z-scores instead of raw prices.
    That helps reduce problems caused by non-stationary price levels.
    """

    # Work on a copy so the original DataFrame is not modified unexpectedly.
    out = df.copy()

    # Log returns measure relative price movement over different lookback windows.
    out["log_return_1d"] = np.log(out["Close"] / out["Close"].shift(1))
    out["log_return_5d"] = np.log(out["Close"] / out["Close"].shift(5))
    out["log_return_20d"] = np.log(out["Close"] / out["Close"].shift(20))

    # Rolling volatility measures how much the stock has been moving recently.
    out["volatility_20d"] = out["log_return_1d"].rolling(20).std()

    # Volume change captures whether trading activity increased or decreased.
    out["volume_log_change"] = np.log(out["Volume"] / out["Volume"].shift(1))

    # Rolling means and standard deviations are used to calculate z-scores.
    close_mean_20d = out["Close"].rolling(20).mean()
    close_std_20d = out["Close"].rolling(20).std()
    volume_mean_20d = out["Volume"].rolling(20).mean()
    volume_std_20d = out["Volume"].rolling(20).std()

    # Z-scores describe how unusual today's close/volume is versus recent history.
    out["close_zscore_20d"] = (out["Close"] - close_mean_20d) / close_std_20d
    out["volume_zscore_20d"] = (out["Volume"] - volume_mean_20d) / volume_std_20d

    # Label: 1 if the next trading day return is positive, otherwise 0.
    out["target_next_day_up"] = (out["log_return_1d"].shift(-1) > 0).astype(int)

    # Rolling features create missing values at the beginning, so drop those rows.
    return out.dropna()


def feature_columns() -> list[str]:
    """Return the exact feature columns used as model inputs."""

    return [
        "log_return_1d",
        "log_return_5d",
        "log_return_20d",
        "volatility_20d",
        "volume_log_change",
        "close_zscore_20d",
        "volume_zscore_20d",
    ]
