from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_price_data(symbol: str, start: str, end: str, output_path: str | Path) -> pd.DataFrame:
    """Download one stock's historical price data and save it as a CSV.

    Args:
        symbol: Stock ticker, such as "AAPL".
        start: First date to download, formatted as "YYYY-MM-DD".
        end: End date for the download, formatted as "YYYY-MM-DD".
        output_path: Where the CSV should be saved.

    Returns:
        A pandas DataFrame containing the downloaded Yahoo Finance data.
    """

    # `auto_adjust=True` adjusts prices for splits and dividends.
    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)

    # Convert strings to Path objects so folder/file operations are easier.
    output_path = Path(output_path)

    # Create the parent folder, such as data/raw/, if it does not exist yet.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the raw data so later scripts can reuse it without downloading again.
    data.to_csv(output_path)
    return data
