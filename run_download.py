from pathlib import Path

from src.config import ProjectConfig
from src.data_loader import download_price_data


def main() -> None:
    """Download raw Yahoo Finance data for every configured stock."""

    # Load the stock list and date range from one shared config object.
    config = ProjectConfig()

    # Download each stock separately and save one CSV per ticker.
    for symbol in config.symbols:
        output_path = Path("data/raw") / f"{symbol}.csv"
        data = download_price_data(
            symbol=symbol,
            start=config.start_date,
            end=config.end_date,
            output_path=output_path,
        )
        print(f"Saved {len(data)} rows to {output_path}")


# This lets the file run as a script with: python run_download.py
if __name__ == "__main__":
    main()
