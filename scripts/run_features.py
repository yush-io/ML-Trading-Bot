from pathlib import Path

from src.config import ProjectConfig
from src.feature_engineering import add_market_features, feature_columns, load_raw_price_data


def main() -> None:
    """Build processed feature files from the raw downloaded CSVs."""

    # Load the symbols from the shared project config.
    config = ProjectConfig()

    # Raw files are read from data/raw/ and processed files go to data/processed/.
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Building features")
    print(f"Feature columns: {', '.join(feature_columns())}")

    # Process each stock independently so the outputs are easy to inspect.
    for symbol in config.symbols:
        raw_path = raw_dir / f"{symbol}.csv"
        processed_path = processed_dir / f"{symbol}_features.csv"

        # Give a clear error if someone skipped the download step.
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw data file: {raw_path}. Run python -m scripts.run_download first.")

        # Load raw prices, create features/labels, and save the processed dataset.
        raw_data = load_raw_price_data(str(raw_path))
        features = add_market_features(raw_data)
        features.to_csv(processed_path, index=False)

        # Print a quick health check for each ticker.
        missing_values = int(features.isna().sum().sum())
        print(
            f"{symbol}: raw_rows={len(raw_data)}, feature_rows={len(features)}, "
            f"missing_values={missing_values}, saved={processed_path}"
        )


# This lets the file run as a module with: python -m scripts.run_features
if __name__ == "__main__":
    main()
