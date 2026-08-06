from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    """Central settings used by the scripts in the project.

    Keeping these values in one class makes it easier to change the stock
    universe or date range without hunting through multiple files.
    """

    # The first version tracks five large, liquid technology stocks.
    symbols: tuple[str, ...] = ("AAPL", "NVDA", "AMZN", "MSFT", "META")

    # Historical data window downloaded from Yahoo Finance.
    start_date: str = "2018-01-01"
    end_date: str = "2026-01-01"

    # The model currently predicts whether the next trading day goes up.
    horizon_days: int = 1
