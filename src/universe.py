"""Stock universe for the project.

This file makes the chosen basket easy to find. Right now `ProjectConfig`
also stores the same symbols, so we may consolidate this later.
"""

# Large, liquid stocks that have plenty of data and news coverage.
STOCK_UNIVERSE = ("AAPL", "NVDA", "AMZN", "MSFT", "META")
