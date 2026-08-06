# ML Trading Bot Roadmap

This project is designed to be built in stages so the core research pipeline is solid before adding polish or deployment.

## Phase 1: Setup and Data

### Day 1
- Create the repo structure
- Install dependencies
- Pull historical market data from Yahoo Finance for the 5-stock basket
- Save raw data locally

### Day 2
- Inspect the data for missing values and gaps
- Verify all five tickers download successfully
- Plot price and volume
- Decide the first trading symbol and date range

### Day 3
- Clean the data
- Standardize column names
- Create a reproducible data loading script

## Phase 2: Features and Labels

### Day 4
- Build basic price-based features
- Focus on stationarity-safe features
- Use log returns, rolling z-scores, volatility, and normalized indicators
- Avoid feeding raw prices directly into the model

### Day 5
- Define the prediction target
- Start with next-day direction or return bucket
- Check for leakage and time alignment issues

### Day 6
- Create a train/validation split that respects time order
- Build a baseline model

## Phase 3: Model Training and Evaluation

### Day 7
- Train the first model
- Compare against a naive baseline

### Day 8
- Tune the model lightly
- Evaluate precision, recall, and ROC-AUC as diagnostics
- Add trading-focused evaluation with Sharpe Ratio, Sortino Ratio, and Maximum Drawdown

### Day 9
- Document what worked and what did not
- Save model outputs and metrics

## Phase 4: Backtesting

### Day 10
- Convert predictions into trading signals
- Build a simple long/flat backtest

### Day 11
- Add transaction costs
- Add slippage assumptions
- Generate equity curve and drawdown plots
- Report Sharpe Ratio, Sortino Ratio, and Maximum Drawdown for the strategy

### Day 12
- Compare strategy performance against buy-and-hold
- Check whether the model adds value after costs

## Phase 5: News and Sentiment

This phase is optional and should stay narrow. Historical article scraping with exact timestamps can become a major data-cleaning and leakage problem.

### Day 13
- Choose one cleaner sentiment source
- Prefer a curated dataset or precomputed sentiment source over raw scraping

### Day 14
- Build a sentiment scoring step
- Aggregate sentiment by day

### Day 15
- Compare price-only versus price-plus-sentiment models
- Check whether sentiment improves risk-adjusted returns

## Phase 6: Polish and Presentation

### Day 16
- Build a simple dashboard
- Show predictions, equity curve, and trade history

### Day 17
- Clean up the code
- Add configuration and comments where helpful

### Day 18
- Write a strong README
- Add screenshots or charts to the repo

### Day 19
- Prepare resume bullets
- Write interview talking points

## Optional Phase 7: Cloud

### Day 20
- Decide whether deployment is worth it
- If yes, host the dashboard or a small API

### Day 21
- Add a scheduled job for daily updates
- Keep the cloud version lightweight and cheap

## Notes

- Build locally first.
- Do not optimize for live trading before the backtest is credible.
- A simple but well-tested project is more impressive than a flashy one that is not methodologically sound.
- In trading, accuracy is not the main goal; risk-adjusted returns matter more.
- Use features that are robust to non-stationarity, such as returns and normalized indicators.
- Start with equal treatment across `AAPL`, `NVDA`, `AMZN`, `MSFT`, and `META` before adding any fancy weighting scheme.
