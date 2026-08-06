# ML Trading Bot

An ML-driven trading research project focused on:

- historical market data collection
- feature engineering
- model training
- walk-forward backtesting
- sentiment augmentation from news articles
- lightweight dashboarding

## Project goal

Build a realistic signal-generation system that can be explained clearly in an interview.

The first version will:

- use daily market data
- track a 5-stock basket: `AAPL`, `NVDA`, `AMZN`, `MSFT`, and `META`
- predict next-day direction or return bucket
- compare against simple baselines
- include transaction costs in backtests

## Planned stack

- Python
- pandas
- scikit-learn
- yfinance
- XGBoost or logistic regression
- Streamlit

## Repo layout

- `src/` core pipeline code
- `data/` local datasets
- `models/` trained model artifacts
- `reports/` backtest outputs and charts
- `notebooks/` exploratory analysis

## Milestones

1. Download and clean market data
2. Build features and labels
3. Train a baseline model
4. Backtest the strategy
5. Add news sentiment
6. Build a simple dashboard

## First step

Run `python run_download.py` to fetch historical Yahoo Finance data and save it to `data/raw/`.

This will download one CSV per symbol in the basket.

Then run `python run_features.py` to create model-ready feature files in `data/processed/`.

Run `python run_train_aapl.py` to train the first AAPL-only baseline model and save a backtest report in `reports/`.

Run `python run_report_aapl.py` to generate AAPL equity curve and drawdown plots in `reports/figures/`.
