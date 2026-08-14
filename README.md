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

Run `python -m scripts.run_download` to fetch historical Yahoo Finance data and save it to `data/raw/`.

This will download one CSV per symbol in the basket.

Then run `python -m scripts.run_features` to create model-ready feature files in `data/processed/`.

Run `python -m scripts.run_train_aapl` to train the first AAPL-only baseline model and save a backtest report in `reports/aapl/`.

Run `python -m scripts.run_report_aapl` to generate AAPL equity curve and drawdown plots in `reports/aapl/figures/`.

Run `python -m scripts.run_evaluate_aapl` to compare signal thresholds and generate signal exposure/confusion matrix plots.

Run `python -m scripts.run_compare_aapl_models` to compare Logistic Regression, Random Forest, Gradient Boosting, and XGBoost when the local environment supports it.

Run `python -m scripts.run_walk_forward_aapl` to validate the AAPL candidate with walk-forward folds, an always-long baseline, and feature subset checks.

Run `python -m scripts.run_walk_forward_universe` to apply the optimized shared universe setup to all five stocks and compare the equal-weight model portfolio against always-long.

Run `python -m scripts.run_optimize_universe` to compare shared model, feature, threshold, selection-mode, per-ticker-threshold, and inverse-volatility weighting setups across the full five-stock portfolio.
