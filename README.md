# ML Trading Bot

A machine learning trading research project that builds, tests, and visualizes daily long/flat trading signals across a five-stock universe: `AAPL`, `AMZN`, `META`, `MSFT`, and `NVDA`.

The goal is not to claim a production-ready trading edge. The goal is to build a realistic, explainable research pipeline that shows how market data moves from raw prices into features, model predictions, backtests, portfolio metrics, and a dashboard that can be reviewed by humans before any paper-trading integration.

## Purpose

This project was built to answer a practical question:

> Can a simple, explainable ML pipeline improve risk-adjusted performance versus just holding a basket of large-cap tech stocks?

The system focuses on:

- clean data ingestion
- stationarity-aware feature engineering
- model comparison
- probability threshold tuning
- walk-forward validation
- per-ticker strategy evaluation
- portfolio-level backtesting
- drawdown and risk analysis
- dashboard-based reporting

The current version is a research and visualization system. It does not place live trades.

## Live dashboard

The Streamlit dashboard is designed to be deployed as a public resume/demo link.

Run locally:

```bash
streamlit run dashboard/app.py
```

The dashboard reads generated CSV reports from `reports/universe/` and displays:

- portfolio return, Sharpe ratio, Sortino ratio, volatility, and max drawdown
- equity curves by strategy
- drawdown curves by strategy
- strategy comparison tables
- per-ticker model performance
- optimization leaderboard
- raw report data previews
- project architecture and implementation notes

## High-level architecture

```text
Yahoo Finance data
        |
        v
Raw CSV files
        |
        v
Feature engineering
        |
        v
Model training and threshold tuning
        |
        v
Walk-forward validation
        |
        v
Strategy backtests
        |
        v
Portfolio-level reports
        |
        v
Streamlit dashboard
```

## Core flow

1. Download historical daily price data for the five-stock universe.
2. Clean Yahoo Finance CSV formatting and standardize date/market columns.
3. Generate model-ready features from returns, volatility, volume changes, and rolling z-scores.
4. Create a next-day direction target called `target_next_day_up`.
5. Train and compare baseline models on AAPL first.
6. Expand the selected setup to all five tickers.
7. Tune probability thresholds for trading signals.
8. Validate with walk-forward splits instead of one lucky train/test split.
9. Convert predicted probabilities into long/flat strategy positions.
10. Include transaction costs in strategy returns.
11. Compare model strategies against always-long baselines.
12. Aggregate ticker-level returns into portfolio-level results.
13. Visualize the research in Streamlit.

## Core components

### Data pipeline

- `scripts/run_download.py` downloads daily market data.
- `src/data_loader.py` handles raw data loading and cleaning.
- `data/raw/` stores raw ticker CSV files.
- `data/processed/` stores feature-engineered datasets.

### Feature engineering

- `scripts/run_features.py` builds processed feature files.
- `src/feature_engineering.py` creates stationarity-aware features.

Current feature examples:

- `log_return_1d`
- `log_return_5d`
- `log_return_20d`
- `volatility_20d`
- `volume_log_change`
- `close_zscore_20d`
- `volume_zscore_20d`

### Modeling

- `src/train.py` contains model training and classification metric helpers.
- `scripts/run_train_aapl.py` trains the first AAPL baseline.
- `scripts/run_compare_aapl_models.py` compares candidate model families.
- `scripts/run_walk_forward_aapl.py` validates the AAPL setup with rolling folds.

Models tested include:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost when installed

### Backtesting

- `src/backtest.py` contains strategy backtest and risk metric logic.
- `scripts/run_evaluate_aapl.py` evaluates AAPL threshold behavior.
- `scripts/run_walk_forward_universe.py` applies the selected model setup to all five stocks.
- `scripts/run_optimize_universe.py` compares universe-level optimization settings.

Backtests include:

- long/flat model signals
- always-long baseline
- probability-threshold sweeps
- transaction costs
- confidence-based position sizing experiments
- inverse-volatility weighting experiments
- drawdown-aware strategy selection modes

### Dashboard

- `dashboard/app.py` is the Streamlit app.
- `.streamlit/config.toml` defines the app theme.
- `dashboard/README.md` explains how to run the dashboard.

The dashboard is intentionally read-only. It does not place Alpaca orders or execute trades when opened.

## Project structure

```text
ML Trading Bot/
  dashboard/
    app.py
    README.md
  scripts/
    run_download.py
    run_features.py
    run_train_aapl.py
    run_report_aapl.py
    run_evaluate_aapl.py
    run_compare_aapl_models.py
    run_walk_forward_aapl.py
    run_walk_forward_universe.py
    run_optimize_universe.py
  src/
    backtest.py
    config.py
    data_loader.py
    feature_engineering.py
    train.py
    universe.py
  reports/
    aapl/
    universe/
  data/
    raw/
    processed/
  .streamlit/
    config.toml
  requirements.txt
  ROADMAP.md
  README.md
```

## Strategy design

The trading strategy is currently long/flat:

- If the model probability is above the selected threshold, the strategy is long.
- If the probability is below the threshold, the strategy is flat.
- It does not short.
- It does not use stop losses or take profits.
- It does not rebalance based on live broker positions yet.

The current best research setup uses:

- model: `random_forest`
- feature set: `returns_only`
- per-ticker thresholds:

```text
AAPL=0.45
AMZN=0.55
META=0.60
MSFT=0.40
NVDA=0.45
```

## Key findings

The strongest current portfolio candidate is the `long_flat` strategy.

Portfolio walk-forward results from `2020-01-02` to `2023-12-29`:

| Strategy | Sharpe | Total return | Max drawdown |
|---|---:|---:|---:|
| `long_flat` | `1.3186` | `252.84%` | `-22.65%` |
| `volatility_adjusted_diversified_full` | `1.2822` | `216.35%` | `-20.97%` |
| `sized_near_full` | `1.2591` | `215.25%` | `-21.74%` |
| `always_long` | `0.7179` | `170.41%` | `-51.85%` |

Interpretation:

- The model-driven long/flat portfolio beat the always-long baseline on Sharpe ratio, total return, and max drawdown in the tested period.
- Confidence-based sizing reduced drawdown in some cases but usually gave up too much return.
- Diversified volatility-adjusted sizing reduced drawdown slightly, but did not beat the original long/flat strategy.
- Concentrated inverse-volatility sizing looked worse because it increased portfolio concentration risk.
- The best current candidate is still the simpler long/flat model strategy.

## Technical implementation details

### Walk-forward validation

Instead of relying on one train/test split, the project uses rolling walk-forward folds. This makes the evaluation more credible because the model is tested across multiple market periods.

### Probability thresholds

The model outputs probabilities, not direct trades. The strategy uses ticker-specific thresholds to decide whether the signal is strong enough to enter a long position.

### Trading metrics

The project evaluates both ML quality and trading quality.

ML metrics:

- accuracy
- precision
- ROC AUC

Trading metrics:

- Sharpe ratio
- Sortino ratio
- annualized return
- annualized volatility
- max drawdown
- total return
- exposure
- trade count

### Transaction costs

Backtests include estimated transaction costs using basis-point fees when positions change. This helps avoid overstating results from overly active strategies.

### Position sizing experiments

The project tested:

- fixed long/flat exposure
- confidence-based bucket sizing
- linear confidence sizing
- inverse-volatility portfolio weighting
- diversified inverse-volatility caps

The result was useful: more complex sizing did not automatically improve the strategy. The simpler long/flat approach remained the strongest candidate.

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- yfinance
- Streamlit
- Altair
- matplotlib
- seaborn

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Download data:

```bash
python -m scripts.run_download
```

Build features:

```bash
python -m scripts.run_features
```

Run the universe walk-forward report:

```bash
python -m scripts.run_walk_forward_universe
```

Run the universe optimizer:

```bash
python -m scripts.run_optimize_universe
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Future work

Planned next steps:

- Deploy the Streamlit dashboard as a public web app.
- Add a daily signal-generation script.
- Add Alpaca paper-trading integration.
- Display paper account positions, orders, and P/L in the dashboard.
- Add stronger risk controls before any live-trading consideration.
- Keep sentiment analysis as optional future research rather than a required feature.

## Important disclaimer

This project is for educational and research purposes only. It is not financial advice, not a production trading system, and not a recommendation to buy or sell any security.

## Signature

Built by Aayush Rashinkar as a machine learning, finance, and data visualization project.
