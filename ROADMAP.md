# ML Trading Bot Roadmap

This project is designed to grow in stages: first build a credible local research pipeline, then expand from one stock to a portfolio, then optionally add sentiment or cloud automation.

## Current Status

The project currently has a working AAPL validation pipeline and is expanding the selected setup to the full five-stock universe.

Completed:
- Created the repo structure
- Added project documentation and dependency list
- Selected a 5-stock universe: `AAPL`, `NVDA`, `AMZN`, `MSFT`, `META`
- Downloaded historical Yahoo Finance data
- Saved raw data locally in `data/raw/`
- Cleaned Yahoo CSV formatting issues
- Standardized the date and market data columns
- Created stationarity-aware features
- Built a reproducible feature generation script
- Trained an AAPL-only logistic regression baseline
- Converted predictions into long/flat trading signals
- Ran an AAPL backtest with transaction costs
- Reported Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and total return
- Compared the model strategy against buy-and-hold
- Added AAPL model comparison across Logistic Regression, Random Forest, Gradient Boosting, and XGBoost when available
- Added AAPL walk-forward validation with an always-long baseline and feature subset comparison
- Added beginner-friendly comments across the Python files

In progress:
- Full five-stock universe walk-forward validation
- Per-ticker model versus always-long comparison
- Equal-weight portfolio backtest

## Phase 1: Data Pipeline

Goal: make the market data reliable and reproducible.

Status: complete for the first version.

Work included:
- Pull historical market data from Yahoo Finance
- Save one raw CSV per ticker
- Verify all five tickers download successfully
- Clean extra Yahoo metadata rows
- Rename/standardize columns such as `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`
- Convert dates and market columns into proper data types

Useful scripts:
- `run_download.py`
- `run_features.py`
- `src/data_loader.py`
- `src/feature_engineering.py`

## Phase 2: Features and Labels

Goal: turn raw prices into model-ready data.

Status: complete for the first version.

Work included:
- Avoid raw prices as direct model inputs
- Use log returns, rolling volatility, volume changes, and rolling z-scores
- Create `target_next_day_up`
- Drop rows without enough rolling history
- Save processed feature files in `data/processed/`

Current feature set:
- `log_return_1d`
- `log_return_5d`
- `log_return_20d`
- `volatility_20d`
- `volume_log_change`
- `close_zscore_20d`
- `volume_zscore_20d`

## Phase 3: AAPL Baseline Model

Goal: validate the full ML-to-backtest workflow on one stock before scaling up.

Status: complete for the first version.

Work included:
- Train on AAPL data before `2023-01-01`
- Test on AAPL data from `2023-01-01` onward
- Train a logistic regression classifier
- Predict probability of next-day upward movement
- Use a `0.55` probability threshold for long/flat signals
- Save AAPL backtest output to `reports/AAPL_baseline_backtest.csv`

Useful scripts:
- `run_train_aapl.py`
- `src/train.py`
- `src/backtest.py`

## Phase 4: AAPL Reporting

Goal: make the first experiment easier to understand visually.

Status: complete for the first version.

Work included:
- Plot model strategy equity curve versus buy-and-hold
- Plot model strategy drawdown
- Save local chart files in `reports/figures/`

Useful script:
- `run_report_aapl.py`

## Phase 5: Improve the AAPL Experiment

Goal: improve the first baseline before expanding the project.

Status: complete for the first model-selection pass.

Work included:
- Tune the trading threshold, such as testing `0.50`, `0.55`, `0.60`, and `0.65`
- Save metrics for each threshold
- Add a signal exposure chart showing when the model was invested versus flat
- Add a simple confusion matrix plot
- Compare model types across a wider threshold sweep
- Validate the selected model with walk-forward folds
- Compare feature subsets to avoid carrying noisy inputs forward
- Compare against an always-long baseline

Useful script:
- `run_evaluate_aapl.py`
- `run_compare_aapl_models.py`
- `run_walk_forward_aapl.py`

Recommended next steps:
- Save trained model artifacts only after the modeling approach stabilizes
- Add chart examples to the README after deciding whether generated report images should be tracked

Success criteria:
- The model does not need to beat buy-and-hold yet
- The project should clearly explain tradeoffs between return, drawdown, and risk-adjusted performance

## Phase 6: Expand to All Five Stocks

Goal: move from AAPL-only testing to a broader stock universe.

Planned work:
- Run the selected AAPL candidate on all five tickers
- Report metrics per ticker
- Compare each ticker strategy against its own always-long benchmark
- Build an equal-weight portfolio backtest
- Evaluate portfolio-level Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and total return

Useful script:
- `run_walk_forward_universe.py`

Important:
- Keep per-ticker results visible so one strong or weak stock does not hide the real behavior
- Use time-based splits, not random shuffling

## Optional Phase 7: News and Sentiment

Goal: test whether sentiment improves risk-adjusted returns.

This phase should stay narrow. Historical article scraping with exact timestamps can become a major data-cleaning and leakage problem.

Recommended approach:
- Prefer a curated dataset or precomputed sentiment source over raw scraping
- Compare price-only versus price-plus-sentiment models
- Only keep sentiment if it improves the research story or the metrics

## Phase 8: Dashboard and Presentation

Goal: make the project easy to demo and discuss.

Planned work:
- Build a simple Streamlit dashboard
- Show equity curves, drawdowns, signals, and per-ticker metrics
- Improve the README with screenshots and a clear project story
- Add resume bullets and interview talking points

## Optional Phase 9: Cloud Automation

Goal: run daily inference automatically after the local project is solid.

Planned work:
- Save/load a trained model
- Add a daily signal script
- Use GitHub Actions or a lightweight cloud scheduler
- Fetch the latest market data after market close
- Generate daily signals and save them to a log

Important:
- Do not add cloud automation before the local backtest and reporting pipeline are credible
- Daily inference is enough; the system does not need to run 24/7

## Notes

- Build locally first.
- Commit at meaningful milestones, not every tiny change.
- In trading, accuracy is not the main goal; risk-adjusted returns matter more.
- Use features that are robust to non-stationarity, such as returns and normalized indicators.
- A simple but well-tested project is more impressive than a flashy one that is not methodologically sound.
