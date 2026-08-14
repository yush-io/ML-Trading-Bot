from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="ML trading bot dashboard",
    page_icon=":material/monitoring:",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "universe"
PORTFOLIO_SUMMARY_PATH = REPORTS_DIR / "universe_walk_forward_portfolio_summary.csv"
SYMBOL_SUMMARY_PATH = REPORTS_DIR / "universe_walk_forward_symbol_summary.csv"
DAILY_RETURNS_PATH = REPORTS_DIR / "universe_walk_forward_daily_returns.csv"
OPTIMIZATION_SUMMARY_PATH = REPORTS_DIR / "optimization_portfolio_summary.csv"

CORE_STRATEGIES = ["long_flat", "always_long", "sized_near_full", "volatility_adjusted_diversified_full"]
CHART_COLORS = ["#0f766e", "#ea580c", "#2563eb", "#7c3aed", "#be123c", "#475569"]


def inject_brand_styles() -> None:
    """Add a light brand layer on top of Streamlit's native components."""

    st.html(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 34rem),
                    radial-gradient(circle at top right, rgba(234, 88, 12, 0.10), transparent 28rem),
                    linear-gradient(180deg, #fbfbf4 0%, #f4f7ef 44%, #edf4ef 100%);
            }

            .hero-card {
                border: 1px solid rgba(15, 118, 110, 0.16);
                border-radius: 28px;
                padding: 34px 38px;
                margin: 4px 0 22px;
                background:
                    linear-gradient(135deg, rgba(10, 37, 35, 0.94), rgba(15, 118, 110, 0.86)),
                    radial-gradient(circle at 82% 18%, rgba(251, 191, 36, 0.34), transparent 16rem);
                box-shadow: 0 24px 70px rgba(15, 23, 42, 0.16);
                color: #f8fafc;
            }

            .eyebrow {
                color: #99f6e4;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.14em;
                margin-bottom: 14px;
                text-transform: uppercase;
            }

            .hero-card h1 {
                color: #ffffff;
                font-size: clamp(2.35rem, 5vw, 4.8rem);
                line-height: 0.95;
                letter-spacing: -0.07em;
                margin: 0 0 16px;
            }

            .hero-card p {
                color: #d9fffa;
                font-size: 1.04rem;
                line-height: 1.65;
                margin: 0;
                max-width: 790px;
            }

            .hero-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin-top: 28px;
            }

            .hero-stat {
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 18px;
                padding: 16px;
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(16px);
            }

            .hero-stat span {
                color: #b8fff0;
                display: block;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.1em;
                text-transform: uppercase;
            }

            .hero-stat strong {
                color: #ffffff;
                display: block;
                font-size: 1.48rem;
                letter-spacing: -0.03em;
                margin-top: 5px;
            }

            .project-card {
                border: 1px solid rgba(15, 118, 110, 0.14);
                border-radius: 24px;
                padding: 24px;
                background: rgba(255, 255, 255, 0.72);
                box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
            }

            .project-card h3 {
                margin-top: 0;
            }

            .project-chip {
                border: 1px solid rgba(15, 118, 110, 0.16);
                border-radius: 999px;
                display: inline-block;
                margin: 4px 6px 4px 0;
                padding: 7px 11px;
                background: #ecfdf5;
                color: #115e59;
                font-size: 0.82rem;
                font-weight: 700;
            }

            @media (max-width: 760px) {
                .hero-card {
                    padding: 26px;
                }

                .hero-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """
    )


@st.cache_data
def load_csv(path: Path, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load a report CSV and keep reruns fast."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=list(parse_dates))


def percent(value: float) -> str:
    return f"{value:.2%}"


def decimal(value: float) -> str:
    return f"{value:.4f}"


def portfolio_returns(daily_returns: pd.DataFrame, strategy: str) -> pd.Series:
    """Recreate portfolio returns using the same aggregation as the research script."""

    strategy_returns = daily_returns[daily_returns["strategy"] == strategy]
    if strategy.startswith("volatility_adjusted"):
        return strategy_returns.groupby("Date")["net_strategy_return"].sum().sort_index()
    return strategy_returns.groupby("Date")["net_strategy_return"].mean().sort_index()


def build_equity_curves(daily_returns: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        returns = portfolio_returns(daily_returns, strategy)
        if returns.empty:
            continue

        equity = np.exp(returns.cumsum())
        drawdown = equity / equity.cummax() - 1
        rows.append(
            pd.DataFrame(
                {
                    "Date": equity.index,
                    "strategy": strategy,
                    "equity": equity.values,
                    "drawdown": drawdown.values,
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["Date", "strategy", "equity", "drawdown"])
    return pd.concat(rows, ignore_index=True)


def strategy_label(strategy: str) -> str:
    return strategy.replace("_", " ")


def metric_delta(value: float, baseline: float, formatter) -> str:
    return formatter(value - baseline)


def chart_line(data: pd.DataFrame, y: str, y_title: str, tooltip_format: str) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y(f"{y}:Q", title=y_title),
            color=alt.Color("strategy:N", title="Strategy", scale=alt.Scale(range=CHART_COLORS)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("strategy:N", title="Strategy"),
                alt.Tooltip(f"{y}:Q", title=y_title, format=tooltip_format),
            ],
        )
        .interactive()
)


inject_brand_styles()

portfolio_summary = load_csv(PORTFOLIO_SUMMARY_PATH, parse_dates=("start", "end"))
symbol_summary = load_csv(SYMBOL_SUMMARY_PATH)
daily_returns = load_csv(DAILY_RETURNS_PATH, parse_dates=("Date",))
optimization_summary = load_csv(OPTIMIZATION_SUMMARY_PATH)

missing_reports = [
    str(path.relative_to(PROJECT_ROOT))
    for path in (PORTFOLIO_SUMMARY_PATH, SYMBOL_SUMMARY_PATH, DAILY_RETURNS_PATH)
    if not path.exists()
]

if missing_reports:
    st.error(
        "Missing dashboard report files. Run `python -m scripts.run_walk_forward_universe` first.\n\n"
        + "\n".join(f"- `{path}`" for path in missing_reports),
        icon=":material/error:",
    )
    st.stop()


available_strategies = portfolio_summary["strategy"].tolist()
default_strategies = [strategy for strategy in CORE_STRATEGIES if strategy in available_strategies]
if not default_strategies:
    default_strategies = available_strategies[: min(4, len(available_strategies))]

with st.sidebar:
    st.header("Trading bot")
    st.caption("Research dashboard for model validation and portfolio backtests.")
    st.badge("Read-only", icon=":material/lock:", color="gray")
    st.badge("Walk-forward tested", icon=":material/check_circle:", color="green")
    st.space("small")
    st.subheader("Controls")
    selected_strategies = st.multiselect(
        "Compare strategies",
        available_strategies,
        default=default_strategies,
    )
    selected_symbols = st.multiselect(
        "Filter tickers",
        sorted(symbol_summary["symbol"].unique()),
        default=sorted(symbol_summary["symbol"].unique()),
    )
    st.caption("This dashboard is read-only. It does not place trades or connect to Alpaca yet.")

if not selected_strategies:
    st.warning("Pick at least one strategy in the sidebar.", icon=":material/tune:")
    st.stop()

baseline_row = portfolio_summary[portfolio_summary["strategy"] == "always_long"].iloc[0]
candidate_rows = portfolio_summary[portfolio_summary["strategy"] != "always_long"]
best_row = candidate_rows.sort_values(["sharpe_ratio", "total_return"], ascending=False).iloc[0]

st.html(
    f"""
    <section class="hero-card">
        <div class="eyebrow">Machine learning trading research</div>
        <h1>Five-stock signal dashboard.</h1>
        <p>
            A walk-forward ML trading system that compares model-driven long/flat signals against an always-long
            benchmark across AAPL, AMZN, META, MSFT, and NVDA. This dashboard is built for research visibility:
            model results, risk metrics, equity curves, drawdowns, and strategy tradeoffs in one place.
        </p>
        <div class="hero-grid">
            <div class="hero-stat">
                <span>Best strategy</span>
                <strong>{strategy_label(best_row["strategy"])}</strong>
            </div>
            <div class="hero-stat">
                <span>Total return</span>
                <strong>{percent(best_row["total_return"])}</strong>
            </div>
            <div class="hero-stat">
                <span>Max drawdown</span>
                <strong>{percent(best_row["max_drawdown"])}</strong>
            </div>
        </div>
    </section>
    """
)

st.badge("Research mode", icon=":material/science:", color="blue")
st.badge("No live orders", icon=":material/lock:", color="gray")
st.badge(f"Best variant: {strategy_label(best_row['strategy'])}", icon=":material/trending_up:", color="green")

with st.container(horizontal=True):
    st.metric(
        "Best strategy return",
        percent(best_row["total_return"]),
        metric_delta(best_row["total_return"], baseline_row["total_return"], percent),
        border=True,
    )
    st.metric(
        "Best strategy Sharpe",
        decimal(best_row["sharpe_ratio"]),
        metric_delta(best_row["sharpe_ratio"], baseline_row["sharpe_ratio"], decimal),
        border=True,
    )
    st.metric(
        "Best max drawdown",
        percent(best_row["max_drawdown"]),
        metric_delta(best_row["max_drawdown"], baseline_row["max_drawdown"], percent),
        border=True,
    )
    st.metric(
        "Always-long return",
        percent(baseline_row["total_return"]),
        f"{int(baseline_row['symbols'])} symbols",
        border=True,
    )

equity_curves = build_equity_curves(daily_returns, selected_strategies)

with st.container(border=False):
    info_left, info_middle, info_right = st.columns(3)
    with info_left:
        st.html(
            """
            <div class="project-card">
                <h3>What this project does</h3>
                <p>
                    Trains and evaluates daily stock-direction models, converts probabilities into long/flat
                    signals, then backtests the resulting portfolio with transaction costs.
                </p>
            </div>
            """
        )
    with info_middle:
        st.html(
            """
            <div class="project-card">
                <h3>Stack</h3>
                <span class="project-chip">Python</span>
                <span class="project-chip">pandas</span>
                <span class="project-chip">scikit-learn</span>
                <span class="project-chip">Streamlit</span>
                <span class="project-chip">Altair</span>
            </div>
            """
        )
    with info_right:
        st.html(
            """
            <div class="project-card">
                <h3>Research guardrails</h3>
                <p>
                    Uses walk-forward validation, always-long baselines, exposure checks, drawdown tracking,
                    and separate strategy variants before any paper trading integration.
                </p>
            </div>
            """
        )

tab_overview, tab_project, tab_symbols, tab_optimization, tab_data = st.tabs(
    [
        ":material/monitoring: Overview",
        ":material/account_tree: Project",
        ":material/finance: Per ticker",
        ":material/tune: Optimization",
        ":material/table_chart: Data",
    ]
)

with tab_overview:
    chart_left, chart_right = st.columns(2)
    with chart_left:
        with st.container(border=True):
            st.subheader("Equity curve")
            st.altair_chart(chart_line(equity_curves, "equity", "Equity multiple", ".2f"))

    with chart_right:
        with st.container(border=True):
            st.subheader("Drawdown")
            st.altair_chart(chart_line(equity_curves, "drawdown", "Drawdown", ".1%"))

    with st.container(border=True):
        st.subheader("Strategy comparison")
        visible_portfolio = portfolio_summary[portfolio_summary["strategy"].isin(selected_strategies)].copy()
        visible_portfolio = visible_portfolio.sort_values(["sharpe_ratio", "total_return"], ascending=False)
        st.dataframe(
            visible_portfolio,
            hide_index=True,
            column_config={
                "strategy": st.column_config.TextColumn("Strategy"),
                "symbols": st.column_config.NumberColumn("Symbols", format="%d"),
                "start": st.column_config.DateColumn("Start"),
                "end": st.column_config.DateColumn("End"),
                "annualized_return": st.column_config.NumberColumn("Annualized return", format="percent"),
                "annualized_volatility": st.column_config.NumberColumn("Annualized volatility", format="percent"),
                "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.4f"),
                "sortino_ratio": st.column_config.NumberColumn("Sortino", format="%.4f"),
                "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
                "total_return": st.column_config.NumberColumn("Total return", format="percent"),
            },
        )

    with st.container(border=True):
        st.subheader("Current read")
        st.write(
            "The original `long_flat` strategy is still the best overall candidate by Sharpe and total return. "
            "The diversified volatility-adjusted version reduces drawdown a little, but gives up return, so it is "
            "useful as a risk-control experiment rather than the main strategy."
        )

with tab_project:
    with st.container(border=True):
        st.subheader("Project purpose")
        st.write(
            "This project tests whether a simple, explainable machine learning pipeline can improve risk-adjusted "
            "performance versus holding a five-stock basket. The system is built as a research pipeline first: "
            "data, features, models, backtests, reports, and dashboard visibility before any broker integration."
        )

    architecture_left, architecture_right = st.columns(2)
    with architecture_left:
        with st.container(border=True):
            st.subheader("Architecture flow")
            st.markdown(
                """
                ```text
                Yahoo Finance data
                        |
                        v
                Raw ticker CSVs
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
                Portfolio reports
                        |
                        v
                Streamlit dashboard
                ```
                """
            )

    with architecture_right:
        with st.container(border=True):
            st.subheader("Core components")
            st.markdown(
                """
                - `src/data_loader.py`: raw data loading and cleaning.
                - `src/feature_engineering.py`: returns, volatility, volume, and z-score features.
                - `src/train.py`: model training and classification metrics.
                - `src/backtest.py`: strategy returns, costs, equity curves, and risk metrics.
                - `scripts/run_walk_forward_universe.py`: selected five-stock walk-forward strategy.
                - `scripts/run_optimize_universe.py`: strategy and portfolio optimization experiments.
                - `dashboard/app.py`: visual reporting layer for the research results.
                """
            )

    detail_left, detail_right = st.columns(2)
    with detail_left:
        with st.container(border=True):
            st.subheader("Implementation details")
            st.markdown(
                """
                - The model predicts next-day direction using daily, stationarity-aware features.
                - Predictions are converted into long/flat signals with per-ticker probability thresholds.
                - Walk-forward folds reduce reliance on one lucky split.
                - Backtests include transaction costs when positions change.
                - Strategy quality is measured with Sharpe, Sortino, total return, max drawdown, exposure, and trade count.
                - The dashboard is read-only and does not place trades.
                """
            )

    with detail_right:
        with st.container(border=True):
            st.subheader("Key findings")
            st.markdown(
                f"""
                - Best current candidate: `{strategy_label(best_row["strategy"])}`.
                - Best strategy total return: `{percent(best_row["total_return"])}`.
                - Best strategy Sharpe ratio: `{decimal(best_row["sharpe_ratio"])}`.
                - Best strategy max drawdown: `{percent(best_row["max_drawdown"])}`.
                - Always-long total return: `{percent(baseline_row["total_return"])}`.
                - Always-long max drawdown: `{percent(baseline_row["max_drawdown"])}`.
                """
            )

    with st.container(border=True):
        st.subheader("Tech stack and project structure")
        st.markdown(
            """
            **Tech stack:** Python, pandas, NumPy, scikit-learn, XGBoost, yfinance, Streamlit, Altair, matplotlib, and seaborn.

            **Clean structure:** reusable logic lives in `src/`, command-line research runs live in `scripts/`, generated outputs live in `reports/`, and the deployable dashboard lives in `dashboard/`.

            **Signature:** Built by Aayush Rashinkar as a machine learning, finance, and data visualization project.
            """
        )

with tab_symbols:
    filtered_symbols = symbol_summary[
        symbol_summary["symbol"].isin(selected_symbols) & symbol_summary["strategy"].isin(selected_strategies)
    ].copy()

    with st.container(border=True):
        st.subheader("Ticker-level performance")
        st.dataframe(
            filtered_symbols.sort_values(["symbol", "avg_sharpe_ratio"], ascending=[True, False]),
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Ticker"),
                "strategy": st.column_config.TextColumn("Strategy"),
                "model": st.column_config.TextColumn("Model"),
                "feature_set": st.column_config.TextColumn("Features"),
                "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
                "folds": st.column_config.NumberColumn("Folds", format="%d"),
                "avg_exposure": st.column_config.NumberColumn("Avg exposure", format="percent"),
                "avg_trade_count": st.column_config.NumberColumn("Avg trades", format="%.1f"),
                "avg_accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
                "avg_precision": st.column_config.NumberColumn("Precision", format="percent"),
                "avg_roc_auc": st.column_config.NumberColumn("ROC AUC", format="%.4f"),
                "avg_sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.4f"),
                "avg_sortino_ratio": st.column_config.NumberColumn("Sortino", format="%.4f"),
                "avg_max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
                "avg_total_return": st.column_config.NumberColumn("Total return", format="percent"),
            },
        )

    if not filtered_symbols.empty:
        symbol_chart_data = filtered_symbols[["symbol", "strategy", "avg_total_return", "avg_max_drawdown"]].melt(
            id_vars=["symbol", "strategy"],
            value_vars=["avg_total_return", "avg_max_drawdown"],
            var_name="metric",
            value_name="value",
        )
        symbol_chart_data["metric"] = symbol_chart_data["metric"].map(
            {"avg_total_return": "Total return", "avg_max_drawdown": "Max drawdown"}
        )
        chart = (
            alt.Chart(symbol_chart_data)
            .mark_bar()
            .encode(
                x=alt.X("symbol:N", title="Ticker"),
                y=alt.Y("value:Q", title="Value", axis=alt.Axis(format="%")),
                color=alt.Color("strategy:N", title="Strategy", scale=alt.Scale(range=CHART_COLORS)),
                column=alt.Column("metric:N", title=None),
                tooltip=[
                    alt.Tooltip("symbol:N", title="Ticker"),
                    alt.Tooltip("strategy:N", title="Strategy"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value", format=".1%"),
                ],
            )
        )
        st.altair_chart(chart)

with tab_optimization:
    if optimization_summary.empty:
        st.info("No optimization summary found yet. Run `python -m scripts.run_optimize_universe` to populate this view.")
    else:
        with st.container(border=True):
            st.subheader("Optimization leaderboard")
            ordered_optimization = optimization_summary.sort_values(["sharpe_ratio", "total_return"], ascending=False)
            st.dataframe(
                ordered_optimization,
                hide_index=True,
                column_config={
                    "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.4f"),
                    "sortino_ratio": st.column_config.NumberColumn("Sortino", format="%.4f"),
                    "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
                    "total_return": st.column_config.NumberColumn("Total return", format="percent"),
                    "annualized_return": st.column_config.NumberColumn("Annualized return", format="percent"),
                    "annualized_volatility": st.column_config.NumberColumn("Annualized volatility", format="percent"),
                },
            )

with tab_data:
    with st.expander("Portfolio summary CSV", icon=":material/table_chart:"):
        st.dataframe(portfolio_summary, hide_index=True)
    with st.expander("Daily returns CSV", icon=":material/query_stats:"):
        st.dataframe(daily_returns.head(500), hide_index=True)
        st.caption("Showing first 500 rows to keep the browser light.")
