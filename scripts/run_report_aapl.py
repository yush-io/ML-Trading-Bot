from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Start with the first completed baseline experiment.
SYMBOL = "AAPL"


def save_equity_curve_plot(backtest: pd.DataFrame, output_path: Path) -> None:
    """Plot model strategy growth versus buy-and-hold growth."""

    plt.figure(figsize=(12, 6))
    plt.plot(backtest["Date"], backtest["equity_curve"], label="Model strategy")
    plt.plot(backtest["Date"], backtest["buy_and_hold"], label="Buy and hold")
    plt.title(f"{SYMBOL} Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_drawdown_plot(backtest: pd.DataFrame, output_path: Path) -> None:
    """Plot strategy drawdown over time."""

    plt.figure(figsize=(12, 4))
    plt.plot(backtest["Date"], backtest["drawdown"], label="Model strategy drawdown", color="darkred")
    plt.title(f"{SYMBOL} Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Create visual reports from the AAPL baseline backtest output."""

    backtest_path = Path("reports/aapl") / f"{SYMBOL}_baseline_backtest.csv"
    figures_dir = Path("reports/aapl/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not backtest_path.exists():
        raise FileNotFoundError(f"Missing backtest file: {backtest_path}. Run python -m scripts.run_train_aapl first.")

    backtest = pd.read_csv(backtest_path, parse_dates=["Date"])

    equity_curve_path = figures_dir / f"{SYMBOL}_equity_curve.png"
    drawdown_path = figures_dir / f"{SYMBOL}_drawdown.png"

    save_equity_curve_plot(backtest, equity_curve_path)
    save_drawdown_plot(backtest, drawdown_path)

    print(f"Saved equity curve plot to {equity_curve_path}")
    print(f"Saved drawdown plot to {drawdown_path}")


# This lets the file run as a module with: python -m scripts.run_report_aapl
if __name__ == "__main__":
    main()
