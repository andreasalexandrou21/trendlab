"""Tearsheet and comparison table.

Deliberately plain. The job is to make an unflattering result as legible as a flattering one,
so there is no cherry-picked window, no linear y-axis hiding a 90% drawdown, and every run
carries its cost drag next to its return.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import BacktestResult

COLUMNS = [
    ("final_equity", "Final EUR", "{:>10,.0f}"),
    ("cagr", "CAGR", "{:>8.1%}"),
    ("ann_vol", "Vol", "{:>7.1%}"),
    ("sharpe", "Sharpe", "{:>7.2f}"),
    ("max_drawdown", "MaxDD", "{:>8.1%}"),
    ("ann_turnover", "Turn/yr", "{:>8.1f}"),
    ("ann_cost_drag", "Cost/yr", "{:>8.1%}"),
]


def comparison_table(results: dict[str, BacktestResult]) -> str:
    """One row per run. Plain text so it drops straight into a README or a commit message."""
    name_width = max(len(n) for n in results) + 2
    header = "strategy".ljust(name_width) + "".join(f"{label:>9}" for _, label, _ in COLUMNS)
    lines = [header, "-" * len(header)]

    for name, result in results.items():
        stats = result.stats()
        row = name.ljust(name_width)
        for key, _, fmt in COLUMNS:
            row += fmt.format(stats[key]).rjust(9)
        if result.ruined:
            row += "  RUINED"
        elif result.meta.get("stopped"):
            stopped_on = result.meta.get("stop_date")
            row += f"  STOPPED {pd.Timestamp(stopped_on).date()}"
        lines.append(row)

    return "\n".join(lines)


def tearsheet(
    results: dict[str, BacktestResult],
    path: str | Path = "reports/tearsheet.png",
    title: str = "trendlab",
) -> Path:
    """Equity on a log scale, drawdown underneath, rolling volatility below that.

    Log scale is not decoration. On a linear axis a run that goes from 1,000 to 15,000 makes
    an 80% drawdown in year two invisible, which is exactly the drawdown that decides whether
    a real person keeps the strategy switched on.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, height_ratios=[3, 2, 1.5])
    ax_eq, ax_dd, ax_vol = axes

    for name, result in results.items():
        equity = result.equity.replace(0.0, float("nan"))  # ruin: stop the line, don't log(0)
        ax_eq.plot(equity.index, equity, label=name, linewidth=1.3)
        ax_dd.fill_between(result.drawdown.index, result.drawdown * 100, 0, alpha=0.25)
        ax_dd.plot(result.drawdown.index, result.drawdown * 100, linewidth=1.0, label=name)
        rolling_vol = result.returns.rolling(90).std() * (365**0.5) * 100
        ax_vol.plot(rolling_vol.index, rolling_vol, linewidth=1.0, label=name)

    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("equity, EUR (log)")
    ax_eq.set_title(title)
    ax_eq.legend(loc="upper left", fontsize=8)
    ax_eq.grid(alpha=0.3, which="both")

    ax_dd.set_ylabel("drawdown, %")
    ax_dd.grid(alpha=0.3)

    ax_vol.set_ylabel("90d vol, % ann.")
    ax_vol.axhline(15, color="black", linestyle="--", linewidth=0.8, label="15% target")
    ax_vol.grid(alpha=0.3)
    ax_vol.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
