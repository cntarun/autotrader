"""Block-bootstrap Monte Carlo on a backtest equity curve.

  python -m autotrader.montecarlo --equity runs/bt_kronos/equity.csv --timeframe 4h \
      --capital 20 --days 365 --sims 5000 --block-days 10 --haircut 0.5

--haircut 0.5 removes half of the strategy's average per-bar return when it is positive
(never adds return to a losing strategy). A rough stand-in for "live edge is about half the
backtest edge", which is common. Note: bars after a breaker halt are flat (zero return).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .data import bars_per_year


def bootstrap(returns: np.ndarray, n_bars: int, sims: int, block: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n <= block:
        raise ValueError("not enough returns for the block size")
    n_blocks = int(np.ceil(n_bars / block))
    starts = rng.integers(0, n - block, size=(sims, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(sims, -1)[:, :n_bars]
    return returns[idx]


def run(equity: pd.DataFrame, timeframe: str, capital: float, days: int, sims: int,
        block_days: int, haircut: float, calendar: str = "24/7") -> dict:
    bpy = bars_per_year(timeframe, calendar)
    r = equity["equity"].pct_change().dropna().to_numpy()
    if haircut:
        r = r - haircut * max(float(r.mean()), 0.0)
    n_bars = int(days * bpy / 365)
    block = max(1, int(block_days * bpy / 365))
    paths = bootstrap(r, n_bars, sims, block)
    curves = capital * np.cumprod(1 + paths, axis=1)
    final = curves[:, -1]
    mdd = (curves / np.maximum.accumulate(curves, axis=1) - 1).min(axis=1)
    pct = [5, 25, 50, 75, 95]
    return dict(
        sims=sims, horizon_days=days, capital=capital, haircut=haircut,
        final_equity_percentiles={f"p{p}": round(float(np.percentile(final, p)), 2) for p in pct},
        return_percentiles={f"p{p}": round(float(np.percentile(final / capital - 1, p)), 4) for p in pct},
        max_drawdown_percentiles={f"p{p}": round(float(np.percentile(mdd, p)), 4) for p in pct},
        prob_loss=round(float((final < capital).mean()), 4),
        prob_drawdown_over_20pct=round(float((mdd < -0.20).mean()), 4),
        prob_double=round(float((final >= 2 * capital).mean()), 4),
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--equity", required=True)
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--capital", type=float, default=20.0)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--sims", type=int, default=5000)
    p.add_argument("--block-days", type=int, default=10)
    p.add_argument("--haircut", type=float, default=0.5)
    p.add_argument("--calendar", default="24/7", choices=["24/7", "weekdays"])
    a = p.parse_args(argv)
    eq = pd.read_csv(a.equity, parse_dates=["timestamps"])
    print(json.dumps(run(eq, a.timeframe, a.capital, a.days, a.sims, a.block_days, a.haircut, a.calendar), indent=2))


if __name__ == "__main__":
    main()
