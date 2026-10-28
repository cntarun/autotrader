"""Global stock-index universe for the ranking test and fine-tuning (Yahoo Finance symbols).

  python -m autotrader.universe --since 2000-01-01 --out-dir data/universe
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .stockdata import fetch_yahoo_daily

UNIVERSE = {
    "SPX": "^GSPC",          # S&P 500 (US)
    "NDX": "^NDX",           # Nasdaq 100 (US)
    "SX5E": "^STOXX50E",     # Euro Stoxx 50 (eurozone)
    "DAX": "^GDAXI",         # DAX 40 (Germany)
    "CAC40": "^FCHI",        # CAC 40 (France)
    "FTSE100": "^FTSE",      # FTSE 100 (UK)
    "NIKKEI": "^N225",       # Nikkei 225 (Japan)
    "HSI": "^HSI",           # Hang Seng (Hong Kong)
    "NIFTY50": "^NSEI",      # Nifty 50 (India)
    "KOSPI": "^KS11",        # KOSPI (South Korea)
    "ASX200": "^AXJO",       # S&P/ASX 200 (Australia)
    "EM": "EEM",             # MSCI Emerging Markets (iShares ETF, USD)
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="2000-01-01")
    p.add_argument("--out-dir", default="data/universe")
    a = p.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for name, sym in UNIVERSE.items():
        df = fetch_yahoo_daily(sym, a.since)
        df.to_csv(f"{a.out_dir}/{name}_1d.csv", index=False)
        print(f"{name:8s} {sym:10s} {len(df):5d} bars {df.timestamps.min().date()} -> {df.timestamps.max().date()}")


if __name__ == "__main__":
    main()
