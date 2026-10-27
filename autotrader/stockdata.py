"""Daily stock / index candles from Yahoo Finance's public chart endpoint (no key needed).

  python -m autotrader.stockdata --symbol ^GSPC --since 2019-06-01 --out data/SPX_1d.csv
  python -m autotrader.stockdata --symbol ^NSEI --since 2019-06-01 --out data/NIFTY50_1d.csv

Timestamps are the trading date (00:00), so they line up with the weekday calendar used by
the forecaster. Index volume is often missing (e.g. Nifty 50); it is filled with 0, which
Kronos handles (it was trained with optional volume).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from .data import normalise

URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_yahoo_daily(symbol: str, since: str = "2019-06-01", until: str | None = None) -> pd.DataFrame:
    start = int(pd.Timestamp(since, tz="UTC").timestamp())
    end = int((pd.Timestamp(until, tz="UTC") if until else pd.Timestamp.now("UTC")).timestamp())
    r = requests.get(URL.format(symbol=requests.utils.quote(symbol)),
                     params={"period1": start, "period2": end, "interval": "1d", "events": "history"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None),
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
        "volume": q.get("volume"),
    })
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0).astype(float)
    df = normalise(df)
    # Drop today's bar if the session may still be open (Yahoo returns a live partial bar).
    today = pd.Timestamp.now(tz).normalize().tz_localize(None)
    return df[df["timestamps"] < today].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Download daily candles from Yahoo Finance to CSV")
    p.add_argument("--symbol", required=True, help="e.g. ^GSPC (S&P 500), ^NSEI (Nifty 50)")
    p.add_argument("--since", default="2019-06-01")
    p.add_argument("--until")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    df = fetch_yahoo_daily(a.symbol, a.since, a.until)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} daily bars {df.timestamps.min().date()} -> {df.timestamps.max().date()} to {a.out}")


if __name__ == "__main__":
    main()
