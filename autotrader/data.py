"""OHLCV data: download via ccxt public endpoints, load/save CSV."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


def timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    unit = tf[-1]
    n = int(tf[:-1])
    return {
        "m": pd.Timedelta(minutes=n),
        "h": pd.Timedelta(hours=n),
        "d": pd.Timedelta(days=n),
        "w": pd.Timedelta(weeks=n),
    }[unit]


def bars_per_year(tf: str, calendar: str = "24/7") -> float:
    """365 days a year for crypto; 252 trading days (6.5h sessions) for stock markets."""
    step = timeframe_to_timedelta(tf)
    if calendar != "weekdays":
        return pd.Timedelta(days=365) / step
    if step >= pd.Timedelta(days=1):
        return 252 * pd.Timedelta(days=1) / step
    return 252 * (pd.Timedelta(hours=6.5) / step)


def future_timestamps(last: pd.Timestamp, tf: str, n: int, calendar: str = "24/7") -> pd.Series:
    """Timestamps of the next n bars. Stock markets skip weekends (holidays are ignored)."""
    step = timeframe_to_timedelta(tf)
    if calendar == "weekdays" and step == pd.Timedelta(days=1):
        return pd.Series(pd.bdate_range(last.normalize() + pd.offsets.BDay(1), periods=n))
    return pd.Series([last + step * (i + 1) for i in range(n)])


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True).dt.tz_localize(None)
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df = df[COLUMNS].drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)
    if df[COLUMNS[1:]].isnull().values.any():
        df[COLUMNS[1:]] = df[COLUMNS[1:]].ffill()
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    return normalise(pd.read_csv(path))


def make_exchange(exchange_id: str, api_key: str | None = None, secret: str | None = None,
                  password: str | None = None, sandbox: bool = False):
    import ccxt  # imported lazily so backtests don't need it

    klass = getattr(ccxt, exchange_id)
    params = {"enableRateLimit": True}
    if api_key:
        params.update({"apiKey": api_key, "secret": secret})
        if password:
            params["password"] = password
    ex = klass(params)
    # ccxt ignores HTTPS_PROXY / REQUESTS_CA_BUNDLE by default; honour them (corporate proxies).
    if hasattr(ex, "session") and (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")):
        ex.session.trust_env = True
    if sandbox:
        ex.set_sandbox_mode(True)
    return ex


def fetch_ohlcv(exchange, symbol: str, timeframe: str, since: pd.Timestamp | None = None,
                limit_bars: int = 1000, drop_open_bar: bool = True) -> pd.DataFrame:
    """Fetch closed bars. Paginates forward from `since` (or just the latest page)."""
    tf_ms = int(timeframe_to_timedelta(timeframe).total_seconds() * 1000)
    rows: list[list[float]] = []
    since_ms = int(since.timestamp() * 1000) if since is not None else None
    while True:
        page = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=720)
        if not page:
            break
        rows.extend(page)
        if since_ms is None or len(rows) >= limit_bars or len(page) < 2:
            break
        next_since = page[-1][0] + tf_ms
        if next_since <= since_ms:
            break
        since_ms = next_since
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["ts"], unit="ms")
    df = normalise(df.drop(columns="ts"))
    if drop_open_bar and len(df):
        now = pd.Timestamp.now("UTC").tz_localize(None)
        df = df[df["timestamps"] + timeframe_to_timedelta(timeframe) <= now]
    return df.tail(limit_bars).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Download OHLCV to CSV (public endpoints, no keys).")
    p.add_argument("--exchange", default="kraken")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--since", default="2022-01-01")
    p.add_argument("--max-bars", type=int, default=50_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ex = make_exchange(a.exchange)
    df = fetch_ohlcv(ex, a.symbol, a.timeframe, pd.Timestamp(a.since, tz="UTC"), a.max_bars)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} bars {df.timestamps.min()} -> {df.timestamps.max()} to {a.out}")


if __name__ == "__main__":
    main()
