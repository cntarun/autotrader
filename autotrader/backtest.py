"""Walk-forward backtest, split into an expensive and a cheap step.

  # 1) expensive: run the model once per decision bar and cache the forecasts
  python -m autotrader.backtest forecast --config C --data data/BTCUSD_4h.csv \
      --start 2024-01-01 --end 2025-12-31 --forecaster kronos --out runs/fc_kronos.csv
  # 2) cheap: simulate trading on cached forecasts (fees, slippage, delay, breakers)
  python -m autotrader.backtest simulate --config C --data data/BTCUSD_4h.csv \
      --forecasts runs/fc_kronos.csv --out runs/bt_kronos
  # 3) stress the same forecasts under worse costs / latency
  python -m autotrader.backtest scenarios --config C --data ... --forecasts ... --out runs/scen

No look-ahead: a forecast made at bar t uses bars <= t only, and is filled at the OPEN of
bar t+1+delay. Tune thresholds on one period, then evaluate once on a later, untouched period.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data import bars_per_year, load_csv
from .forecaster import Forecast, build_forecaster
from .strategy import decide, realised_vol_annual


# ---------------------------------------------------------------- step 1: forecasts
def run_forecasts(cfg: Config, df: pd.DataFrame, forecaster, start=None, end=None, step: int = 1,
                  out: str | None = None, progress: bool = True, batch: int = 1) -> pd.DataFrame:
    done = pd.DataFrame()
    if out and Path(out).exists():
        done = pd.read_csv(out, parse_dates=["timestamps"])
    done_ts = set(done["timestamps"]) if len(done) else set()
    idx = df.index[df.index >= cfg.model.lookback - 1]
    if start:
        idx = idx[df.loc[idx, "timestamps"] >= pd.Timestamp(start)]
    if end:
        idx = idx[df.loc[idx, "timestamps"] <= pd.Timestamp(end)]
    idx = [t for t in idx[::step] if df.at[t, "timestamps"] not in done_ts]
    rows = []
    many = getattr(forecaster, "forecast_many", None)
    for n in range(0, len(idx), batch):
        chunk = idx[n:n + batch]
        hists = [df.iloc[max(0, t - cfg.model.lookback + 1): t + 1] for t in chunk]
        fcs = many(hists) if many and batch > 1 else [forecaster.forecast(h) for h in hists]
        for t, fc in zip(chunk, fcs):
            rows.append(dict(timestamps=df.at[t, "timestamps"], ref_close=fc.ref_close,
                             mu=fc.mu, sigma=fc.sigma, p_up=fc.p_up))
        if progress and (n // batch) % 10 == 0:
            print(f"[forecast] {n + len(chunk)}/{len(idx)} {rows[-1]['timestamps']} "
                  f"mu={rows[-1]['mu']:+.4f} p_up={rows[-1]['p_up']:.2f}", flush=True)
        if out and (n // batch) % 10 == 9:  # checkpoint so long runs can resume
            _save_fc(done, rows, out)
    res = _save_fc(done, rows, out) if out else pd.DataFrame(rows)
    return res


def _save_fc(done, rows, out):
    res = pd.concat([done, pd.DataFrame(rows)], ignore_index=True).sort_values("timestamps")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    return res


# ---------------------------------------------------------------- step 2: simulate
def simulate(cfg: Config, df: pd.DataFrame, fc: pd.DataFrame, capital: float | None = None,
             fee_mult: float = 1.0, slip_mult: float = 1.0, delay_bars: int = 0,
             breakers: bool = True) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.costs.fee_rate *= fee_mult
    cfg.costs.slippage_bps *= slip_mult
    fee, slip = cfg.costs.fee_rate, cfg.costs.slippage_bps / 10_000
    capital = capital or cfg.paper_starting_quote
    bpy = bars_per_year(cfg.exchange.timeframe)

    df = df.reset_index(drop=True)
    pos_of = {ts: i for i, ts in enumerate(df["timestamps"])}
    fc = fc[fc["timestamps"].isin(pos_of)].copy()
    first = pos_of[fc["timestamps"].iloc[0]]
    orders: dict[int, pd.Series] = {}
    for _, r in fc.iterrows():
        orders[pos_of[r["timestamps"]]] = r

    quote, qty, peak, halted = capital, 0.0, capital, ""
    entry_px, fees_paid, trades, equity_rows = None, 0.0, [], []
    pending: tuple[int, float] | None = None   # (fill_bar, target_weight)
    weight = 0.0
    for i in range(first, len(df)):
        bar = df.iloc[i]
        # fill a pending order at this bar's open
        if pending and pending[0] == i:
            target = pending[1]
            pending = None
            eq_open = quote + qty * bar.open
            delta = (target * eq_open) - qty * bar.open
            if target == 0 and qty > 0:
                delta = -qty * bar.open
            if abs(delta) >= cfg.risk.min_order_notional:
                if delta > 0:
                    px = bar.open * (1 + slip)
                    notional = min(delta, quote / (1 + fee))
                    q = notional / px
                    quote -= notional * (1 + fee)
                    qty += q
                    entry_px = px
                    side = "buy"
                else:
                    px = bar.open * (1 - slip)
                    q = min(-delta / bar.open, qty)
                    notional = q * px
                    quote += notional * (1 - fee)
                    qty -= q
                    side = "sell"
                fees_paid += notional * fee
                pnl = None
                if side == "sell" and entry_px:
                    pnl = px / entry_px - 1 - 2 * fee
                    entry_px = None if qty <= 1e-12 else entry_px
                trades.append(dict(timestamps=bar.timestamps, side=side, price=px, qty=q,
                                   notional=notional, fee=notional * fee, trade_return=pnl))
        equity = quote + qty * bar.close
        peak = max(peak, equity)
        if breakers and not halted and 100 * (1 - equity / peak) >= cfg.risk.max_drawdown_pct:
            halted = f"max_drawdown at {bar.timestamps}"
        weight = qty * bar.close / equity if equity > 0 else 0.0
        equity_rows.append(dict(timestamps=bar.timestamps, equity=equity, weight=weight, close=bar.close))

        if i in orders and pending is None and i + 1 + delay_bars < len(df):
            r = orders[i]
            f = Forecast(r.ref_close, r.mu, r.sigma, r.p_up, np.empty((0, 0)))
            rv = realised_vol_annual(df["close"].iloc[: i + 1], bpy)
            target = 0.0 if halted else decide(f, weight, cfg.strategy, cfg.costs, rv).weight
            if abs(target - weight) > 1e-9:
                pending = (i + 1 + delay_bars, target)

    eq = pd.DataFrame(equity_rows)
    tr = pd.DataFrame(trades)
    return dict(summary=_summary(eq, tr, fees_paid, capital, bpy, halted, fc, df, cfg), equity=eq, trades=tr)


def _perf(equity: pd.Series, bpy: float) -> dict:
    r = equity.pct_change().dropna()
    years = max(len(r) / bpy, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    vol = r.std(ddof=0) * math.sqrt(bpy)
    downside = r[r < 0].std(ddof=0) * math.sqrt(bpy)
    mdd = (equity / equity.cummax() - 1).min()
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    return dict(total_return=total, cagr=cagr, ann_vol=vol,
                sharpe=(r.mean() * bpy) / vol if vol > 0 else 0.0,
                sortino=(r.mean() * bpy) / downside if downside > 0 else 0.0,
                max_drawdown=mdd, calmar=cagr / abs(mdd) if mdd < 0 else 0.0)


def forecast_quality(fc: pd.DataFrame, df: pd.DataFrame, horizon: int) -> dict:
    closes = df.set_index("timestamps")["close"]
    pos = {ts: i for i, ts in enumerate(closes.index)}
    real = []
    for ts, ref in zip(fc["timestamps"], fc["ref_close"]):
        j = pos[ts] + horizon
        real.append(closes.iloc[j] / ref - 1 if j < len(closes) else np.nan)
    f = fc.assign(realised=real).dropna(subset=["realised"])
    if len(f) < 10:
        return {}
    up = (f["realised"] > 0).astype(float)
    return dict(n=len(f),
                directional_hit_rate=float((np.sign(f["mu"]) == np.sign(f["realised"])).mean()),
                information_coefficient=float(f["mu"].rank().corr(f["realised"].rank())),  # Spearman
                brier_p_up=float(((f["p_up"] - up) ** 2).mean()),
                brier_climatology=float(((up.mean() - up) ** 2).mean()),
                # Calibration: >1 means real moves are bigger than the forecast spread
                # (the model is overconfident and p_up is too extreme).
                realised_to_forecast_spread=float(f["realised"].std() / f["sigma"].mean()),
                # Hit rate when the forecast is confident enough to trade
                hit_rate_when_mu_over_1_5pct=float((f.loc[f["mu"] > 0.015, "realised"] > 0).mean())
                if (f["mu"] > 0.015).any() else None,
                n_mu_over_1_5pct=int((f["mu"] > 0.015).sum()))


def _summary(eq, tr, fees_paid, capital, bpy, halted, fc, df, cfg) -> dict:
    strat = _perf(eq["equity"], bpy)
    bench = _perf(eq["close"] / eq["close"].iloc[0] * capital, bpy)
    sells = tr[tr["side"] == "sell"] if len(tr) else tr
    tret = sells["trade_return"].dropna() if len(sells) else pd.Series(dtype=float)
    return dict(
        period=[str(eq.timestamps.iloc[0]), str(eq.timestamps.iloc[-1])], bars=len(eq),
        start_equity=capital, end_equity=float(eq.equity.iloc[-1]),
        strategy={k: round(float(v), 4) for k, v in strat.items()},
        buy_and_hold={k: round(float(v), 4) for k, v in bench.items()},
        trades=int(len(tr)), round_trips=int(len(tret)),
        win_rate=round(float((tret > 0).mean()), 4) if len(tret) else None,
        avg_round_trip_return=round(float(tret.mean()), 5) if len(tret) else None,
        fees_paid=round(fees_paid, 4), exposure=round(float((eq.weight > 0.01).mean()), 4),
        halted=halted or None,
        costs=dict(fee_rate=cfg.costs.fee_rate, slippage_bps=cfg.costs.slippage_bps),
        forecast_quality=forecast_quality(fc, df, cfg.model.horizon),
    )


SCENARIOS = [
    ("base", dict()),
    ("fees_x2", dict(fee_mult=2.0)),
    ("slippage_x3", dict(slip_mult=3.0)),
    ("one_bar_latency", dict(delay_bars=1)),
    ("worst_case", dict(fee_mult=2.0, slip_mult=3.0, delay_bars=1)),
]


def _write(res: dict, out: str) -> None:
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(res["summary"], indent=2))
    res["equity"].to_csv(o / "equity.csv", index=False)
    res["trades"].to_csv(o / "trades.csv", index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("forecast", "simulate", "scenarios"):
        s = sub.add_parser(name)
        s.add_argument("--config")
        s.add_argument("--data", required=True)
        s.add_argument("--out", required=True)
        if name == "forecast":
            s.add_argument("--forecaster", default="kronos", choices=["kronos", "naive"])
            s.add_argument("--start")
            s.add_argument("--end")
            s.add_argument("--step", type=int, default=1)
            s.add_argument("--batch", type=int, default=1, help="decision points per model call (batching did not help on CPU)")
        else:
            s.add_argument("--forecasts", required=True)
            s.add_argument("--capital", type=float)
            s.add_argument("--no-breakers", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    df = load_csv(a.data)

    if a.cmd == "forecast":
        fcr = build_forecaster(a.forecaster, cfg.model, cfg.exchange.timeframe)
        res = run_forecasts(cfg, df, fcr, a.start, a.end, a.step, a.out, batch=a.batch)
        print(f"wrote {len(res)} forecasts to {a.out}")
        return
    fc = pd.read_csv(a.forecasts, parse_dates=["timestamps"])
    if a.cmd == "simulate":
        res = simulate(cfg, df, fc, a.capital, breakers=not a.no_breakers)
        _write(res, a.out)
        print(json.dumps(res["summary"], indent=2))
        return
    rows = []
    for name, kw in SCENARIOS:
        res = simulate(cfg, df, fc, a.capital, breakers=not a.no_breakers, **kw)
        _write(res, f"{a.out}/{name}")
        s = res["summary"]
        rows.append(dict(scenario=name, end_equity=round(s["end_equity"], 2),
                         total_return=s["strategy"]["total_return"], sharpe=s["strategy"]["sharpe"],
                         max_dd=s["strategy"]["max_drawdown"], trades=s["trades"], fees=s["fees_paid"],
                         bh_return=s["buy_and_hold"]["total_return"]))
    table = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    table.to_csv(f"{a.out}/scenarios.csv", index=False)
    print(table.to_markdown(index=False) if _has_tabulate() else table.to_string(index=False))


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
