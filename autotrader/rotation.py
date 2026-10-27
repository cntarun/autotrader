"""Weekly cross-sectional index ranking ("rotation") with Kronos.

Each Friday, forecast every index 5 trading days ahead, rank them, hold the top K equally
weighted for the next week, and compare with holding all indices equally (EW).

  # 1) forecasts (expensive; resumable)
  python -m autotrader.rotation forecast --config deploy/config/config.index-daily.yaml \
      --data-dir data/universe --start 2021-06-01 --forecaster kronos --out runs/rot_fc_kronos.csv
  # 2) evaluate: tune on the dev window, freeze, test once on the test window
  python -m autotrader.rotation evaluate --data-dir data/universe \
      --forecasts runs/rot_fc_kronos.csv --baseline runs/rot_fc_naive.csv --out runs/rotation.json

Returns are weekly close-to-close in each index's own currency. Costs are charged on turnover.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import load_csv
from .forecaster import build_forecaster

DEV = ("2022-01-07", "2025-08-29")      # tuning allowed here
TEST = ("2025-09-05", "2026-10-23")     # frozen settings, evaluated once
# Fixed before any Kronos results. buffer: keep a held index until it falls below rank top_k+buffer
# (cuts turnover; added after the naive-only run showed ~145%/week turnover = ~11%/yr in costs).
PRESET = {"top_k": 3, "score": "debiased_mu", "z_window": 26, "buffer": 2}
GRID = {"top_k": [2, 3, 4], "score": ["mu", "debiased_mu", "p_up"], "z_window": [13, 26, 52],
        "buffer": [0, 2, 4]}
PASS_RULES = (
    "On the TEST window with the dev-chosen settings, after 0.10% fees + 5 bps slippage: "
    "(1) information ratio vs equal-weight >= 0.5; (2) Sharpe above equal-weight's; "
    "(3) information ratio above the same procedure driven by the naive forecaster; "
    "(4) information ratio > 0 at 0.25% fees."
)


def load_universe(data_dir: str) -> dict[str, pd.DataFrame]:
    return {p.name[:-7]: load_csv(p) for p in sorted(Path(data_dir).glob("*_1d.csv"))}


def weekly_closes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Friday-labelled last close of each week, per index (NaN when the market was shut all week)."""
    cols = {}
    for name, df in data.items():
        s = df.set_index("timestamps")["close"]
        cols[name] = s.resample("W-FRI").last()
    return pd.DataFrame(cols)


def decision_points(df: pd.DataFrame, weeks: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    """For each Friday label, the row index of the index's last bar in that week."""
    ts = df["timestamps"]
    out = {}
    for w in weeks:
        j = ts.searchsorted(w, side="right") - 1
        if j >= 0 and ts.iloc[j] > w - pd.Timedelta(days=7):
            out[w] = int(j)
    return out


# ------------------------------------------------------------------ forecasts
def run_forecasts(cfg, data: dict[str, pd.DataFrame], forecaster, start: str, end: str | None,
                  out: str, progress: bool = True) -> pd.DataFrame:
    weeks = weekly_closes(data).index
    weeks = weeks[(weeks >= pd.Timestamp(start)) & (weeks <= pd.Timestamp(end or weeks.max()))]
    done = pd.read_csv(out, parse_dates=["week"]) if Path(out).exists() else pd.DataFrame()
    done_keys = set(zip(done["week"], done["index"])) if len(done) else set()
    points = {n: decision_points(df, weeks) for n, df in data.items()}
    many = getattr(forecaster, "forecast_many", None)
    rows, t0, n_done = [], time.time(), 0
    for k, w in enumerate(weeks):
        todo = [(n, points[n][w]) for n in data if w in points[n] and (w, n) not in done_keys
                and points[n][w] + 1 >= cfg.model.lookback]
        if not todo:
            continue
        hists = [data[n].iloc[j - cfg.model.lookback + 1: j + 1] for n, j in todo]
        fcs = many(hists) if many else [forecaster.forecast(h) for h in hists]
        for (n, j), fc in zip(todo, fcs):
            rows.append(dict(week=w, index=n, bar=data[n]["timestamps"].iloc[j], ref_close=fc.ref_close,
                             mu=fc.mu, sigma=fc.sigma, p_up=fc.p_up))
        n_done += 1
        if progress and (n_done % 20 == 0 or k == len(weeks) - 1):
            left = (time.time() - t0) / n_done * (len(weeks) - k - 1)
            print(f"[rotation] week {k + 1}/{len(weeks)} {w.date()} | ~{left / 60:.0f} min left", flush=True)
        if n_done % 20 == 0:
            _save(done, rows, out)
    return _save(done, rows, out)


def _save(done, rows, out):
    res = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    if len(res):
        res = res.drop_duplicates(["week", "index"], keep="last").sort_values(["week", "index"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    return res


# ------------------------------------------------------------------ scoring + simulation
def scores(fc: pd.DataFrame, kind: str, z_window: int) -> pd.DataFrame:
    """Week x index score matrix. debiased_mu compares each index's forecast with its own
    previous `z_window` forecasts (removes a persistent bullish/bearish lean per index)."""
    mu = fc.pivot(index="week", columns="index", values="mu").sort_index()
    if kind == "mu":
        return mu
    if kind == "p_up":
        return fc.pivot(index="week", columns="index", values="p_up").sort_index()
    past = mu.shift(1).rolling(z_window, min_periods=max(6, z_window // 2))
    return (mu - past.mean()) / (past.std() + 1e-9)


def simulate(score: pd.DataFrame, closes: pd.DataFrame, top_k: int, start: str, end: str,
             fee: float = 0.001, slip_bps: float = 5, delay: int = 0, buffer: int = 0) -> dict:
    """Weekly rebalance into the top_k scored indices; returns strategy and equal-weight series."""
    rets = closes.pct_change(fill_method=None).shift(-1)       # return over the NEXT week
    weeks = score.index[(score.index >= pd.Timestamp(start)) & (score.index <= pd.Timestamp(end))]
    cost = fee + slip_bps / 10_000
    prev_w = pd.Series(0.0, index=closes.columns)
    prev_ew = pd.Series(0.0, index=closes.columns)
    rows = []
    for w in weeks:
        sig_week = score.index[score.index.get_loc(w) - delay] if delay else w
        s = score.loc[sig_week].dropna()
        r = rets.loc[w] if w in rets.index else pd.Series(dtype=float)
        avail = [c for c in s.index if c in r.index and not np.isnan(r[c])]
        if len(avail) < top_k + 1:
            continue
        ranked = list(s[avail].sort_values(ascending=False).index)
        held = [c for c in prev_w.index[prev_w > 0] if c in ranked[:top_k + buffer]]
        pick = (held + [c for c in ranked if c not in held])[:top_k]
        wts = pd.Series(0.0, index=closes.columns)
        wts[pick] = 1.0 / top_k
        ew = pd.Series(0.0, index=closes.columns)
        ew[avail] = 1.0 / len(avail)
        strat = float((wts[avail] * r[avail]).sum()) - cost * float((wts - prev_w).abs().sum())
        bench = float((ew[avail] * r[avail]).sum()) - cost * float((ew - prev_ew).abs().sum())
        # cross-sectional rank correlation of the score with next-week returns
        ic = float(pd.Series(s[avail]).rank().corr(pd.Series(r[avail]).rank()))
        rows.append(dict(week=w, strategy=strat, equal_weight=bench, ic=ic, picks=",".join(pick),
                         turnover=float((wts - prev_w).abs().sum())))
        prev_w, prev_ew = wts, ew
    return {"weekly": pd.DataFrame(rows)}


def metrics(weekly: pd.DataFrame) -> dict:
    if len(weekly) < 8:
        return {"weeks": len(weekly)}
    s, b = weekly["strategy"], weekly["equal_weight"]
    a = s - b

    def perf(x):
        eq = (1 + x).cumprod()
        vol = x.std(ddof=0) * math.sqrt(52)
        return dict(total_return=round(float(eq.iloc[-1] - 1), 4),
                    sharpe=round(float(x.mean() * 52 / vol), 2) if vol > 0 else 0.0,
                    max_drawdown=round(float((eq / eq.cummax() - 1).min()), 4))

    te = a.std(ddof=0) * math.sqrt(52)
    ic = weekly["ic"].dropna()
    return dict(weeks=len(weekly), strategy=perf(s), equal_weight=perf(b),
                information_ratio=round(float(a.mean() * 52 / te), 2) if te > 0 else 0.0,
                active_return_annual=round(float(a.mean() * 52), 4),
                weeks_beating_ew=round(float((a > 0).mean()), 3),
                mean_weekly_ic=round(float(ic.mean()), 4),
                ic_t_stat=round(float(ic.mean() / (ic.std(ddof=1) / math.sqrt(len(ic)))), 2) if len(ic) > 2 else None,
                avg_turnover=round(float(weekly["turnover"].mean()), 3))


def evaluate(fc: pd.DataFrame, closes: pd.DataFrame, naive: pd.DataFrame | None = None) -> dict:
    out: dict = {"pass_rules": PASS_RULES, "dev": DEV, "test": TEST, "preset": PRESET}

    def run(f, params, window, **kw):
        sc = scores(f, params["score"], params["z_window"])
        return metrics(simulate(sc, closes, params["top_k"], *window, buffer=params["buffer"], **kw)["weekly"])

    # dev: grid search (allowed), pick by information ratio
    grid = []
    for k, sk, zw, bf in itertools.product(GRID["top_k"], GRID["score"], GRID["z_window"], GRID["buffer"]):
        if sk != "debiased_mu" and zw != GRID["z_window"][0]:
            continue                        # z_window only matters for the debiased score
        params = {"top_k": k, "score": sk, "z_window": zw, "buffer": bf}
        m = run(fc, params, DEV)
        grid.append({**params, "dev_ir": m.get("information_ratio", -9), "dev_sharpe": m.get("strategy", {}).get("sharpe")})
    grid.sort(key=lambda g: g["dev_ir"], reverse=True)
    chosen = {k: grid[0][k] for k in ("top_k", "score", "z_window", "buffer")}
    out["dev_grid_top5"] = grid[:5]
    out["chosen_on_dev"] = chosen
    out["dev_preset"] = run(fc, PRESET, DEV)
    # test: evaluated once with the frozen choice (and the preset, for reference)
    out["test_chosen"] = run(fc, chosen, TEST)
    out["test_preset"] = run(fc, PRESET, TEST)
    out["test_chosen_fees_0_25"] = run(fc, chosen, TEST, fee=0.0025)
    out["test_chosen_delay_1w"] = run(fc, chosen, TEST, delay=1)
    if naive is not None:
        out["test_naive_same_settings"] = run(naive, chosen, TEST)
    t = out["test_chosen"]
    if "information_ratio" in t:
        checks = {
            "1 IR >= 0.5": t["information_ratio"] >= 0.5,
            "2 Sharpe > equal-weight": t["strategy"]["sharpe"] > t["equal_weight"]["sharpe"],
            "3 IR > naive": naive is not None and t["information_ratio"] > out["test_naive_same_settings"].get("information_ratio", 9),
            "4 IR > 0 at 0.25% fees": out["test_chosen_fees_0_25"].get("information_ratio", -1) > 0,
        }
        out["checks"] = checks
        out["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast")
    f.add_argument("--config", required=True)
    f.add_argument("--data-dir", required=True)
    f.add_argument("--start", default="2021-06-01")
    f.add_argument("--end")
    f.add_argument("--forecaster", default="kronos", choices=["kronos", "naive"])
    f.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--data-dir", required=True)
    e.add_argument("--forecasts", required=True)
    e.add_argument("--baseline")
    e.add_argument("--out")
    a = p.parse_args(argv)
    data = load_universe(a.data_dir)
    if a.cmd == "forecast":
        cfg = load_config(a.config)
        fcr = build_forecaster(a.forecaster, cfg.model, "1d", "weekdays")
        res = run_forecasts(cfg, data, fcr, a.start, a.end, a.out)
        print(f"wrote {len(res)} forecasts to {a.out}")
        return
    closes = weekly_closes(data)
    fc = pd.read_csv(a.forecasts, parse_dates=["week", "bar"])
    naive = pd.read_csv(a.baseline, parse_dates=["week", "bar"]) if a.baseline else None
    rep = evaluate(fc, closes, naive)
    print(json.dumps(rep, indent=2, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
