"""One-shot evaluation report on cached forecasts.

  python -m autotrader.report --config deploy/config/config.example.yaml \
      --data data/BTCEUR_4h.csv --forecasts runs/fc_kronos.csv \
      --baseline runs/fc_naive.csv --label kronos_btceur --out runs/report_kronos_btceur.json

Prints markdown tables (forecast quality, cost scenarios, tune/test split, Monte Carlo)
and writes the same numbers as JSON. The preset thresholds in the config were chosen
BEFORE looking at any real data, so the "preset" row is a clean out-of-sample test.
The tune/test rows show how much optimising thresholds on old data actually helps on new data.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from .backtest import SCENARIOS, forecast_quality, simulate
from .config import load_config
from .data import load_csv
from .montecarlo import run as monte_carlo

FEE_TIERS = [
    ("Kraken Pro taker 0.40%", 0.0040, 10),
    ("Kraken Pro maker 0.25%", 0.0025, 5),
    ("Bitvavo maker 0.15%", 0.0015, 5),
    ("0.10% tier", 0.0010, 5),
    ("zero cost (signal only)", 0.0, 0),
]


def _row(name: str, s: dict) -> dict:
    st, bh = s["strategy"], s["buy_and_hold"]
    return {"run": name, "end_equity": round(s["end_equity"], 2), "return": round(st["total_return"], 4),
            "sharpe": round(st["sharpe"], 2), "max_dd": round(st["max_drawdown"], 4), "trades": s["trades"],
            "win_rate": s["win_rate"], "fees": round(s["fees_paid"], 2), "exposure": s["exposure"],
            "buy_hold_return": round(bh["total_return"], 4), "halted": bool(s["halted"])}


def _with_costs(cfg, fee, slip):
    c = copy.deepcopy(cfg)
    c.costs.fee_rate, c.costs.slippage_bps = fee, slip
    return c


def evaluate(cfg, df: pd.DataFrame, fc: pd.DataFrame, capital: float = 20.0) -> dict:
    out: dict = {"decisions": len(fc), "period": [str(fc.timestamps.min()), str(fc.timestamps.max())]}
    out["forecast_quality"] = forecast_quality(fc, df, cfg.model.horizon)
    base = simulate(cfg, df, fc, capital)
    out["preset"] = _row("preset config", base["summary"])
    out["stress"] = [_row(n, simulate(cfg, df, fc, capital, **kw)["summary"]) for n, kw in SCENARIOS[1:]]
    out["fee_tiers"] = [_row(n, simulate(_with_costs(cfg, f, s), df, fc, capital)["summary"])
                        for n, f, s in FEE_TIERS]

    # Tune thresholds on the first half, then test ONCE on the untouched second half.
    mid = fc.timestamps.iloc[len(fc) // 2]
    first, second = fc[fc.timestamps < mid], fc[fc.timestamps >= mid]
    grid = []
    for ent in (0.5, 1.0, 1.5, 2.0):
        for pu in (0.52, 0.55, 0.58, 0.62, 0.70):
            c = copy.deepcopy(cfg)
            c.strategy.entry_edge_multiple, c.strategy.min_prob_up = ent, pu
            c.strategy.exit_prob_up = min(c.strategy.exit_prob_up, pu)
            s = simulate(c, df, first, capital)["summary"]
            grid.append((s["strategy"]["sharpe"], s["strategy"]["total_return"], ent, pu, s["trades"]))
    grid.sort(key=lambda g: (g[0], g[1]), reverse=True)
    sh, ret, ent, pu, trades = grid[0]
    c = copy.deepcopy(cfg)
    c.strategy.entry_edge_multiple, c.strategy.min_prob_up = ent, pu
    c.strategy.exit_prob_up = min(c.strategy.exit_prob_up, pu)
    tested = simulate(c, df, second, capital)["summary"]
    preset_second = simulate(cfg, df, second, capital)["summary"]
    out["tune_test"] = {
        "split_at": str(mid),
        "best_on_first_half": {"entry_edge_multiple": ent, "min_prob_up": pu, "sharpe": round(sh, 2),
                               "return": round(ret, 4), "trades": trades},
        "best_params_on_second_half": _row("tuned params, unseen half", tested),
        "preset_on_second_half": _row("preset params, unseen half", preset_second),
    }
    try:
        out["monte_carlo_1y"] = monte_carlo(base["equity"], cfg.exchange.timeframe, capital, 365, 3000, 10, 0.5)
    except ValueError:  # too few bars for the bootstrap block size
        out["monte_carlo_1y"] = None
    out["_equity"] = base["equity"]
    return out


def _md(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_string(index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--data", required=True)
    p.add_argument("--forecasts", required=True)
    p.add_argument("--baseline", help="naive forecasts on the same decision times")
    p.add_argument("--label", default="run")
    p.add_argument("--capital", type=float, default=20.0)
    p.add_argument("--out")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    df = load_csv(a.data)
    fc = pd.read_csv(a.forecasts, parse_dates=["timestamps"])
    rep = {"label": a.label, "model": evaluate(cfg, df, fc, a.capital)}
    if a.baseline:
        base = pd.read_csv(a.baseline, parse_dates=["timestamps"])
        base = base[base.timestamps.isin(fc.timestamps)]
        rep["naive_baseline"] = evaluate(cfg, df, base, a.capital)

    for key in ("model", "naive_baseline"):
        if key not in rep:
            continue
        r = rep[key]
        print(f"\n### {a.label} - {key}  ({r['decisions']} decisions, {r['period'][0]} -> {r['period'][1]})")
        print("forecast quality:", json.dumps(r["forecast_quality"]))
        print(_md([r["preset"]] + r["stress"]))
        print(_md(r["fee_tiers"]))
        tt = r["tune_test"]
        print("tune/test split at", tt["split_at"], "| best on first half:", tt["best_on_first_half"])
        print(_md([tt["best_params_on_second_half"], tt["preset_on_second_half"]]))
        mc = r["monte_carlo_1y"]
        print("monte carlo 1y:", {k: mc[k] for k in ("final_equity_percentiles", "prob_loss",
                                                     "prob_drawdown_over_20pct")} if mc else "skipped (too little data)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        rep["model"]["_equity"].to_csv(Path(a.out).with_suffix(".equity.csv"), index=False)
        clean = {k: ({kk: vv for kk, vv in v.items() if kk != "_equity"} if isinstance(v, dict) else v)
                 for k, v in rep.items()}
        Path(a.out).write_text(json.dumps(clean, indent=2, default=str))


if __name__ == "__main__":
    main()
