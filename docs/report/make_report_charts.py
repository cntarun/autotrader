"""Generate the figures for docs/report/REPORT.md.

  python docs/report/make_report_charts.py            # uses committed results + derived data
  python docs/report/make_report_charts.py --refresh  # first rebuild derived data from local
                                                      # price files and CPU-run forecasts

Inputs:  docs/report/results/*.json  (raw Colab outputs, as pasted from the notebooks)
         docs/report/data/*.csv      (derived series, rebuilt with --refresh)
Outputs: docs/report/img/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES, DATA, IMG = HERE / "results", HERE / "data", HERE / "img"
sys.path.insert(0, str(ROOT))

# Palettes: validated categorical slots 1-3 for each mode (dataviz validator, all pairs),
# neutral inks and a recessive grid. Dark is its own set of steps, not an inversion.
THEMES = {
    "light": dict(KRONOS="#2a78d6", BENCH="#eb6834", NAIVE="#1baf7a", SURFACE="#fcfcfb",
                  INK="#0b0b0b", INK2="#52514e", GRID="#e6e5e0", REF="#8a8983"),
    "dark": dict(KRONOS="#3987e5", BENCH="#d95926", NAIVE="#199e70", SURFACE="#1a1a19",
                 INK="#f0efec", INK2="#c3c2b7", GRID="#383835", REF="#7c7b75"),
}
KRONOS = BENCH = NAIVE = SURFACE = INK = INK2 = GRID = REF = None


def set_theme(name: str) -> None:
    g = globals()
    g.update(THEMES[name])
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 12, "font.size": 10, "legend.frameon": False, "axes.axisbelow": True,
        "legend.labelcolor": INK,
    })


set_theme("light")


def load(name):
    return json.loads((RES / name).read_text())


def save(fig, name):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", IMG / name)


def bar_labels(ax, bars, fmt, horizontal=False, pad=3):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if np.isnan(v):
            continue
        if horizontal:
            ax.annotate(fmt(v), (v, b.get_y() + b.get_height() / 2), xytext=(pad if v >= 0 else -pad, 0),
                        textcoords="offset points", ha="left" if v >= 0 else "right", va="center",
                        fontsize=8, color=INK2)
        else:
            ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, pad if v >= 0 else -pad),
                        textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8, color=INK2)


def bars(ax, x, h, color, label=None, width=0.8, horizontal=False):
    kw = dict(color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
    return ax.barh(x, h, height=width, **kw) if horizontal else ax.bar(x, h, width=width, **kw)


pct = lambda v: f"{v:+.1f}%"  # noqa: E731


# ------------------------------------------------------------------ derived data (--refresh)
def refresh():
    from autotrader.backtest import simulate
    from autotrader.config import load_config
    from autotrader.data import load_csv
    from autotrader.montecarlo import run as mc

    DATA.mkdir(parents=True, exist_ok=True)
    kr = Path("/home/user/src/Kronos")
    # BTC/EUR CPU run (decisions every 12h, real Kronos-small)
    cfg = load_config(ROOT / "deploy/config/config.example.yaml")
    df = load_csv(kr / "data/BTCEUR_4h.csv")
    fk = pd.read_csv(kr / "runs/fc_kronos_btceur.csv", parse_dates=["timestamps"])
    fn = pd.read_csv(kr / "runs/fc_naive_btceur.csv", parse_dates=["timestamps"])
    ek = simulate(cfg, df, fk)["equity"]
    en = simulate(cfg, df, fn)["equity"]
    out = pd.DataFrame({"timestamps": ek.timestamps, "kronos": ek.equity.values,
                        "naive": en.set_index("timestamps").equity.reindex(ek.timestamps).values,
                        "buy_hold": 20 * ek.close.values / ek.close.iloc[0]})
    out.to_csv(DATA / "btc_equity_12h.csv", index=False)
    sims = mc(ek, "4h", 20.0, 365, 5000, 10, 0.5)
    # full final-equity distribution for the histogram
    from autotrader.montecarlo import bootstrap
    from autotrader.data import bars_per_year
    r = ek.equity.pct_change().dropna().to_numpy()
    r = r - 0.5 * max(float(r.mean()), 0.0)
    bpy = bars_per_year("4h")
    paths = bootstrap(r, int(365 * bpy / 365), 5000, max(1, int(10 * bpy / 365)))
    pd.DataFrame({"final_equity": 20 * np.prod(1 + paths, axis=1)}).to_csv(DATA / "btc_mc_final.csv", index=False)
    (DATA / "btc_mc_summary.json").write_text(json.dumps(sims, indent=2))

    # market context: indexed prices over the clean window
    btc = df.set_index("timestamps").close.resample("1D").last()
    series = {"BTC/EUR": btc}
    for name, f in [("S&P 500", "data/universe/SPX_1d.csv"), ("Nifty 50", "data/universe/NIFTY50_1d.csv")]:
        series[name] = load_csv(ROOT / f).set_index("timestamps").close
    ctx = pd.DataFrame({k: v.loc["2025-09-01":"2026-10-26"] for k, v in series.items()}).ffill().dropna()
    (100 * ctx / ctx.iloc[0]).to_csv(DATA / "market_context.csv")

    # share of 5-trading-day periods with a rise, per test window (for the bias chart)
    rows = []
    files = {"S&P 500": "data/universe/SPX_1d.csv", "Nifty 50": "data/universe/NIFTY50_1d.csv",
             "Sensex": "data/SENSEX_1d.csv", "Nifty Bank": "data/NIFTYBANK_1d.csv"}
    windows = {"clean": ("2025-09-01", "2026-10-26"), "long": ("2022-01-03", "2025-08-29")}
    for name, f in files.items():
        c = load_csv(ROOT / f).set_index("timestamps").close
        fwd = c.shift(-5) / c - 1
        for w, (a, b) in windows.items():
            s = fwd.loc[a:b].dropna()
            rows.append({"index": name, "window": w, "share_up_5d": float((s > 0).mean())})
    pd.DataFrame(rows).to_csv(DATA / "share_up.csv", index=False)
    print("derived data refreshed")


# ------------------------------------------------------------------ figures
def fig_scorecard():
    b = load("colab_btceur.json")["runs/report_BTCEUR.json"]
    ix = load("index_spx_nifty.json")
    io = load("india_overlay.json")
    rot = load("rotation.json")
    rows = [
        ("BTC/EUR 4h, buy-or-cash", b["preset"]["return"], b["preset"]["buy_hold_return"], b["naive"][0] / 20 - 1),
        ("S&P 500 2025-26, buy-or-cash", ix["report_SPX_clean.json"]["preset"]["return"],
         ix["report_SPX_clean.json"]["preset"]["buy_hold_return"], ix["report_SPX_clean.json"]["naive"][0] / 1000 - 1),
        ("Nifty 50 2025-26, buy-or-cash", ix["report_NIFTY50_clean.json"]["preset"]["return"],
         ix["report_NIFTY50_clean.json"]["preset"]["buy_hold_return"], ix["report_NIFTY50_clean.json"]["naive"][0] / 1000 - 1),
        ("S&P 500 2022-25*, buy-or-cash", ix["report_SPX_long.json"]["preset"]["return"],
         ix["report_SPX_long.json"]["preset"]["buy_hold_return"], ix["report_SPX_long.json"]["naive"][0] / 1000 - 1),
        ("Nifty 50 2022-25*, buy-or-cash", ix["report_NIFTY50_long.json"]["preset"]["return"],
         ix["report_NIFTY50_long.json"]["preset"]["buy_hold_return"], ix["report_NIFTY50_long.json"]["naive"][0] / 1000 - 1),
        ("Sensex 2025-26, overlay", io["report_SENSEX_clean_index-overlay.json"]["preset"]["return"],
         io["report_SENSEX_clean_index-overlay.json"]["preset"]["buy_hold_return"],
         io["report_SENSEX_clean_index-overlay.json"]["naive"][0] / 1000 - 1),
        ("Nifty Bank 2025-26, overlay", io["report_NIFTYBANK_clean_index-overlay.json"]["preset"]["return"],
         io["report_NIFTYBANK_clean_index-overlay.json"]["preset"]["buy_hold_return"],
         io["report_NIFTYBANK_clean_index-overlay.json"]["naive"][0] / 1000 - 1),
        ("12-index ranking 2025-26", rot["test_chosen"]["strategy"]["total_return"],
         rot["test_chosen"]["equal_weight"]["total_return"], rot["test_naive_same_settings"]["strategy"]["total_return"]),
    ]
    labels = [r[0] for r in rows][::-1]
    vals = np.array([[r[1], r[2], r[3]] for r in rows][::-1]) * 100
    y = np.arange(len(rows))
    h = 0.26
    fig, ax = plt.subplots(figsize=(10, 7.2))
    for k, (col, name) in enumerate([(KRONOS, "Kronos strategy"), (BENCH, "Buy & hold (equal weight for ranking)"),
                                     (NAIVE, "Same rule, random forecasts")]):
        bb = bars(ax, y + (1 - k) * h, vals[:, k], col, name, width=h, horizontal=True)
        bar_labels(ax, bb, pct, horizontal=True)
    ax.axvline(0, color=REF, linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Total return over the test period, after costs (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(vals.min() - 12, vals.max() + 12)
    ax.set_title("Kronos vs simply holding, across every test")
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.01, -0.01, "* 2022-25 may overlap Kronos's training data (optimistic). "
             "Ranking benchmark = all 12 indices equally weighted.", fontsize=8, color=INK2)
    save(fig, "fig01_scorecard.png")


def fig_market_context():
    ctx = pd.read_csv(DATA / "market_context.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for col, colr in zip(ctx.columns, [KRONOS, BENCH, NAIVE]):
        ax.plot(ctx.index, ctx[col], color=colr, linewidth=2, label=col)
        ax.annotate(f"{col}  {ctx[col].iloc[-1] - 100:+.0f}%", (ctx.index[-1], ctx[col].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axhline(100, color=REF, linewidth=1)
    ax.set_ylabel("Price, indexed (1 Sep 2025 = 100)")
    ax.set_title("The test period: three very different markets")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(ctx.index[0], ctx.index[-1] + pd.Timedelta(days=75))
    save(fig, "fig02_market_context.png")


def fig_btc_equity():
    e = pd.read_csv(DATA / "btc_equity_12h.csv", parse_dates=["timestamps"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for col, colr, name in [("kronos", KRONOS, "Kronos"), ("buy_hold", BENCH, "Buy & hold"), ("naive", NAIVE, "Random forecasts")]:
        ax.plot(e.timestamps, e[col], color=colr, linewidth=2, label=name)
        ax.annotate(f"{name}  €{e[col].iloc[-1]:.2f}", (e.timestamps.iloc[-1], e[col].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    changed = e.timestamps[e.kronos.diff().abs() > 1e-9]
    if len(changed):
        halt = changed.iloc[-1]
        ax.axvline(halt, color=REF, linewidth=1)
        ax.annotate("Trading stopped: 20% drawdown limit", (halt, e.kronos.iloc[-1]), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color=INK2, va="bottom")
    ax.axhline(20, color=REF, linewidth=1)
    ax.set_ylabel("Account value (€, started at €20)")
    ax.set_title("BTC/EUR: €20 with Kronos, decisions every 12 hours")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(e.timestamps.iloc[0], e.timestamps.iloc[-1] + pd.Timedelta(days=80))
    save(fig, "fig03_btc_equity.png")


def fig_btc_fees():
    b = load("colab_btceur.json")["runs/report_BTCEUR.json"]
    tiers = b["fee_tiers"]
    names = ["0.40% (Kraken taker)", "0.25% (Kraken maker)", "0.15% (Bitvavo maker)", "0.10%", "Zero cost"]
    vals = [v[0] for v in tiers.values()]
    bh = 20 * (1 + b["preset"]["buy_hold_return"])
    naive = b["naive"][0]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bb = bars(ax, np.arange(5), vals, KRONOS, "Kronos, by fee level", width=0.6)
    bar_labels(ax, bb, lambda v: f"€{v:.2f}")
    ax.axhline(20, color=REF, linewidth=1.2, label="Starting €20")
    ax.axhline(naive, color=NAIVE, linewidth=1.5, label=f"Random forecasts, 0.40% fees (€{naive:.2f})")
    ax.axhline(bh, color=BENCH, linewidth=1.5, label=f"Buy & hold (€{bh:.2f})")
    ax.set_xticks(np.arange(5), names, fontsize=9)
    ax.set_ylim(12, 23)
    ax.set_ylabel("Final value (€)")
    ax.set_title("BTC/EUR: loses money at every fee level, and trails random forecasts")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    save(fig, "fig04_btc_fees.png")


def forecast_rows():
    b = load("colab_btceur.json")["runs/report_BTCEUR.json"]
    ix = load("index_spx_nifty.json")
    io = load("india_overlay.json")
    return [
        ("BTC/EUR 4h", b),
        ("S&P 500 2025-26", ix["report_SPX_clean.json"]),
        ("Nifty 50 2025-26", ix["report_NIFTY50_clean.json"]),
        ("Sensex 2025-26", io["report_SENSEX_clean_index-daily.json"]),
        ("Nifty Bank 2025-26", io["report_NIFTYBANK_clean_index-daily.json"]),
        ("S&P 500 2022-25*", ix["report_SPX_long.json"]),
        ("Nifty 50 2022-25*", ix["report_NIFTY50_long.json"]),
    ]


def fig_forecast_skill():
    rows = forecast_rows()
    names = [r[0] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    bb = bars(ax, x, [r[1]["ic"] for r in rows], KRONOS, width=0.6)
    bar_labels(ax, bb, lambda v: f"{v:+.2f}")
    ax.axhline(0, color=REF, linewidth=1)
    ax.set_xticks(x, names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Rank correlation, forecast vs actual")
    ax.set_title("Ranking skill (0 = none)")
    ax = axes[1]
    bb = bars(ax, x, [100 * r[1]["hit"] for r in rows], KRONOS, width=0.6)
    bar_labels(ax, bb, lambda v: f"{v:.0f}%")
    ax.axhline(50, color=REF, linewidth=1)
    ax.set_ylim(30, 70)
    ax.set_xticks(x, names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Up/down called correctly (%)")
    ax.set_title("Direction calls (line = coin flip)")
    fig.suptitle("Forecast quality: a small ranking signal on indices, none on BTC", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.text(0.01, -0.2, "* may overlap Kronos's training data.", fontsize=8, color=INK2)
    save(fig, "fig05_forecast_skill.png")


def fig_bias():
    share = pd.read_csv(DATA / "share_up.csv")
    ix = load("index_spx_nifty.json")
    io = load("india_overlay.json")
    rows = [("S&P 500 2025-26", "S&P 500", "clean", ix["report_SPX_clean.json"]),
            ("Nifty 50 2025-26", "Nifty 50", "clean", ix["report_NIFTY50_clean.json"]),
            ("Sensex 2025-26", "Sensex", "clean", io["report_SENSEX_clean_index-daily.json"]),
            ("Nifty Bank 2025-26", "Nifty Bank", "clean", io["report_NIFTYBANK_clean_index-daily.json"]),
            ("S&P 500 2022-25", "S&P 500", "long", ix["report_SPX_long.json"]),
            ("Nifty 50 2022-25", "Nifty 50", "long", ix["report_NIFTY50_long.json"])]
    x = np.arange(len(rows))
    up = [100 * share[(share["index"] == i) & (share.window == w)].share_up_5d.iloc[0] for _, i, w, _ in rows]
    inv = [100 * r["preset"]["exposure"] for *_, r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    h = 0.38
    b1 = bars(ax, x - h / 2, up, BENCH, "Weeks the index actually rose", width=h)
    b2 = bars(ax, x + h / 2, inv, KRONOS, "Time Kronos's buy-or-cash rule was invested", width=h)
    bar_labels(ax, b1, lambda v: f"{v:.0f}%")
    bar_labels(ax, b2, lambda v: f"{v:.0f}%")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=9)
    ax.set_ylim(0, 80)
    ax.set_ylabel("%")
    ax.set_title("Bearish lean: Kronos was invested far less often than markets rose")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "fig06_bearish_bias.png")


def fig_calibration():
    rows = forecast_rows()
    names = [r[0] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    h = 0.38
    b1 = bars(ax, x - h / 2, [r[1]["brier"] for r in rows], KRONOS, "Kronos probability", width=h)
    b2 = bars(ax, x + h / 2, [r[1]["brier_clim"] for r in rows], BENCH, "Always predict the base rate", width=h)
    bar_labels(ax, b1, lambda v: f"{v:.2f}")
    bar_labels(ax, b2, lambda v: f"{v:.2f}")
    ax.set_xticks(x, names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_ylim(0, 0.6)
    ax.set_title("Probability accuracy")
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1]
    bb = bars(ax, x, [r[1]["spread_ratio"] for r in rows], KRONOS, width=0.6)
    bar_labels(ax, bb, lambda v: f"{v:.1f}×")
    ax.axhline(1, color=REF, linewidth=1)
    ax.set_xticks(x, names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Real moves ÷ forecast range")
    ax.set_title("Overconfidence (line = well calibrated)")
    fig.suptitle("Calibration: Kronos's probabilities are worse than a constant guess", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=1.03)
    save(fig, "fig07_calibration.png")


def fig_overlay():
    io = load("india_overlay.json")
    names = ["Sensex", "Nifty Bank"]
    keys = ["report_SENSEX_clean_index-overlay.json", "report_NIFTYBANK_clean_index-overlay.json"]
    rs = [io[k] for k in keys]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    h = 0.2
    series = [("Overlay, 0.10% fees", KRONOS, [r["preset"]["sharpe"] for r in rs]),
              ("Buy & hold", BENCH, [r["preset"]["buy_hold_sharpe"] for r in rs]),
              ("Overlay, random forecasts", NAIVE, [r["naive"][1] for r in rs])]
    for k, (n, c, v) in enumerate(series):
        bar_labels(ax, bars(ax, x + (k - 1) * h, v, c, n, width=h), lambda v: f"{v:+.2f}")
    ax.axhline(0, color=REF, linewidth=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("Sharpe ratio (risk-adjusted return)")
    ax.set_title("Risk-adjusted return")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    h = 0.3
    b1 = bars(ax, x - h / 2, [-100 * r["preset"]["max_dd"] for r in rs], KRONOS, "Overlay", width=h)
    b2 = bars(ax, x + h / 2, [-100 * r["preset"]["buy_hold_max_dd"] for r in rs], BENCH, "Buy & hold", width=h)
    bar_labels(ax, b1, lambda v: f"{v:.1f}%")
    bar_labels(ax, b2, lambda v: f"{v:.1f}%")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 26)
    ax.set_ylabel("Largest fall from a peak (%)")
    ax.set_title("Worst drop (lower is better)")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Stay-invested overlay: slightly better than holding, but not robust", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=1.03)
    save(fig, "fig08_overlay.png")


def fig_rotation():
    r = load("rotation.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    names = ["Kronos top 2\n(chosen)", "Kronos top 3\n(preset)", "All 12 equally", "Random\nforecasts"]
    vals = [r["test_chosen"]["strategy"]["total_return"], r["test_preset"]["strategy"]["total_return"],
            r["test_chosen"]["equal_weight"]["total_return"], r["test_naive_same_settings"]["strategy"]["total_return"]]
    cols = [KRONOS, KRONOS, BENCH, NAIVE]
    bb = ax.bar(np.arange(4), np.array(vals) * 100, width=0.6, color=cols, edgecolor=SURFACE, linewidth=1.5)
    bar_labels(ax, bb, pct)
    ax.axhline(0, color=REF, linewidth=1)
    ax.set_xticks(np.arange(4), names, fontsize=9)
    ax.set_ylabel("Total return, Sep 2025 – Oct 2026 (%)")
    ax.set_ylim(-30, 30)
    ax.set_title("Test-period return (60 weeks)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=KRONOS, label="Kronos ranking"), Patch(color=BENCH, label="Equal weight"),
                       Patch(color=NAIVE, label="Random forecasts")], loc="lower left", fontsize=8)
    ax = axes[1]
    names = ["Dev period\n(chosen)", "Test\n(chosen)", "Test, 0.25%\nfees", "Test, 1-week\ndelay", "Test\n(preset)"]
    vals = [r["dev_grid_top5"][0]["dev_ir"], r["test_chosen"]["information_ratio"],
            r["test_chosen_fees_0_25"]["information_ratio"], r["test_chosen_delay_1w"]["information_ratio"],
            r["test_preset"]["information_ratio"]]
    bb = bars(ax, np.arange(5), vals, KRONOS, width=0.6)
    bar_labels(ax, bb, lambda v: f"{v:.2f}")
    ax.axhline(0.5, color=BENCH, linewidth=1.5)
    ax.annotate("pass line 0.5", (-0.45, 0.5), xytext=(0, 3), textcoords="offset points", ha="left", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(5), names, fontsize=9)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Information ratio vs equal weight")
    ax.set_title("Edge over equal weight (random forecasts: −4.56)")
    fig.suptitle("12-index weekly ranking: beat equal weight, but not by the pre-set margin", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=1.03)
    save(fig, "fig09_rotation.png")


def fig_finetune():
    f = load("finetune.json")
    idx = ["NIFTY50", "SPX"]
    names = ["Nifty 50", "S&P 500"]
    x = np.arange(2)
    h = 0.32
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, key, title, fmt in [(axes[0], "ic", "Ranking skill (higher is better)", lambda v: f"{v:.2f}"),
                                (axes[1], "brier", "Brier score (lower is better)", lambda v: f"{v:.2f}")]:
        b1 = bars(ax, x - h / 2, [f[f"{i}_base"][key] for i in idx], BENCH, "Original Kronos", width=h)
        b2 = bars(ax, x + h / 2, [f[f"{i}_finetuned"][key] for i in idx], KRONOS, "Fine-tuned", width=h)
        bar_labels(ax, b1, fmt)
        bar_labels(ax, b2, fmt)
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylim(0, 0.28)
    axes[1].set_ylim(0, 0.75)
    ax = axes[2]
    v = [f["finetune"]["pretrained_val_loss"], f["finetune"]["best_val_loss"]]
    bb = ax.bar([0, 1], v, width=0.55, color=[BENCH, KRONOS], edgecolor=SURFACE, linewidth=1.5)
    bar_labels(ax, bb, lambda v: f"{v:.2f}")
    ax.set_xticks([0, 1], ["Original", "Fine-tuned"])
    ax.set_ylim(0, 3.8)
    ax.set_title("Validation loss, Jan–Aug 2025")
    fig.suptitle("Fine-tuning: a better fit to the data, but not a better forecaster", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=1.03)
    save(fig, "fig10_finetune.png")


def fig_montecarlo():
    d = pd.read_csv(DATA / "btc_mc_final.csv")
    s = json.loads((DATA / "btc_mc_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.hist(d.final_equity, bins=50, color=KRONOS, edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(20, color=BENCH, linewidth=1.5)
    ax.annotate("€20 start", (20, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Account value after one more year (€)")
    ax.set_ylabel("Simulations (of 5,000)")
    ax.set_title(f"BTC/EUR one-year Monte Carlo: {100 * s['prob_loss']:.0f}% of simulations end below €20")
    save(fig, "fig11_btc_montecarlo.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--theme", choices=list(THEMES), default="light")
    p.add_argument("--out", help="output folder (default docs/report/img)")
    a = p.parse_args()
    global IMG
    if a.out:
        IMG = Path(a.out)
    set_theme(a.theme)
    if a.refresh:
        refresh()
    for f in (fig_scorecard, fig_market_context, fig_btc_equity, fig_btc_fees, fig_forecast_skill, fig_bias,
              fig_calibration, fig_overlay, fig_rotation, fig_finetune, fig_montecarlo):
        f()


if __name__ == "__main__":
    main()
