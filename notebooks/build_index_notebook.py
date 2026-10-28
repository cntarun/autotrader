"""Builds a self-contained Colab T4 notebook that backtests Kronos on daily stock indices.

It clones upstream Kronos at a pinned commit and unpacks this repo's autotrader/ code, configs and
the bundled daily candles (download them first with autotrader.stockdata).

  # S&P 500 + Nifty 50, buy-or-cash rule, clean + long windows (the original notebook)
  python notebooks/build_index_notebook.py
  # Sensex + Nifty Bank, buy-or-cash AND overlay rules, clean window only
  python notebooks/build_index_notebook.py --datasets SENSEX,NIFTYBANK \
      --configs index-daily,index-overlay --windows clean --out kronos_india_overlay_colab.ipynb
"""
import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
ALL_WINDOWS = {"clean": ("2025-09-01", None), "long": ("2022-01-03", "2025-08-29")}
NAMES = {"SPX": "S&P 500", "NIFTY50": "Nifty 50", "SENSEX": "Sensex", "NIFTYBANK": "Nifty Bank"}

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", default="SPX,NIFTY50", help="names of data/<NAME>_1d.csv files")
ap.add_argument("--configs", default="index-daily", help="deploy/config/config.<name>.yaml, comma-separated")
ap.add_argument("--windows", default="clean,long")
ap.add_argument("--out", default="kronos_index_backtest_colab.ipynb")
args = ap.parse_args()
DATA = {n: f"data/{n}_1d.csv" for n in args.datasets.split(",")}
CONFIGS = args.configs.split(",")
WINDOWS = [(w, *ALL_WINDOWS[w]) for w in args.windows.split(",")]
TITLE = " + ".join(NAMES.get(n, n) for n in DATA)

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for p in sorted((ROOT / "autotrader").glob("*.py")):
        tar.add(p, arcname=f"autotrader/{p.name}")
    for p in sorted((ROOT / "deploy" / "config").glob("*.yaml")):
        tar.add(p, arcname=f"deploy/config/{p.name}")
    for path in DATA.values():
        tar.add(ROOT / path, arcname=path)
bundle = base64.b64encode(buf.getvalue()).decode()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


cells = [
    md(f"""
# Kronos stock-index backtest: {TITLE} (daily, GPU)

Tests the **real Kronos-small model** on daily candles, **after costs** (0.10% per trade + 0.05% slippage,
like a low-cost index-ETF broker), against a naive baseline and buy-and-hold. No keys, no money.

Strategies (`deploy/config/config.<name>.yaml`): {", ".join(CONFIGS)}.
- **index-daily**: buy-or-cash. Invested only when Kronos expects a rise.
- **index-overlay**: invested by default, cut to 50% only on strongly bearish forecasts. Its pass/fail
  criteria were fixed before any results existed; the last cell checks them automatically.

Windows: {", ".join(w for w, _, _ in WINDOWS)} (clean = Sep 2025 → Oct 2026, after Kronos was published).

**Run:** `Runtime → Change runtime type → T4 GPU → Save`, then `Runtime → Run all`.
Cell 6 prints progress with a time estimate. At the end, **copy the line starting with
`KRONOS_INDEX_RESULTS`** (and the verdict lines) into the Claude chat.
"""),
    code("""
# 1. Check the GPU
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then Run all again"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
"""),
    code(f"""
# 2. Upstream Kronos (pinned) + this repo's autotrader code, configs and daily index candles
import base64, io, os, sys, tarfile, subprocess
os.chdir("/content")
if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos"], check=True)
os.chdir("/content/Kronos")
subprocess.run(["git", "checkout", "-q", "{UPSTREAM_COMMIT}"], check=True)
BUNDLE = "{bundle}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(".")
os.makedirs("runs", exist_ok=True)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
print(sorted(os.listdir("autotrader")))
"""),
    code("""
# 3. Install the few missing packages (Colab already has torch, pandas, numpy)
!pip -q install einops==0.8.1 ccxt==4.5.90 prometheus-client pyyaml safetensors
"""),
    code(f"""
# 4. Data (bundled, so results are reproducible; same files as the CPU checks in the chat)
import pandas as pd
DATASETS = {json.dumps(DATA)}
for k, v in DATASETS.items():
    d = pd.read_csv(v, parse_dates=["timestamps"])
    print(f"{{k}}: {{len(d)}} daily bars {{d.timestamps.min().date()}} -> {{d.timestamps.max().date()}}")
"""),
    code(f"""
# 5. GPU copies of the strategy configs (identical model settings, device=cuda)
import yaml
CONFIGS = {CONFIGS!r}
for name in CONFIGS:
    cfg = yaml.safe_load(open(f"deploy/config/config.{{name}}.yaml"))
    cfg.setdefault("model", {{}})["device"] = "cuda"
    yaml.safe_dump(cfg, open(f"runs/gpu_{{name}}.yaml", "w"))
    print(name, cfg["strategy"])
"""),
    code(f"""
# 6. Forecasts. Clean windows first, then long windows. Cached + resumable.
import time
from autotrader.config import load_config
from autotrader.data import load_csv
from autotrader.forecaster import KronosForecaster, NaiveBootstrapForecaster
from autotrader.backtest import run_forecasts

C = load_config(f"runs/gpu_{{CONFIGS[0]}}.yaml")
kronos = KronosForecaster(C.model, "1d", calendar="weekdays")
naive = NaiveBootstrapForecaster(C.model, "1d")
WINDOWS = {WINDOWS!r}
for win, start, end in WINDOWS:
    for name, path in DATASETS.items():
        df = load_csv(path)
        t = time.time()
        print(f"--- {{name}} {{win}}: Kronos forecasts", flush=True)
        fk = run_forecasts(C, df, kronos, start=start, end=end, out=f"runs/fc_kronos_{{name}}_{{win}}.csv", batch=32)
        run_forecasts(C, df, naive, start=start, end=end, out=f"runs/fc_naive_{{name}}_{{win}}.csv", progress=False)
        print(f"{{name}} {{win}}: {{len(fk)}} Kronos forecasts done in {{(time.time()-t)/60:.1f}} min", flush=True)
"""),
    code("""
# 7. Reports for every strategy x index x window (forecasts are shared)
import os
for cname in CONFIGS:
    for win, _, _ in WINDOWS:
        for name, path in DATASETS.items():
            fk = f"runs/fc_kronos_{name}_{win}.csv"
            if not os.path.exists(fk):
                print("skipping", name, win, "(no forecasts)"); continue
            !python -W ignore -m autotrader.report --config runs/gpu_{cname}.yaml --data {path} --capital 1000 \\
                --forecasts {fk} --baseline runs/fc_naive_{name}_{win}.csv \\
                --label {name}_{win}_{cname} --out runs/report_{name}_{win}_{cname}.json
"""),
    code("""
# 8. COPY THE LINE PRINTED BELOW and paste it into the Claude chat
import glob, json, os, shutil
def short(r):
    m = r["model"]; q = m["forecast_quality"]; p = m["preset"]; tt = m["tune_test"]; mc = m["monte_carlo_1y"]
    out = {"decisions": m["decisions"], "period": m["period"],
           "hit": round(q.get("directional_hit_rate", 0), 3), "ic": round(q.get("information_coefficient", 0), 3),
           "brier": round(q.get("brier_p_up", 0), 3), "brier_clim": round(q.get("brier_climatology", 0), 3),
           "spread_ratio": round(q.get("realised_to_forecast_spread", 0), 2),
           "hit_when_mu>1.5%": q.get("hit_rate_when_mu_over_1_5pct"), "n_mu>1.5%": q.get("n_mu_over_1_5pct"),
           "preset": {k: p[k] for k in ("end_equity", "return", "sharpe", "max_dd", "trades", "fees", "exposure", "buy_hold_return", "buy_hold_sharpe", "buy_hold_max_dd") if k in p},
           "fee_tiers": {f["run"]: [f["end_equity"], f["sharpe"], f["trades"]] for f in m["fee_tiers"]},
           "stress": {s["run"]: [s["end_equity"], s["sharpe"]] for s in m["stress"]},
           "tuned_unseen": [tt["best_on_first_half"], tt["best_params_on_second_half"]["return"],
                            tt["preset_on_second_half"]["return"], tt["preset_on_second_half"]["buy_hold_return"]],
           "mc": [mc["final_equity_percentiles"], mc["prob_loss"]] if mc else None}
    if "naive_baseline" in r:
        nb = r["naive_baseline"]
        out["naive"] = [nb["preset"]["end_equity"], nb["preset"]["sharpe"], round(nb["forecast_quality"].get("information_coefficient", 0), 3)]
    return out
summary = {os.path.basename(f): short(json.load(open(f))) for f in sorted(glob.glob("runs/report_*.json"))}
print("KRONOS_INDEX_RESULTS", json.dumps(summary, separators=(",", ":"), default=str))

# Pre-registered pass/fail check for the overlay rule (see deploy/config/config.index-overlay.yaml)
for f in sorted(glob.glob("runs/report_*_index-overlay.json")):
    r = json.load(open(f)); m = r["model"]; p = m["preset"]
    tier = {t["run"]: t for t in m["fee_tiers"]}["broker 0.25%"]
    naive_sharpe = r["naive_baseline"]["preset"]["sharpe"]
    checks = {
        "1 sharpe >= buy&hold + 0.2": p["sharpe"] >= p["buy_hold_sharpe"] + 0.2,
        "2 smaller max drawdown": p["max_dd"] > p["buy_hold_max_dd"],
        "3 beats naive-driven rule": p["sharpe"] > naive_sharpe,
        "4 holds at 0.25% fees": tier["sharpe"] >= tier["buy_hold_sharpe"] + 0.2,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print(f"VERDICT {os.path.basename(f)}: {verdict} | " + " | ".join(f"{k}: {'yes' if v else 'no'}" for k, v in checks.items())
          + f" | sharpe {p['sharpe']} vs b&h {p['buy_hold_sharpe']}, maxDD {p['max_dd']} vs {p['buy_hold_max_dd']}")
shutil.make_archive("/content/kronos_index_runs", "zip", "runs")
try:
    from google.colab import files; files.download("/content/kronos_index_runs.zip")
except Exception as e:
    print("download manually from the Files panel:", e)
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / args.out
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
