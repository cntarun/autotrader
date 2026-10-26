"""Builds notebooks/kronos_backtest_colab.ipynb: a self-contained GPU backtest for Google Colab.

The notebook clones upstream Kronos at a pinned commit (for model/), and unpacks this branch's
autotrader/ code, configs and the BTC/EUR candle file from an embedded archive, so it does not
need access to any private repo. Re-run this script after changing autotrader/.

  python notebooks/build_colab_notebook.py
"""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for p in sorted((ROOT / "autotrader").glob("*.py")):
        tar.add(p, arcname=f"autotrader/{p.name}")
    for p in sorted((ROOT / "deploy" / "config").glob("*.yaml")):
        tar.add(p, arcname=f"deploy/config/{p.name}")
    tar.add(ROOT / "data" / "BTCEUR_4h.csv", arcname="data/BTCEUR_4h.csv")
bundle = base64.b64encode(buf.getvalue()).decode()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


cells = [
    md("""
# Kronos real-model backtest (GPU)

Backtests the **real Kronos-small model** (pinned weights) as a long-only BTC trading signal,
**after real fees**, against a naive baseline and buy-and-hold. No API keys, no money, read-only
public market data.

**Before running:** `Runtime → Change runtime type → T4 GPU → Save`, then `Runtime → Run all`.
It takes roughly 10–20 minutes. At the end, **copy the text printed by the last cell** and paste it back
into the Claude chat. A `kronos_runs.zip` with all outputs also downloads.

Not financial advice: this measures whether the signal had an edge in the past. It does not promise one.
"""),
    code("""
# 1. Check the GPU
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then Run all again"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
"""),
    code(f"""
# 2. Get upstream Kronos (pinned commit) + unpack the autotrader code and BTC/EUR data
import base64, io, os, tarfile, subprocess
os.chdir("/content")
if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos"], check=True)
os.chdir("/content/Kronos")
subprocess.run(["git", "checkout", "-q", "{UPSTREAM_COMMIT}"], check=True)
BUNDLE = "{bundle}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(".")
os.makedirs("runs", exist_ok=True)
import sys
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
print(sorted(os.listdir("autotrader")))
"""),
    code("""
# 3. Install the few missing packages (Colab already has torch, pandas, numpy)
!pip -q install einops==0.8.1 ccxt==4.5.90 prometheus-client pyyaml safetensors
"""),
    code("""
# 4. Market data. BTC/EUR is bundled (same file used in the CPU run). BTC/USD and ETH/USD are
#    downloaded from public endpoints: Bitstamp 4h, else Coinbase 1h resampled to 4h.
import pandas as pd
from autotrader.data import make_exchange, fetch_ohlcv, normalise

def get_4h(symbol, since="2024-06-01"):
    try:
        return fetch_ohlcv(make_exchange("bitstamp"), symbol, "4h", pd.Timestamp(since, tz="UTC"), 20000)
    except Exception as e:
        print(f"bitstamp failed for {symbol}: {str(e)[:80]} -> trying coinbase 1h")
    h = fetch_ohlcv(make_exchange("coinbase"), symbol, "1h", pd.Timestamp(since, tz="UTC"), 80000)
    h = h.set_index("timestamps").resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"}
    ).dropna().reset_index()
    return normalise(h)

DATASETS = {"BTCEUR": "data/BTCEUR_4h.csv"}
for sym in ["BTC/USD", "ETH/USD"]:
    name = sym.replace("/", "")
    try:
        d = get_4h(sym)
        if len(d) > 2000:
            d.to_csv(f"data/{name}_4h.csv", index=False)
            DATASETS[name] = f"data/{name}_4h.csv"
    except Exception as e:
        print(f"skipping {sym}: {str(e)[:120]}")
for k, v in DATASETS.items():
    d = pd.read_csv(v, parse_dates=["timestamps"])
    print(f"{k}: {len(d)} bars {d.timestamps.min()} -> {d.timestamps.max()}")
"""),
    code("""
# 5. GPU config: identical to deploy/config/config.example.yaml except device=cuda
import yaml
cfg = yaml.safe_load(open("deploy/config/config.example.yaml"))
cfg["model"]["device"] = "cuda"
cfg["paths"] = {"state_dir": "./state", "kill_switch_file": "./state/KILL"}
yaml.safe_dump(cfg, open("runs/gpu.yaml", "w"))
print(cfg["model"]); print(cfg["strategy"]); print(cfg["costs"])
"""),
    code("""
# 6. Forecasts: every 4h bar from 2025-09-01 (after Kronos was published) to the end of the data.
#    Kronos on GPU, naive baseline on the same bars. Cached + resumable: re-running skips done bars.
import time
from autotrader.config import load_config
from autotrader.data import load_csv
from autotrader.forecaster import KronosForecaster, NaiveBootstrapForecaster
from autotrader.backtest import run_forecasts

C = load_config("runs/gpu.yaml")
kronos = KronosForecaster(C.model, "4h")
naive = NaiveBootstrapForecaster(C.model, "4h")
for name, path in DATASETS.items():
    df = load_csv(path)
    t = time.time()
    fk = run_forecasts(C, df, kronos, start="2025-09-01", out=f"runs/fc_kronos_{name}.csv", batch=32, progress=False)
    print(f"{name}: {len(fk)} Kronos forecasts in {time.time()-t:.0f}s")
    run_forecasts(C, df, naive, start="2025-09-01", out=f"runs/fc_naive_{name}.csv", progress=False)
"""),
    code("""
# 7. Reports: forecast quality, fees, stress tests, tune/test split, Monte Carlo
for name, path in DATASETS.items():
    !python -W ignore -m autotrader.report --config runs/gpu.yaml --data {path} \\
        --forecasts runs/fc_kronos_{name}.csv --baseline runs/fc_naive_{name}.csv \\
        --label {name} --out runs/report_{name}.json

# Cross-check against the CPU run in the chat (decisions only at 00:00 and 12:00 UTC)
fc = pd.read_csv("runs/fc_kronos_BTCEUR.csv", parse_dates=["timestamps"])
fc[fc.timestamps.dt.hour.isin([0, 12])].to_csv("runs/fc_kronos_BTCEUR_12h.csv", index=False)
nv = pd.read_csv("runs/fc_naive_BTCEUR.csv", parse_dates=["timestamps"])
nv[nv.timestamps.dt.hour.isin([0, 12])].to_csv("runs/fc_naive_BTCEUR_12h.csv", index=False)
!python -W ignore -m autotrader.report --config runs/gpu.yaml --data data/BTCEUR_4h.csv \\
    --forecasts runs/fc_kronos_BTCEUR_12h.csv --baseline runs/fc_naive_BTCEUR_12h.csv \\
    --label BTCEUR_12h --out runs/report_BTCEUR_12h.json
"""),
    md("""
### 8. Diagnostic (optional): wider sampling

The early CPU results suggested the forecast paths are too narrow (overconfident). This cell re-runs
BTC/EUR with `top_p=1.0` and `temperature=1.2` to see whether the spread becomes realistic. It is
a **diagnostic**, not a setting to adopt on this evidence alone.
"""),
    code("""
import copy
C2 = copy.deepcopy(C)
C2.model.top_p, C2.model.temperature = 1.0, 1.2
k2 = KronosForecaster(C2.model, "4h")
df = load_csv(DATASETS["BTCEUR"])
run_forecasts(C2, df, k2, start="2025-09-01", out="runs/fc_kronos_BTCEUR_wide.csv", batch=32, progress=False)
!python -W ignore -m autotrader.report --config runs/gpu.yaml --data data/BTCEUR_4h.csv \\
    --forecasts runs/fc_kronos_BTCEUR_wide.csv --baseline runs/fc_naive_BTCEUR.csv \\
    --label BTCEUR_wide_sampling --out runs/report_BTCEUR_wide.json
"""),
    code("""
# 9. COPY EVERYTHING PRINTED BELOW and paste it into the Claude chat
import glob, json, shutil
def short(r):
    m = r["model"]; q = m["forecast_quality"]; p = m["preset"]; tt = m["tune_test"]
    out = {"decisions": m["decisions"], "period": m["period"],
           "hit": round(q.get("directional_hit_rate", 0), 3), "ic": round(q.get("information_coefficient", 0), 3),
           "brier": round(q.get("brier_p_up", 0), 3), "brier_clim": round(q.get("brier_climatology", 0), 3),
           "spread_ratio": round(q.get("realised_to_forecast_spread", 0), 2),
           "hit_when_mu>1.5%": q.get("hit_rate_when_mu_over_1_5pct"), "n_mu>1.5%": q.get("n_mu_over_1_5pct"),
           "preset": {k: p[k] for k in ("end_equity", "return", "sharpe", "max_dd", "trades", "fees", "buy_hold_return")},
           "fee_tiers": {f["run"]: [f["end_equity"], f["sharpe"], f["trades"]] for f in m["fee_tiers"]},
           "stress": {s["run"]: [s["end_equity"], s["sharpe"]] for s in m["stress"]},
           "tuned_unseen": [tt["best_on_first_half"], tt["best_params_on_second_half"]["return"],
                            tt["preset_on_second_half"]["return"], tt["preset_on_second_half"]["buy_hold_return"]],
           "mc": [m["monte_carlo_1y"]["final_equity_percentiles"], m["monte_carlo_1y"]["prob_loss"]]
                 if m["monte_carlo_1y"] else None}
    if "naive_baseline" in r:
        nb = r["naive_baseline"]
        out["naive"] = [nb["preset"]["end_equity"], nb["preset"]["sharpe"], round(nb["forecast_quality"].get("information_coefficient", 0), 3)]
    return out
summary = {r: short(json.load(open(r))) for r in sorted(glob.glob("runs/report_*.json"))}
print("KRONOS_COLAB_RESULTS", json.dumps(summary, separators=(",", ":"), default=str))
shutil.make_archive("/content/kronos_runs", "zip", "runs")
try:
    from google.colab import files; files.download("/content/kronos_runs.zip")
except Exception as e:
    print("download manually from the Files panel:", e)
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "kronos_backtest_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size // 1024} KB, bundle {len(bundle) // 1024} KB)")
