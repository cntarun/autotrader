"""Builds two self-contained Colab T4 notebooks:

  notebooks/kronos_rotation_colab.ipynb   weekly ranking of 12 global indices (~60-90 min)
  notebooks/kronos_finetune_colab.ipynb   fine-tune Kronos on index history, test on Nifty 50 / S&P 500 (~35-50 min)

Both clone upstream Kronos at a pinned commit and unpack this repo's autotrader/ code, configs and
data/universe/*.csv (download first: python -m autotrader.universe).

  python notebooks/build_research_notebooks.py
"""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted((ROOT / "autotrader").glob("*.py")):
            tar.add(p, arcname=f"autotrader/{p.name}")
        for p in sorted((ROOT / "deploy" / "config").glob("*.yaml")):
            tar.add(p, arcname=f"deploy/config/{p.name}")
        for p in sorted((ROOT / "data" / "universe").glob("*_1d.csv")):
            tar.add(p, arcname=f"data/universe/{p.name}")
    return base64.b64encode(buf.getvalue()).decode()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def setup_cells(b64: str) -> list:
    return [
        code("""
# 1. Check the GPU
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then Run all again"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
"""),
        code(f"""
# 2. Upstream Kronos (pinned) + this repo's autotrader code, configs and 12 index histories
import base64, io, os, sys, tarfile, subprocess
os.chdir("/content")
if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos"], check=True)
os.chdir("/content/Kronos")
subprocess.run(["git", "checkout", "-q", "{UPSTREAM_COMMIT}"], check=True)
BUNDLE = "{b64}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(".")
os.makedirs("runs", exist_ok=True)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
print(sorted(os.listdir("data/universe")))
"""),
        code("""
# 3. Install the few missing packages (Colab already has torch, pandas, numpy)
!pip -q install einops==0.8.1 ccxt==4.5.90 prometheus-client pyyaml safetensors
"""),
        code("""
# 4. GPU copies of the configs (same settings, device=cuda)
import yaml
for name in ["index-daily", "index-overlay"]:
    cfg = yaml.safe_load(open(f"deploy/config/config.{name}.yaml"))
    cfg.setdefault("model", {})["device"] = "cuda"
    yaml.safe_dump(cfg, open(f"runs/gpu_{name}.yaml", "w"))
print("ok")
"""),
    ]


def finish_cell(zip_name: str) -> dict:
    return code(f"""
# Download all outputs
import shutil
shutil.make_archive("/content/{zip_name}", "zip", "runs")
try:
    from google.colab import files; files.download("/content/{zip_name}.zip")
except Exception as e:
    print("download manually from the Files panel:", e)
""")


def write(name: str, cells: list) -> None:
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "notebooks" / name
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


b64 = bundle()

# ---------------------------------------------------------------- notebook A: ranking
rotation = [
    md("""
# Kronos weekly index ranking (GPU)

Each Friday, the **pretrained Kronos-small** forecasts 12 global stock indices one week ahead. We hold
the top-ranked ones and compare with holding all 12 equally, after 0.10% fees + 5 bps slippage.
Settings may be tuned on 2022 → Aug 2025 only; they are then frozen and tested once on
Sep 2025 → Oct 2026. Pass/fail rules were written down before any results (cell 6 checks them).

**Run:** `Runtime → Change runtime type → T4 GPU → Save`, then `Runtime → Run all`.
Cell 5 prints progress with a time estimate (expect roughly 60–90 minutes; keep the tab open).
At the end, copy the lines starting with **`ROTATION_RESULTS`** and **`VERDICT`** into the Claude chat.
"""),
    *setup_cells(b64),
    code("""
# 5. Weekly forecasts for all 12 indices: Kronos, then the naive baseline (fast). Resumable.
import time
from autotrader.config import load_config
from autotrader.forecaster import KronosForecaster, NaiveBootstrapForecaster
from autotrader.rotation import load_universe, run_forecasts
C = load_config("runs/gpu_index-daily.yaml")
data = load_universe("data/universe")
t = time.time()
fk = run_forecasts(C, data, KronosForecaster(C.model, "1d", calendar="weekdays"), "2021-06-01", None, "runs/rot_fc_kronos.csv")
print(f"Kronos: {len(fk)} forecasts in {(time.time()-t)/60:.0f} min", flush=True)
run_forecasts(C, data, NaiveBootstrapForecaster(C.model, "1d"), "2021-06-01", None, "runs/rot_fc_naive.csv", progress=False)
"""),
    code("""
# 6. Evaluate: tune on dev, freeze, test once. COPY THE TWO LINES PRINTED AT THE END.
import json
import pandas as pd
from autotrader.rotation import evaluate, weekly_closes
closes = weekly_closes(data)
fc = pd.read_csv("runs/rot_fc_kronos.csv", parse_dates=["week", "bar"])
nv = pd.read_csv("runs/rot_fc_naive.csv", parse_dates=["week", "bar"])
rep = evaluate(fc, closes, nv)
json.dump(rep, open("runs/rotation_report.json", "w"), indent=2, default=str)
keep = {k: rep.get(k) for k in ["chosen_on_dev", "dev_grid_top5", "dev_preset", "test_chosen", "test_preset",
                                "test_chosen_fees_0_25", "test_chosen_delay_1w", "test_naive_same_settings", "checks"]}
print("ROTATION_RESULTS", json.dumps(keep, separators=(",", ":"), default=str))
print("VERDICT", rep.get("verdict"), rep.get("checks"))
"""),
    finish_cell("kronos_rotation_runs"),
]
write("kronos_rotation_colab.ipynb", rotation)

# ---------------------------------------------------------------- notebook B: fine-tuning
finetune = [
    md("""
# Fine-tuned Kronos for stock indices (GPU)

1. Fine-tune Kronos-small on the daily history of 12 global indices (training data ends 2024-12-31).
2. Check it on Jan → Aug 2025; keep it only if it beats the original model there.
3. Test original vs fine-tuned on **Nifty 50** and **S&P 500**, Sep 2025 → Oct 2026, after costs.

Pass/fail rules were written down before any results; the last cell checks them.

**Run:** `Runtime → Change runtime type → T4 GPU → Save`, then `Runtime → Run all` (roughly 35–50 minutes).
At the end, copy the lines starting with **`FINETUNE_RESULTS`** and **`VERDICT`** into the Claude chat.
"""),
    *setup_cells(b64),
    code("""
# 5. Fine-tune (prints one line per epoch; stops early if validation stops improving)
!python -W ignore -m autotrader.finetune --data-dir data/universe --out models/kronos-small-indices --device cuda
"""),
    code("""
# 6. Test-window forecasts for Nifty 50 and S&P 500 with the original and the fine-tuned model
import copy, json, os, time
from autotrader.backtest import run_forecasts
from autotrader.config import load_config
from autotrader.data import load_csv
from autotrader.forecaster import KronosForecaster, NaiveBootstrapForecaster
meta = json.load(open("models/kronos-small-indices/finetune_meta.json"))
print("fine-tuning:", {k: meta[k] for k in ["improved_over_pretrained", "pretrained_val_loss", "best_val_loss", "best_epoch"]})
C = load_config("runs/gpu_index-daily.yaml")
models = {"base": KronosForecaster(C.model, "1d", calendar="weekdays")}
if meta["improved_over_pretrained"]:
    Cf = copy.deepcopy(C)
    Cf.model.predictor, Cf.model.predictor_revision = "models/kronos-small-indices", None
    models["finetuned"] = KronosForecaster(Cf.model, "1d", calendar="weekdays")
else:
    print("Fine-tuning did not beat the original model on validation data: only the base model is tested.")
naive = NaiveBootstrapForecaster(C.model, "1d")
for idx in ["NIFTY50", "SPX"]:
    df = load_csv(f"data/universe/{idx}_1d.csv")
    for tag, f in models.items():
        t = time.time()
        print(f"--- {idx} {tag}", flush=True)
        run_forecasts(C, df, f, start="2025-09-01", out=f"runs/fc_{tag}_{idx}.csv", batch=32)
        print(f"{idx} {tag}: done in {(time.time()-t)/60:.1f} min", flush=True)
    run_forecasts(C, df, naive, start="2025-09-01", out=f"runs/fc_naive_{idx}.csv", progress=False)
"""),
    code("""
# 7. Reports for every model x strategy x index
for tag in models:
    for strat in ["index-daily", "index-overlay"]:
        for idx in ["NIFTY50", "SPX"]:
            !python -W ignore -m autotrader.report --config runs/gpu_{strat}.yaml --data data/universe/{idx}_1d.csv \\
                --capital 1000 --forecasts runs/fc_{tag}_{idx}.csv --baseline runs/fc_naive_{idx}.csv \\
                --label {idx}_{tag}_{strat} --out runs/report_{idx}_{tag}_{strat}.json > /dev/null
print("reports done")
"""),
    code("""
# 8. Pre-registered checks. COPY THE LINES PRINTED BELOW.
import glob
def load(idx, tag, strat):
    p = f"runs/report_{idx}_{tag}_{strat}.json"
    return json.load(open(p)) if os.path.exists(p) else None
summary = {"finetune": {k: meta[k] for k in ["improved_over_pretrained", "pretrained_val_loss", "best_val_loss", "best_epoch"]}}
for idx in ["NIFTY50", "SPX"]:
    for tag in models:
        q = load(idx, tag, "index-daily")["model"]["forecast_quality"]
        g = lambda k, n: round(q[k], n) if q.get(k) is not None else None
        row = {"ic": g("information_coefficient", 3), "hit": g("directional_hit_rate", 3),
               "brier": g("brier_p_up", 3), "brier_clim": g("brier_climatology", 3),
               "spread_ratio": g("realised_to_forecast_spread", 2)}
        for strat in ["index-daily", "index-overlay"]:
            r = load(idx, tag, strat); p = r["model"]["preset"]
            tier = {t["run"]: t for t in r["model"]["fee_tiers"]}["broker 0.25%"]
            row[strat] = {k: p[k] for k in ["return", "sharpe", "max_dd", "trades", "exposure",
                                            "buy_hold_return", "buy_hold_sharpe", "buy_hold_max_dd"]}
            row[strat]["sharpe_at_0.25pct"] = tier["sharpe"]
            row[strat]["naive_sharpe"] = r["naive_baseline"]["preset"]["sharpe"]
        summary[f"{idx}_{tag}"] = row
print("FINETUNE_RESULTS", json.dumps(summary, separators=(",", ":"), default=str))

if "finetuned" not in models:
    print("VERDICT model improved: NO (fine-tuning did not beat the original model on validation data)")
else:
    def better(i):
        ft, bs = summary[f"{i}_finetuned"], summary[f"{i}_base"]
        if None in (ft["ic"], bs["ic"], ft["brier"], bs["brier"]):
            return False
        return ft["ic"] > bs["ic"] and ft["brier"] < bs["brier"]
    improved = all(better(i) for i in ["NIFTY50", "SPX"])
    def overlay_pass(i):
        o = summary[f"{i}_finetuned"]["index-overlay"]
        return (o["sharpe"] >= o["buy_hold_sharpe"] + 0.2 and o["max_dd"] > o["buy_hold_max_dd"]
                and o["sharpe"] > o["naive_sharpe"] and o["sharpe_at_0.25pct"] >= o["buy_hold_sharpe"] + 0.2)
    usable = improved and all(overlay_pass(i) for i in ["NIFTY50", "SPX"])
    print(f"VERDICT model improved: {'YES' if improved else 'NO'} | usable (overlay passes on both): {'YES' if usable else 'NO'}")
"""),
    finish_cell("kronos_finetune_runs"),
]
write("kronos_finetune_colab.ipynb", finetune)
