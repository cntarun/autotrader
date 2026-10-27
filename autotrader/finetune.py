"""Fine-tune the Kronos predictor on daily stock-index history (tokenizer kept frozen).

Strict time split (no future data in training):
  train: windows ending on or before --train-end           (default 2024-12-31)
  val:   windows whose 5 forecast days fall in --val-start..--val-end (default 2025-01-01..2025-08-29)
  test:  not touched here; done afterwards with the backtest tools (Sep 2025 onwards)

  python -m autotrader.finetune --data-dir data/universe --out models/kronos-small-indices --device cuda

Differences from the repo's finetune_csv pipeline: many series at once, date-based splits, and
normalisation by the context window only (exactly as at inference), with validation loss measured
only on the forecast days. The pretrained model's validation loss is measured first; the
fine-tuned weights are kept only if they beat it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import Kronos, KronosTokenizer  # noqa: E402
from model.kronos import calc_time_stamps  # noqa: E402

from .data import load_csv  # noqa: E402

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


class Windows:
    """All (series, end_index) windows of length lookback + horizon, split by date."""

    def __init__(self, data: dict[str, pd.DataFrame], lookback: int, horizon: int, clip: float = 5.0):
        self.lookback, self.horizon, self.clip = lookback, horizon, clip
        self.L = lookback + horizon
        self.x, self.stamp, self.ts = {}, {}, {}
        for name, df in data.items():
            self.x[name] = df[FEATURES].to_numpy(np.float32)
            self.stamp[name] = calc_time_stamps(df["timestamps"]).to_numpy(np.float32)
            self.ts[name] = df["timestamps"].to_numpy()

    def ends(self, first_target_from=None, last_target_to=None, stride: int = 1) -> list[tuple[str, int]]:
        out = []
        for name, ts in self.ts.items():
            for e in range(self.L - 1, len(ts), stride):
                first_target = ts[e - self.horizon + 1]
                if first_target_from is not None and first_target < np.datetime64(first_target_from):
                    continue
                if last_target_to is not None and ts[e] > np.datetime64(last_target_to):
                    continue
                out.append((name, e))
        return out

    def batch(self, items: list[tuple[str, int]]):
        xs, st = [], []
        for name, e in items:
            w = self.x[name][e - self.L + 1: e + 1]
            ctx = w[: self.lookback]
            mean, std = ctx.mean(axis=0), ctx.std(axis=0)          # context-only, as at inference
            xs.append(np.clip((w - mean) / (std + 1e-5), -self.clip, self.clip))
            st.append(self.stamp[name][e - self.L + 1: e + 1])
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(st))


def step_loss(model, tokenizer, x, stamp, horizon: int | None):
    with torch.no_grad():
        t0, t1 = tokenizer.encode(x, half=True)
    logits = model(t0[:, :-1], t1[:, :-1], stamp[:, :-1, :])
    s1, s2 = logits
    o0, o1 = t0[:, 1:], t1[:, 1:]
    if horizon:  # only the forecast days
        s1, s2, o0, o1 = s1[:, -horizon:], s2[:, -horizon:], o0[:, -horizon:], o1[:, -horizon:]
    loss, _, _ = model.head.compute_loss(s1, s2, o0, o1)
    return loss


@torch.no_grad()
def val_loss(model, tokenizer, wins: Windows, items, device, batch: int, seed: int = 0) -> float:
    model.eval()
    torch.manual_seed(seed)
    tot, n = 0.0, 0
    for i in range(0, len(items), batch):
        x, st = wins.batch(items[i:i + batch])
        loss = step_loss(model, tokenizer, x.to(device), st.to(device), wins.horizon)
        tot += loss.item() * len(x)
        n += len(x)
    return tot / max(n, 1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--tokenizer-revision", default="0e0117387f39004a9016484a186a908917e22426")
    p.add_argument("--predictor", default="NeoQuasar/Kronos-small")
    p.add_argument("--predictor-revision", default="901c26c1332695a2a8f243eb2f37243a37bea320")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--val-start", default="2025-01-01")
    p.add_argument("--val-end", default="2025-08-29")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--steps-per-epoch", type=int, default=250)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = {q.name[:-7]: load_csv(q) for q in sorted(Path(a.data_dir).glob("*_1d.csv"))}
    wins = Windows(data, a.lookback, a.horizon)
    train_items = wins.ends(last_target_to=a.train_end)
    val_items = wins.ends(first_target_from=a.val_start, last_target_to=a.val_end, stride=a.horizon)
    print(f"series={len(data)} train windows={len(train_items)} val windows={len(val_items)}", flush=True)

    tokenizer = KronosTokenizer.from_pretrained(a.tokenizer, revision=a.tokenizer_revision).to(a.device).eval()
    model = Kronos.from_pretrained(a.predictor, revision=a.predictor_revision).to(a.device)
    for prm in tokenizer.parameters():
        prm.requires_grad_(False)

    base = val_loss(model, tokenizer, wins, val_items, a.device, a.batch)
    print(f"pretrained model: val loss {base:.4f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)
    total = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05, div_factor=10)
    best, best_epoch, bad, history = base, 0, 0, [{"epoch": 0, "val_loss": base}]
    out = Path(a.out)
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for _ in range(a.steps_per_epoch):
            idx = rng.integers(0, len(train_items), size=a.batch)
            x, st = wins.batch([train_items[i] for i in idx])
            loss = step_loss(model, tokenizer, x.to(a.device), st.to(a.device), None)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            run += loss.item()
        v = val_loss(model, tokenizer, wins, val_items, a.device, a.batch)
        history.append({"epoch": epoch, "train_loss": run / a.steps_per_epoch, "val_loss": v})
        print(f"epoch {epoch}/{a.epochs} train {run / a.steps_per_epoch:.4f} | val {v:.4f} "
              f"(pretrained {base:.4f}) | {time.time() - t0:.0f}s", flush=True)
        if v < best - 1e-4:
            best, best_epoch, bad = v, epoch, 0
            model.save_pretrained(out)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop", flush=True)
                break

    improved = best_epoch > 0
    meta = {"improved_over_pretrained": improved, "pretrained_val_loss": base, "best_val_loss": best,
            "best_epoch": best_epoch, "history": history, "args": vars(a),
            "series": sorted(data), "train_windows": len(train_items), "val_windows": len(val_items)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "finetune_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    verdict = (f"fine-tuned model saved to {out} (val loss {best:.4f} vs pretrained {base:.4f})" if improved
               else f"fine-tuning did NOT beat the pretrained model (best {best:.4f} vs {base:.4f}); nothing saved")
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
