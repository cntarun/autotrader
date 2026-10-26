"""Turn Kronos into a probabilistic forecast: N independent sample paths -> distribution
of the h-bar-ahead return. Also a naive bootstrap forecaster used as the null baseline:
if Kronos can't beat it in backtests, it should not trade real money."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelCfg
from .data import timeframe_to_timedelta

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Forecast:
    ref_close: float          # last closed price the forecast is based on
    mu: float                 # mean predicted h-bar return
    sigma: float              # std of predicted h-bar return across sample paths
    p_up: float               # share of paths ending above ref_close
    paths: np.ndarray         # (samples, horizon) predicted closes

    @classmethod
    def from_paths(cls, ref_close: float, paths: np.ndarray) -> "Forecast":
        rets = paths[:, -1] / ref_close - 1.0
        return cls(ref_close, float(rets.mean()), float(rets.std(ddof=0)), float((rets > 0).mean()), paths)


class KronosForecaster:
    name = "kronos"

    def __init__(self, cfg: ModelCfg, timeframe: str, model=None, tokenizer=None):
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer

        self.cfg = cfg
        self.timeframe = timeframe
        self._torch = torch
        if tokenizer is None:
            tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer, revision=cfg.tokenizer_revision)
        if model is None:
            model = Kronos.from_pretrained(cfg.predictor, revision=cfg.predictor_revision)
        tokenizer.eval()
        model.eval()
        self.predictor = KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)

    def forecast(self, hist: pd.DataFrame) -> Forecast:
        """hist: closed bars only, oldest first, columns from data.COLUMNS."""
        return self.forecast_many([hist])[0]

    def forecast_many(self, hists: list[pd.DataFrame]) -> list[Forecast]:
        """Several decision points in one model call (faster backtests). Each hist must have
        the same length after trimming to `lookback`."""
        cfg = self.cfg
        step = timeframe_to_timedelta(self.timeframe)
        dfs, xts, yts = [], [], []
        trimmed = []
        for hist in hists:
            hist = hist.tail(cfg.lookback).reset_index(drop=True)
            if len(hist) < min(64, cfg.lookback):
                raise ValueError(f"need more history, got {len(hist)} bars")
            last = hist["timestamps"].iloc[-1]
            y_ts = pd.Series([last + step * (i + 1) for i in range(cfg.horizon)])
            x_df = hist[["open", "high", "low", "close", "volume", "amount"]]
            # Repeat each series N times with sample_count=1 to get N separate paths
            # (predict()'s own sample_count averages them away).
            dfs += [x_df] * cfg.samples
            xts += [hist["timestamps"]] * cfg.samples
            yts += [y_ts] * cfg.samples
            trimmed.append(hist)
        self._torch.manual_seed(cfg.seed)
        preds = self.predictor.predict_batch(dfs, xts, yts, pred_len=cfg.horizon, T=cfg.temperature,
                                             top_p=cfg.top_p, sample_count=1, verbose=False)
        closes = np.stack([p["close"].to_numpy() for p in preds])
        out = []
        for k, hist in enumerate(trimmed):
            paths = closes[k * cfg.samples:(k + 1) * cfg.samples]
            out.append(Forecast.from_paths(float(hist["close"].iloc[-1]), paths))
        return out


class NaiveBootstrapForecaster:
    """Null model: future paths are resampled from the last `lookback` bar returns."""
    name = "naive"

    def __init__(self, cfg: ModelCfg, timeframe: str):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def forecast(self, hist: pd.DataFrame) -> Forecast:
        closes = hist["close"].tail(self.cfg.lookback).to_numpy()
        rets = np.diff(np.log(closes))
        draws = self.rng.choice(rets, size=(self.cfg.samples, self.cfg.horizon), replace=True)
        paths = closes[-1] * np.exp(np.cumsum(draws, axis=1))
        return Forecast.from_paths(float(closes[-1]), paths)


def build_forecaster(kind: str, cfg: ModelCfg, timeframe: str):
    if kind == "kronos":
        return KronosForecaster(cfg, timeframe)
    if kind == "naive":
        return NaiveBootstrapForecaster(cfg, timeframe)
    raise ValueError(f"unknown forecaster {kind}")
