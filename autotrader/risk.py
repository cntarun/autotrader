"""Kill switch, circuit breakers and per-order limits. A halt persists until a human resets it."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RiskCfg
from .forecaster import Forecast
from .state import BotState


@dataclass
class KillSwitch:
    path: str

    def status(self) -> str | None:
        """Returns None, 'HALT' or 'FLATTEN'."""
        if os.environ.get("KRONOS_KILL", "").strip() in ("1", "true", "HALT"):
            return "HALT"
        if os.environ.get("KRONOS_KILL", "").strip() == "FLATTEN":
            return "FLATTEN"
        p = Path(self.path)
        if p.exists():
            return "FLATTEN" if p.read_text().strip().upper() == "FLATTEN" else "HALT"
        return None


class RiskManager:
    def __init__(self, cfg: RiskCfg):
        self.cfg = cfg

    @staticmethod
    def halt(st: BotState, reason: str) -> None:
        if not st.halted:
            st.halted, st.halt_reason = True, reason

    def roll_day(self, st: BotState, now: pd.Timestamp, equity: float) -> None:
        day = now.strftime("%Y-%m-%d")
        if st.day != day:
            st.day, st.day_start_equity, st.trades_today = day, equity, 0

    def check_portfolio(self, st: BotState, equity: float) -> None:
        st.peak_equity = max(st.peak_equity, equity)
        dd = 100 * (1 - equity / st.peak_equity) if st.peak_equity > 0 else 0.0
        day_loss = 100 * (1 - equity / st.day_start_equity) if st.day_start_equity > 0 else 0.0
        if dd >= self.cfg.max_drawdown_pct:
            self.halt(st, f"max_drawdown {dd:.1f}% >= {self.cfg.max_drawdown_pct}%")
        elif day_loss >= self.cfg.max_daily_loss_pct:
            self.halt(st, f"daily_loss {day_loss:.1f}% >= {self.cfg.max_daily_loss_pct}%")
        elif st.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.halt(st, f"consecutive_losses {st.consecutive_losses}")

    def check_data(self, last_bar: pd.Timestamp, now: pd.Timestamp, bar: pd.Timedelta,
                   calendar: str = "24/7") -> str | None:
        if calendar == "weekdays" and bar == pd.Timedelta(days=1):
            # trading days elapsed since the bar after last_bar (weekends don't count)
            nxt = (last_bar.normalize() + pd.offsets.BDay(1)).date()
            age_bars = float(np.busday_count(nxt, now.date()))
        else:
            age_bars = (now - (last_bar + bar)) / bar
        if age_bars > self.cfg.max_data_staleness_bars:
            return f"stale_data last_bar={last_bar} age={age_bars:.1f} bars"
        return None

    def check_forecast(self, fc: Forecast) -> str | None:
        if not all(map(lambda v: v == v, (fc.mu, fc.sigma, fc.p_up))):  # NaN check
            return "forecast_nan"
        if abs(fc.mu) > self.cfg.max_forecast_abs_return:
            return f"forecast_outlier mu={fc.mu:+.3f}"
        return None

    def clamp_order(self, st: BotState, notional: float) -> tuple[float, str | None]:
        """Returns (allowed_notional, reason_if_blocked). notional is signed (+buy/-sell).
        Sells only reduce risk (long-only), so they skip the halt and trade-count limits."""
        is_sell = notional < 0
        if not is_sell and st.halted:
            return 0.0, f"halted: {st.halt_reason}"
        if not is_sell and st.trades_today >= self.cfg.max_trades_per_day:
            return 0.0, "max_trades_per_day"
        if abs(notional) < self.cfg.min_order_notional:
            return 0.0, f"below_min_notional {abs(notional):.2f}"
        if is_sell:
            return notional, None
        return min(self.cfg.max_order_notional, notional), None
