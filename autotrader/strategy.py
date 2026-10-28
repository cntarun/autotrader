"""Long-only, no-leverage position rule driven by the forecast distribution."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostsCfg, StrategyCfg
from .forecaster import Forecast


@dataclass
class Decision:
    weight: float
    reason: str


def round_trip_cost(costs: CostsCfg) -> float:
    return 2 * (costs.fee_rate + costs.slippage_bps / 10_000)


def realised_vol_annual(closes: pd.Series, bars_per_year: float, window: int = 60) -> float:
    r = np.diff(np.log(closes.tail(window + 1).to_numpy()))
    return float(r.std(ddof=0) * math.sqrt(bars_per_year)) if len(r) > 1 else float("nan")


def decide_overlay(fc: Forecast, current_weight: float, s: StrategyCfg) -> Decision:
    """Invested by default; step down to `overlay_floor` only on strongly bearish forecasts."""
    tag = f"p_up={fc.p_up:.2f} mu={fc.mu:+.4f}"
    bearish = fc.p_up <= s.overlay_cut_p_up and fc.mu <= s.overlay_cut_mu
    reduced = current_weight < s.max_weight - s.rebalance_band
    if bearish:
        target, why = s.overlay_floor, "cut (strongly bearish)"
    elif reduced and current_weight > 1e-9 and fc.p_up < s.overlay_restore_p_up:
        target, why = s.overlay_floor, "stay reduced"
    else:
        target, why = s.max_weight, "full"
    if abs(target - current_weight) < s.rebalance_band:
        return Decision(current_weight, f"hold {tag}")
    return Decision(target, f"{why} {tag}")


def decide(fc: Forecast, current_weight: float, s: StrategyCfg, costs: CostsCfg,
           realised_vol: float | None = None) -> Decision:
    if s.mode == "overlay":
        return decide_overlay(fc, current_weight, s)
    hurdle = s.entry_edge_multiple * round_trip_cost(costs)
    in_position = current_weight > 1e-9

    if in_position and (fc.p_up < s.exit_prob_up or fc.mu < 0):
        return Decision(0.0, f"exit p_up={fc.p_up:.2f} mu={fc.mu:+.4f}")
    if not in_position and not (fc.mu > hurdle and fc.p_up >= s.min_prob_up):
        return Decision(0.0, f"flat mu={fc.mu:+.4f}<=hurdle {hurdle:.4f} or p_up={fc.p_up:.2f}")

    if s.sizing == "binary":
        w = s.max_weight
    else:  # fractional Kelly on the forecast distribution
        var = max(fc.sigma ** 2, 1e-6)
        w = min(s.max_weight, max(0.0, s.kelly_fraction * fc.mu / var))
    if s.target_vol_annual and realised_vol and realised_vol > 0 and not math.isnan(realised_vol):
        w *= min(1.0, s.target_vol_annual / realised_vol)
    w = float(np.clip(w, 0.0, s.max_weight))

    if in_position and abs(w - current_weight) < s.rebalance_band:
        return Decision(current_weight, f"hold p_up={fc.p_up:.2f} mu={fc.mu:+.4f}")
    return Decision(w, f"{'resize' if in_position else 'enter'} p_up={fc.p_up:.2f} mu={fc.mu:+.4f}")
