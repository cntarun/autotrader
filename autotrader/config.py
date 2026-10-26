"""Config loading. Non-secret settings live in config.yaml; secrets only in env vars."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

LIVE_CONFIRM_PHRASE = "I_ACCEPT_REAL_MONEY_RISK"


@dataclass
class ExchangeCfg:
    id: str = "kraken"                       # any ccxt exchange id
    symbols: list[str] = field(default_factory=lambda: ["BTC/USD"])
    timeframe: str = "4h"
    sandbox: bool = False                    # use exchange testnet if ccxt supports it


@dataclass
class ModelCfg:
    tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    predictor: str = "NeoQuasar/Kronos-small"
    # Pinned Hugging Face revisions (same ones the repo's regression test uses).
    tokenizer_revision: str | None = "0e0117387f39004a9016484a186a908917e22426"
    predictor_revision: str | None = "901c26c1332695a2a8f243eb2f37243a37bea320"
    device: str = "cpu"
    max_context: int = 512
    lookback: int = 400                      # bars of history fed to the model
    horizon: int = 6                         # bars ahead to forecast (6 x 4h = 1 day)
    samples: int = 16                        # independent sample paths per decision
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 42


@dataclass
class StrategyCfg:
    entry_edge_multiple: float = 1.5         # expected return must beat this x round-trip cost
    min_prob_up: float = 0.58                # share of sample paths ending higher, to enter
    exit_prob_up: float = 0.48               # exit when it falls below this
    max_weight: float = 1.0                  # max share of the sleeve in the coin
    sizing: str = "binary"                   # binary | kelly
    kelly_fraction: float = 0.25
    target_vol_annual: float | None = None   # e.g. 0.5 -> scale down when realised vol is higher
    rebalance_band: float = 0.10             # ignore weight changes smaller than this


@dataclass
class CostsCfg:
    fee_rate: float = 0.004                  # per side (Kraken Pro taker at low volume ~0.40%)
    slippage_bps: float = 10.0               # per side


@dataclass
class RiskCfg:
    capital_cap: float = 20.0                # bot never manages more quote currency than this
    min_order_notional: float = 5.0
    max_order_notional: float = 25.0
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 20.0
    max_trades_per_day: int = 6
    max_consecutive_losses: int = 5
    max_data_staleness_bars: int = 2
    max_forecast_abs_return: float = 0.25    # forecasts beyond +/-25% are treated as broken
    max_slippage_bps: float = 100.0
    reconcile_tolerance_pct: float = 2.0


@dataclass
class PathsCfg:
    state_dir: str = "./state"
    kill_switch_file: str = "./state/KILL"


@dataclass
class RuntimeCfg:
    loop_delay_seconds: int = 90             # wait after bar close before deciding
    health_port: int = 9108


@dataclass
class Config:
    mode: str = "paper"                      # paper | live
    paper_starting_quote: float = 20.0
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    costs: CostsCfg = field(default_factory=CostsCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(dc)}
    for key, value in (data or {}).items():
        if key not in known:
            raise ValueError(f"Unknown config key: {key!r} in {type(dc).__name__}")
        current = getattr(dc, key)
        if is_dataclass(current):
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    path = path or os.environ.get("KRONOS_CONFIG")
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    # A few safe env overrides (useful in CI/containers).
    if os.environ.get("KRONOS_MODE"):
        cfg.mode = os.environ["KRONOS_MODE"]
    if os.environ.get("KRONOS_STATE_DIR"):
        cfg.paths.state_dir = os.environ["KRONOS_STATE_DIR"]
        cfg.paths.kill_switch_file = str(Path(cfg.paths.state_dir) / "KILL")
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.mode not in ("paper", "live"):
        raise ValueError("mode must be 'paper' or 'live'")
    if cfg.strategy.sizing not in ("binary", "kelly"):
        raise ValueError("strategy.sizing must be 'binary' or 'kelly'")
    if not 0 < cfg.strategy.max_weight <= 1:
        raise ValueError("strategy.max_weight must be in (0, 1] - no leverage allowed")
    if cfg.strategy.exit_prob_up > cfg.strategy.min_prob_up:
        raise ValueError("exit_prob_up must be <= min_prob_up")
    if cfg.model.lookback > cfg.model.max_context:
        raise ValueError("model.lookback must be <= model.max_context")
    if cfg.risk.min_order_notional > cfg.risk.max_order_notional:
        raise ValueError("risk.min_order_notional must be <= risk.max_order_notional")
    if cfg.mode == "live":
        if os.environ.get("KRONOS_LIVE_CONFIRM") != LIVE_CONFIRM_PHRASE:
            raise PermissionError(
                f"Live mode needs env KRONOS_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}. Refusing to start."
            )
        hard_cap = float(os.environ.get("KRONOS_HARD_CAP", "25"))
        if cfg.risk.capital_cap > hard_cap:
            raise PermissionError(
                f"risk.capital_cap={cfg.risk.capital_cap} exceeds KRONOS_HARD_CAP={hard_cap}."
            )
