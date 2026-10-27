"""Offline tests for the autotrader: a tiny randomly-initialised Kronos + synthetic prices.
These check plumbing and safety logic, NOT trading performance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import run_forecasts, simulate
from autotrader.config import LIVE_CONFIRM_PHRASE, Config, load_config, validate
from autotrader.forecaster import Forecast, KronosForecaster, NaiveBootstrapForecaster
from autotrader.montecarlo import run as mc_run
from autotrader.risk import RiskManager
from autotrader.run import Bot
from autotrader.state import BotState
from autotrader.strategy import decide


def synthetic(n=700, seed=0, drift=0.0005, vol=0.01, tf="4h"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n))
    vol_ = rng.uniform(10, 100, n)
    ts = pd.date_range("2024-01-01", periods=n, freq=tf)
    return pd.DataFrame(dict(timestamps=ts, open=open_, high=high, low=low, close=close,
                             volume=vol_, amount=vol_ * close))


def small_cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.model.lookback, cfg.model.horizon, cfg.model.samples = 64, 3, 4
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.paths.kill_switch_file = str(tmp_path / "state" / "KILL")
    return cfg


@pytest.fixture(scope="module")
def tiny_kronos():
    from model import Kronos, KronosTokenizer
    tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
                          ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=4, s2_bits=4,
                          beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4)
    mdl = Kronos(s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0,
                 attn_dropout_p=0, resid_dropout_p=0, token_dropout_p=0, learn_te=True)
    return mdl, tok


def test_kronos_forecaster_returns_distribution(tiny_kronos, tmp_path):
    cfg = small_cfg(tmp_path)
    mdl, tok = tiny_kronos
    f = KronosForecaster(cfg.model, "4h", model=mdl, tokenizer=tok)
    fc = f.forecast(synthetic(200))
    assert fc.paths.shape == (cfg.model.samples, cfg.model.horizon)
    assert 0.0 <= fc.p_up <= 1.0 and np.isfinite(fc.mu) and fc.sigma >= 0
    # same seed -> same forecast (reproducible audits)
    assert f.forecast(synthetic(200)).mu == pytest.approx(fc.mu)


def test_strategy_respects_cost_hurdle_and_no_leverage(tmp_path):
    cfg = small_cfg(tmp_path)
    tiny_edge = Forecast(100, 0.001, 0.01, 0.9, np.empty((0, 0)))   # below ~1.4% hurdle
    assert decide(tiny_edge, 0.0, cfg.strategy, cfg.costs).weight == 0.0
    big_edge = Forecast(100, 0.05, 0.02, 0.9, np.empty((0, 0)))
    d = decide(big_edge, 0.0, cfg.strategy, cfg.costs)
    assert 0 < d.weight <= 1.0
    bearish = Forecast(100, -0.01, 0.02, 0.3, np.empty((0, 0)))
    assert decide(bearish, 1.0, cfg.strategy, cfg.costs).weight == 0.0
    cfg.strategy.max_weight = 2.0
    with pytest.raises(ValueError):
        validate(cfg)


def test_live_mode_requires_explicit_confirmation(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("mode: live\n")
    monkeypatch.delenv("KRONOS_LIVE_CONFIRM", raising=False)
    with pytest.raises(PermissionError):
        load_config(p)
    monkeypatch.setenv("KRONOS_LIVE_CONFIRM", LIVE_CONFIRM_PHRASE)
    monkeypatch.setenv("KRONOS_HARD_CAP", "10")
    with pytest.raises(PermissionError):          # capital_cap 20 > hard cap 10
        load_config(p)


def test_risk_manager_breakers_and_order_limits():
    rm = RiskManager(Config().risk)
    st = BotState(peak_equity=20, day_start_equity=20)
    rm.check_portfolio(st, 15.9)                   # -20.5% drawdown
    assert st.halted and "max_drawdown" in st.halt_reason
    assert rm.clamp_order(st, 10)[0] == 0          # no buys while halted
    assert rm.clamp_order(st, -10)[0] == -10       # sells (risk-reducing) still allowed
    st2 = BotState()
    assert rm.clamp_order(st2, 3)[1].startswith("below_min_notional")
    assert rm.clamp_order(st2, 100)[0] == 25       # max_order_notional


class FakeForecaster:
    def __init__(self, mu, p_up):
        self.mu, self.p_up = mu, p_up

    def forecast(self, hist):
        return Forecast(float(hist.close.iloc[-1]), self.mu, 0.02, self.p_up, np.empty((0, 0)))


def make_bot(tmp_path, fc, n=300):
    cfg = small_cfg(tmp_path)
    df = synthetic(n)
    now = df.timestamps.iloc[-1] + pd.Timedelta(hours=4, minutes=2)
    return Bot(cfg, forecaster=fc, data_source=lambda s, k: df.tail(k).reset_index(drop=True),
               now_fn=lambda: now), cfg


def test_paper_cycle_buys_then_is_idempotent(tmp_path):
    bot, cfg = make_bot(tmp_path, FakeForecaster(0.05, 0.9))
    rep = bot.cycle()
    fill = rep["symbols"]["BTC/USD"]["fill"]
    assert fill["side"] == "buy" and fill["notional"] <= cfg.risk.capital_cap
    st = bot.store.load()
    assert st.quote >= 0 and st.positions["BTC/USD"] > 0 and st.fees_paid > 0
    rep2 = bot.cycle()                              # same bar again -> no new trade
    assert rep2["symbols"]["BTC/USD"].get("skipped") == "bar already processed"
    assert (tmp_path / "state" / "fills.csv").exists()
    assert (tmp_path / "state" / "metrics.prom").read_text().count("kronos_equity_quote") >= 1


def test_kill_switch_flatten_and_halt(tmp_path):
    bot, cfg = make_bot(tmp_path, FakeForecaster(0.05, 0.9))
    bot.cycle()
    (tmp_path / "state" / "KILL").write_text("FLATTEN")
    rep = bot.cycle()
    assert rep["symbols"]["BTC/USD"]["fill"]["side"] == "sell"
    st = bot.store.load()
    assert st.halted and st.positions["BTC/USD"] == pytest.approx(0, abs=1e-12)
    (tmp_path / "state" / "KILL").write_text("HALT")
    rep = bot.cycle()
    assert rep.get("note", "").startswith("kill switch HALT")


def test_stale_data_blocks_trading(tmp_path):
    cfg = small_cfg(tmp_path)
    df = synthetic(300)
    later = df.timestamps.iloc[-1] + pd.Timedelta(days=2)
    bot = Bot(cfg, forecaster=FakeForecaster(0.05, 0.9), data_source=lambda s, k: df.tail(k),
              now_fn=lambda: later)
    rep = bot.cycle()
    assert rep["symbols"]["BTC/USD"]["skipped"].startswith("stale_data")


def test_backtest_and_montecarlo_end_to_end(tiny_kronos, tmp_path):
    cfg = small_cfg(tmp_path)
    df = synthetic(260)
    mdl, tok = tiny_kronos
    fc_k = run_forecasts(cfg, df, KronosForecaster(cfg.model, "4h", mdl, tok), step=4, progress=False)
    fc_n = run_forecasts(cfg, df, NaiveBootstrapForecaster(cfg.model, "4h"), progress=False)
    for fc in (fc_k, fc_n):
        res = simulate(cfg, df, fc)
        s = res["summary"]
        assert s["end_equity"] > 0 and s["bars"] == len(res["equity"])
        assert {"sharpe", "max_drawdown", "cagr"} <= set(s["strategy"])
    mc = mc_run(res["equity"], "4h", 20.0, 90, 200, 2, 0.5)
    assert 0 <= mc["prob_loss"] <= 1


def test_no_lookahead_in_forecasts(tmp_path):
    """Changing future bars must not change a past forecast."""
    cfg = small_cfg(tmp_path)
    df = synthetic(200)
    df2 = df.copy()
    df2.loc[150:, ["open", "high", "low", "close"]] *= 3
    a = run_forecasts(cfg, df, NaiveBootstrapForecaster(cfg.model, "4h"), progress=False)
    b = run_forecasts(cfg, df2, NaiveBootstrapForecaster(cfg.model, "4h"), progress=False)
    cut = df.timestamps.iloc[149]
    pd.testing.assert_frame_equal(a[a.timestamps <= cut], b[b.timestamps <= cut])


def test_weekday_calendar_for_stock_markets():
    from autotrader.data import bars_per_year, future_timestamps
    from autotrader.risk import RiskManager
    fri = pd.Timestamp("2026-10-23")                     # a Friday
    ts = future_timestamps(fri, "1d", 3, "weekdays")
    assert [t.day_name() for t in ts] == ["Monday", "Tuesday", "Wednesday"]
    assert list(future_timestamps(fri, "1d", 2, "24/7").dt.day_name()) == ["Saturday", "Sunday"]
    assert bars_per_year("1d", "weekdays") == 252 and bars_per_year("1d") == 365
    rm = RiskManager(Config().risk)
    day = pd.Timedelta(days=1)
    # Friday's bar checked on Monday morning is NOT stale on a weekday calendar...
    assert rm.check_data(fri, pd.Timestamp("2026-10-26 08:00"), day, "weekdays") is None
    # ...but is stale on a 24/7 calendar, and stale on weekdays by Thursday
    assert rm.check_data(fri, pd.Timestamp("2026-10-26 08:00"), day, "24/7")
    assert rm.check_data(fri, pd.Timestamp("2026-10-29 08:00"), day, "weekdays")


def test_index_daily_preset_loads():
    cfg = load_config("deploy/config/config.index-daily.yaml")
    assert cfg.exchange.calendar == "weekdays" and cfg.exchange.timeframe == "1d"
