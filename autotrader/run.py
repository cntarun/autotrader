"""Main entry point.

  python -m autotrader.run --config deploy/config/config.yaml --once      # one cycle (cron/CI/systemd)
  python -m autotrader.run --config ... --loop                             # long-running (Docker), serves :9108
  python -m autotrader.run --config ... --status                           # print state
  python -m autotrader.run --config ... --reset-halt --reason "checked X"  # human clears a halt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback

import pandas as pd

from . import metrics as M
from .broker import LiveBroker, PaperBroker
from .config import Config, load_config
from .data import bars_per_year, fetch_ohlcv, make_exchange, timeframe_to_timedelta
from .forecaster import build_forecaster
from .notify import notify
from .risk import KillSwitch, RiskManager
from .state import BotState, StateStore
from .strategy import decide, realised_vol_annual

log = logging.getLogger("autotrader")


class Bot:
    def __init__(self, cfg: Config, forecaster=None, data_source=None, now_fn=None):
        self.cfg = cfg
        self.store = StateStore(cfg.paths.state_dir)
        self.risk = RiskManager(cfg.risk)
        self.kill = KillSwitch(cfg.paths.kill_switch_file)
        self.bar = timeframe_to_timedelta(cfg.exchange.timeframe)
        self._forecaster = forecaster
        self._data_source = data_source            # callable(symbol, n_bars) -> DataFrame (tests)
        self._public_ex = None
        self.now = now_fn or (lambda: pd.Timestamp.now("UTC").tz_localize(None))

    # -- lazy deps -----------------------------------------------------------------
    @property
    def forecaster(self):
        if self._forecaster is None:
            self._forecaster = build_forecaster("kronos", self.cfg.model, self.cfg.exchange.timeframe,
                                                self.cfg.exchange.calendar)
        return self._forecaster

    def history(self, symbol: str) -> pd.DataFrame:
        n = self.cfg.model.lookback + 5
        if self._data_source:
            return self._data_source(symbol, n)
        if self._public_ex is None:
            self._public_ex = make_exchange(self.cfg.exchange.id)
        since = self.now() - self.bar * (n + 2)
        return fetch_ohlcv(self._public_ex, symbol, self.cfg.exchange.timeframe, since, n)

    def make_broker(self, st: BotState):
        if self.cfg.mode == "live":
            return LiveBroker(self.cfg.exchange.id, self.cfg.costs, self.cfg.risk, self.cfg.exchange.sandbox)
        return PaperBroker(st, self.cfg.costs)

    # -- one decision cycle -----------------------------------------------------------
    def cycle(self) -> dict:
        cfg, st_store = self.cfg, self.store
        t0 = time.time()
        with st_store.lock():
            st = st_store.load()
            if not st.initialised:
                st.quote = cfg.paper_starting_quote if cfg.mode == "paper" else 0.0
                st.initialised = True
            broker = self.make_broker(st)
            symbols = cfg.exchange.symbols
            report: dict = {"ts": str(self.now()), "mode": cfg.mode, "symbols": {}}

            # 1. market data + mark-to-market
            hist, prices, holdings = {}, {}, {}
            quote = 0.0
            for sym in symbols:
                df = self.history(sym)
                hist[sym] = df
                prices[sym] = broker.price(sym, float(df["close"].iloc[-1]))
                base_qty, quote = broker.balances(sym)
                if cfg.mode == "live":
                    self._reconcile(st, sym, base_qty)
                holdings[sym] = base_qty
            equity_all = quote + sum(holdings[s] * prices[s] for s in symbols)
            equity = min(equity_all, cfg.risk.capital_cap) if cfg.mode == "live" else equity_all
            now = self.now()
            self.risk.roll_day(st, now, equity)
            self.risk.check_portfolio(st, equity)

            # 2. kill switch
            # HALT freezes everything (no orders at all). FLATTEN sells everything, then freezes.
            # Automatic circuit breakers behave like FLATTEN: exits are allowed, entries are not.
            ks = self.kill.status()
            if ks:
                self.risk.halt(st, f"kill_switch:{ks}")
            if ks == "HALT":
                st.last_cycle_ts = time.time()
                report.update(equity=round(equity, 4), halted=True, halt_reason=st.halt_reason,
                              note="kill switch HALT: no orders sent")
                st_store.save(st)
                st_store.append_jsonl("decisions.jsonl", report)
                self._export_metrics(st, equity, prices, symbols, equity / len(symbols))
                return report
            flatten = ks == "FLATTEN"

            sleeve = equity / len(symbols)
            for sym in symbols:
                df = hist[sym]
                last_bar = df["timestamps"].iloc[-1]
                M.LAST_BAR.labels(sym).set(last_bar.timestamp())
                cur_w = holdings[sym] * prices[sym] / sleeve if sleeve > 0 else 0.0
                rec = {"last_bar": str(last_bar), "price": prices[sym], "weight_before": round(cur_w, 4)}

                if flatten:
                    target, reason = 0.0, "kill_switch FLATTEN"
                elif st.last_bar.get(sym) == str(last_bar):
                    rec["skipped"] = "bar already processed"
                    report["symbols"][sym] = rec
                    continue
                else:
                    if (why := self.risk.check_data(last_bar, now, self.bar, cfg.exchange.calendar)):
                        M.ERRORS.labels("data").inc()
                        rec["skipped"] = why
                        notify(f"{sym}: {why}", "warn")
                        report["symbols"][sym] = rec
                        continue
                    fc = self.forecaster.forecast(df)
                    if (why := self.risk.check_forecast(fc)):
                        self.risk.halt(st, why)
                    M.MU.labels(sym).set(fc.mu)
                    M.P_UP.labels(sym).set(fc.p_up)
                    rv = realised_vol_annual(df["close"], bars_per_year(cfg.exchange.timeframe, cfg.exchange.calendar))
                    d = decide(fc, cur_w, cfg.strategy, cfg.costs, rv)
                    target, reason = (0.0, f"halted: {st.halt_reason}") if st.halted else (d.weight, d.reason)
                    rec.update(mu=round(fc.mu, 5), p_up=round(fc.p_up, 3), sigma=round(fc.sigma, 5))

                delta = (target - cur_w) * sleeve
                if target == 0.0 and holdings[sym] > 0:
                    delta = -holdings[sym] * prices[sym]           # close the whole position
                allowed, blocked = self.risk.clamp_order(st, delta) if abs(delta) > 1e-9 else (0.0, None)
                rec.update(target=round(target, 4), reason=reason)
                if blocked:
                    rec["blocked"] = blocked
                if allowed:
                    fill = broker.order(sym, allowed, prices[sym])
                    status = "filled" if fill else "unfilled"
                    M.ORDERS.labels(sym, "buy" if allowed > 0 else "sell", status).inc()
                    if fill:
                        self._book_fill(st, fill, prices[sym])
                        rec["fill"] = fill.as_row()
                        notify(f"{fill.side.upper()} {fill.qty:.6f} {sym} @ {fill.price:.2f} "
                               f"(fee {fill.fee:.4f}) - {reason}", "trade")
                    else:
                        notify(f"{sym}: order {allowed:+.2f} not filled", "warn")
                if not flatten:
                    st.last_bar[sym] = str(last_bar)
                report["symbols"][sym] = rec

            # 3. post-trade bookkeeping
            if cfg.mode == "paper":
                equity = st.quote + sum(st.positions.get(s, 0) * prices[s] for s in symbols)
            st.peak_equity = max(st.peak_equity, equity)
            st.last_cycle_ts = time.time()
            report.update(equity=round(equity, 4), halted=st.halted, halt_reason=st.halt_reason)
            st_store.save(st)
            st_store.append_jsonl("decisions.jsonl", report)

        self._export_metrics(st, equity, prices, symbols, sleeve)
        M.CYCLE_SECONDS.observe(time.time() - t0)
        if st.halted:
            notify(f"HALTED: {st.halt_reason}. Manual reset required.", "critical")
        return report

    def _reconcile(self, st: BotState, sym: str, actual_base: float) -> None:
        expected = st.positions.get(sym)
        if expected is None:
            st.positions[sym] = actual_base
            return
        ref = max(expected, actual_base, 1e-12)
        if abs(actual_base - expected) / ref * 100 > self.cfg.risk.reconcile_tolerance_pct and ref > 1e-8:
            self.risk.halt(st, f"reconcile_mismatch {sym} expected={expected} actual={actual_base}")
        st.positions[sym] = actual_base

    def _book_fill(self, st: BotState, fill, ref_price: float) -> None:
        st.trades_today += 1
        st.fees_paid += fill.fee
        slip_bps = abs(fill.price / ref_price - 1) * 10_000
        M.SLIPPAGE.observe(slip_bps)
        if slip_bps > self.cfg.risk.max_slippage_bps:
            self.risk.halt(st, f"slippage {slip_bps:.0f}bps")
        if fill.side == "buy":
            st.entry_price[fill.symbol] = fill.price
        elif fill.symbol in st.entry_price:
            pnl = fill.price / st.entry_price.pop(fill.symbol) - 1 - 2 * self.cfg.costs.fee_rate
            st.consecutive_losses = st.consecutive_losses + 1 if pnl < 0 else 0
        if self.cfg.mode == "live":
            delta = fill.qty if fill.side == "buy" else -fill.qty
            st.positions[fill.symbol] = st.positions.get(fill.symbol, 0.0) + delta
        self.store.append_fill(fill.as_row())

    def _export_metrics(self, st: BotState, equity, prices, symbols, sleeve) -> None:
        M.EQUITY.set(equity)
        M.DRAWDOWN.set(100 * (1 - equity / st.peak_equity) if st.peak_equity else 0)
        M.DAILY_PNL.set(100 * (equity / st.day_start_equity - 1) if st.day_start_equity else 0)
        M.HALTED.set(1 if st.halted else 0)
        M.FEES.set(st.fees_paid)
        M.LAST_CYCLE.set(st.last_cycle_ts)
        for s in symbols:
            M.WEIGHT.labels(s).set(st.positions.get(s, 0) * prices[s] / sleeve if sleeve else 0)
        M.dump_textfile(f"{self.cfg.paths.state_dir}/metrics.prom")


def seconds_until_next_bar(now: pd.Timestamp, bar: pd.Timedelta, delay: int) -> float:
    next_close = now.floor(bar) + bar
    return max(5.0, (next_close - now).total_seconds() + delay)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--reset-halt", action="store_true")
    p.add_argument("--reason", default="")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(a.config)
    bot = Bot(cfg)

    if a.status:
        print(json.dumps(bot.store.load().__dict__, indent=2, default=str))
        return 0
    if a.reset_halt:
        if not a.reason:
            p.error("--reset-halt needs --reason")
        with bot.store.lock():
            st = bot.store.load()
            st.halted, old = False, st.halt_reason
            st.halt_reason, st.consecutive_losses = "", 0
            bot.store.save(st)
            bot.store.append_jsonl("audit.jsonl", {"ts": time.time(), "action": "reset_halt",
                                                    "previous": old, "reason": a.reason})
        notify(f"halt cleared (was: {old}) - {a.reason}", "warn")
        return 0

    interval = int(bot.bar.total_seconds())
    if a.once:
        try:
            print(json.dumps(bot.cycle(), indent=2, default=str))
            return 0
        except Exception as exc:  # noqa: BLE001
            M.ERRORS.labels("cycle").inc()
            notify(f"cycle failed: {exc!r}", "critical")
            traceback.print_exc()
            return 1

    M.serve(cfg.runtime.health_port)
    M.set_health(False, "waiting for first cycle", 2 * interval + 600)
    while True:
        try:
            rep = bot.cycle()
            M.set_health(True, f"equity={rep['equity']} halted={rep['halted']}", 2 * interval + 600)
        except Exception as exc:  # noqa: BLE001
            M.ERRORS.labels("cycle").inc()
            M.set_health(False, repr(exc), 2 * interval + 600)
            notify(f"cycle failed: {exc!r}", "critical")
            traceback.print_exc()
        time.sleep(seconds_until_next_bar(bot.now(), bot.bar, cfg.runtime.loop_delay_seconds))


if __name__ == "__main__":
    sys.exit(main())
