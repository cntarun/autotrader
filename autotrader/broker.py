"""Brokers. PaperBroker simulates fills; LiveBroker sends marketable-limit IOC orders via ccxt."""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

from .config import CostsCfg, RiskCfg
from .state import BotState


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    notional: float
    ts: float
    order_id: str
    mode: str

    def as_row(self) -> dict:
        return asdict(self)


class PaperBroker:
    mode = "paper"

    def __init__(self, st: BotState, costs: CostsCfg):
        self.st, self.costs = st, costs

    def balances(self, symbol: str) -> tuple[float, float]:
        return self.st.positions.get(symbol, 0.0), self.st.quote

    def price(self, symbol: str, last_close: float) -> float:
        return last_close

    def order(self, symbol: str, notional: float, ref_price: float) -> Fill | None:
        slip = self.costs.slippage_bps / 10_000
        fee_rate = self.costs.fee_rate
        if notional > 0:
            px = ref_price * (1 + slip)
            notional = min(notional, self.st.quote / (1 + fee_rate))
            qty = notional / px
            fee = notional * fee_rate
            self.st.quote -= notional + fee
            self.st.positions[symbol] = self.st.positions.get(symbol, 0.0) + qty
            side = "buy"
        else:
            px = ref_price * (1 - slip)
            qty = min(-notional / px, self.st.positions.get(symbol, 0.0))
            notional = qty * px
            fee = notional * fee_rate
            self.st.quote += notional - fee
            self.st.positions[symbol] = self.st.positions.get(symbol, 0.0) - qty
            side = "sell"
        if qty <= 0:
            return None
        return Fill(symbol, side, qty, px, fee, notional, time.time(), f"paper-{int(time.time()*1e3)}", "paper")


class LiveBroker:
    """Real orders. Keys come only from env vars; use a trade-only, no-withdrawal,
    IP-whitelisted key on a dedicated sub-account."""
    mode = "live"

    def __init__(self, exchange_id: str, costs: CostsCfg, risk: RiskCfg, sandbox: bool = False):
        from .data import make_exchange

        key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
        if not key or not secret:
            raise PermissionError("EXCHANGE_API_KEY / EXCHANGE_API_SECRET not set")
        self.ex = make_exchange(exchange_id, key, secret, os.environ.get("EXCHANGE_API_PASSWORD"), sandbox)
        self.ex.load_markets()
        self.costs, self.risk = costs, risk

    def balances(self, symbol: str) -> tuple[float, float]:
        base, quote = symbol.split("/")
        bal = self.ex.fetch_balance()
        return float(bal.get(base, {}).get("free") or 0), float(bal.get(quote, {}).get("free") or 0)

    def price(self, symbol: str, last_close: float) -> float:
        t = self.ex.fetch_ticker(symbol)
        return float(t.get("last") or last_close)

    def order(self, symbol: str, notional: float, ref_price: float) -> Fill | None:
        max_slip = self.risk.max_slippage_bps / 10_000
        side = "buy" if notional > 0 else "sell"
        limit_px = ref_price * (1 + max_slip) if side == "buy" else ref_price * (1 - max_slip)
        qty = float(self.ex.amount_to_precision(symbol, abs(notional) / ref_price))
        px = float(self.ex.price_to_precision(symbol, limit_px))
        if qty <= 0:
            return None
        o = self.ex.create_order(symbol, "limit", side, qty, px, {"timeInForce": "IOC"})
        for _ in range(10):  # wait for the IOC to settle
            o = self.ex.fetch_order(o["id"], symbol)
            if o.get("status") in ("closed", "canceled", "expired", "rejected"):
                break
            time.sleep(1)
        filled = float(o.get("filled") or 0)
        if filled <= 0:
            return None
        avg = float(o.get("average") or o.get("price") or px)
        fee = float((o.get("fee") or {}).get("cost") or filled * avg * self.costs.fee_rate)
        return Fill(symbol, side, filled, avg, fee, filled * avg, time.time(), str(o["id"]), "live")
