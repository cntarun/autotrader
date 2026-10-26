"""Prometheus metrics, a /healthz endpoint, and a textfile dump for one-shot runs."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram,
                               generate_latest, write_to_textfile)

REGISTRY = CollectorRegistry()
EQUITY = Gauge("kronos_equity_quote", "Bot-managed equity in quote currency", registry=REGISTRY)
DRAWDOWN = Gauge("kronos_drawdown_pct", "Drawdown from peak equity, percent", registry=REGISTRY)
DAILY_PNL = Gauge("kronos_daily_pnl_pct", "PnL since UTC day start, percent", registry=REGISTRY)
HALTED = Gauge("kronos_halted", "1 if trading is halted", registry=REGISTRY)
WEIGHT = Gauge("kronos_position_weight", "Position weight of sleeve", ["symbol"], registry=REGISTRY)
MU = Gauge("kronos_forecast_mu", "Mean forecast return over horizon", ["symbol"], registry=REGISTRY)
P_UP = Gauge("kronos_forecast_p_up", "Share of sample paths ending higher", ["symbol"], registry=REGISTRY)
LAST_BAR = Gauge("kronos_data_last_bar_timestamp_seconds", "Last closed bar time", ["symbol"], registry=REGISTRY)
LAST_CYCLE = Gauge("kronos_last_cycle_timestamp_seconds", "Last successful cycle", registry=REGISTRY)
FEES = Gauge("kronos_fees_paid_total", "Cumulative fees paid (quote)", registry=REGISTRY)
ORDERS = Counter("kronos_orders_total", "Orders", ["symbol", "side", "status"], registry=REGISTRY)
ERRORS = Counter("kronos_errors_total", "Errors by stage", ["stage"], registry=REGISTRY)
SLIPPAGE = Histogram("kronos_slippage_bps", "Fill vs reference price, bps", buckets=(1, 5, 10, 25, 50, 100, 250),
                     registry=REGISTRY)
CYCLE_SECONDS = Histogram("kronos_cycle_duration_seconds", "Cycle duration", buckets=(1, 5, 15, 30, 60, 120, 300),
                          registry=REGISTRY)

_health = {"ok": False, "detail": "starting", "max_age": 0}


def set_health(ok: bool, detail: str, max_age_seconds: int) -> None:
    _health.update(ok=ok, detail=detail, max_age=max_age_seconds, ts=time.time())


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/metrics"):
            body, ctype, code = generate_latest(REGISTRY), CONTENT_TYPE_LATEST, 200
        elif self.path.startswith("/healthz"):
            fresh = time.time() - _health.get("ts", 0) < _health["max_age"]
            ok = _health["ok"] and fresh
            body = json.dumps({**_health, "fresh": fresh}).encode()
            ctype, code = "application/json", 200 if ok else 503
        else:
            body, ctype, code = b"not found", "text/plain", 404
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep logs quiet
        pass


def serve(port: int) -> None:
    srv = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def dump_textfile(path: str) -> None:
    write_to_textfile(path, REGISTRY)
