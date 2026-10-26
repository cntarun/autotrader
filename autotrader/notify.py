"""Alerts to Slack, Telegram and/or ntfy.sh (free phone push). All optional, set via env."""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


def notify(text: str, level: str = "info") -> None:
    prefix = {"info": "ℹ️", "trade": "💱", "warn": "⚠️", "critical": "🚨"}.get(level, "")
    msg = f"{prefix} [kronos-{os.environ.get('KRONOS_MODE', 'paper')}] {text}"
    log.info("notify %s: %s", level, text)
    try:
        if url := os.environ.get("SLACK_WEBHOOK_URL"):
            requests.post(url, json={"text": msg}, timeout=10)
        if (tok := os.environ.get("TELEGRAM_BOT_TOKEN")) and (chat := os.environ.get("TELEGRAM_CHAT_ID")):
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": msg}, timeout=10)
        if topic := os.environ.get("NTFY_TOPIC"):
            requests.post(f"https://ntfy.sh/{topic}", data=msg.encode(),
                          headers={"Priority": "urgent" if level == "critical" else "default"}, timeout=10)
    except requests.RequestException as exc:  # alerts must never crash trading
        log.warning("notify failed: %s", exc)
