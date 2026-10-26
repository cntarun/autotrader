"""Persistent bot state (atomic JSON) plus append-only audit logs."""
from __future__ import annotations

import csv
import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class BotState:
    quote: float = 0.0                                        # paper quote balance
    positions: dict[str, float] = field(default_factory=dict)  # base qty per symbol
    entry_price: dict[str, float] = field(default_factory=dict)
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    fees_paid: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    last_bar: dict[str, str] = field(default_factory=dict)
    last_cycle_ts: float = 0.0
    initialised: bool = False


class StateStore:
    def __init__(self, state_dir: str):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "state.json"

    @contextmanager
    def lock(self):
        """Prevents two runs (e.g. a timer and a manual run) trading at the same time."""
        with open(self.dir / ".lock", "w") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another autotrader process holds the state lock") from exc
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def load(self) -> BotState:
        if not self.path.exists():
            return BotState()
        return BotState(**json.loads(self.path.read_text()))

    def save(self, st: BotState) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(st), indent=2, sort_keys=True))
        os.replace(tmp, self.path)

    def append_jsonl(self, name: str, record: dict) -> None:
        with open(self.dir / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def append_fill(self, row: dict) -> None:
        path = self.dir / "fills.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
