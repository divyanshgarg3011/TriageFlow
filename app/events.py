"""In-process event bus. Every agent step becomes an event; the UI streams them over SSE.

The same events double as the audit trail / trace for each run.
"""
from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any

_seq = itertools.count(1)


class EventBus:
    def __init__(self, history_limit: int = 5000) -> None:
        self.history: list[dict] = []
        self.history_limit = history_limit
        self._subscribers: set[asyncio.Queue] = set()

    def emit(self, run_id: str | None, kind: str, agent: str | None = None, **data: Any) -> dict:
        event = {"seq": next(_seq), "ts": time.time(), "run_id": run_id, "type": kind, "agent": agent, "data": data}
        self.history.append(event)
        if len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]
        for q in list(self._subscribers):
            q.put_nowait(event)
        return event

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def for_run(self, run_id: str) -> list[dict]:
        return [e for e in self.history if e["run_id"] == run_id]


bus = EventBus()
