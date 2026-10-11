"""FastAPI app: REST endpoints + a Server-Sent Events stream of every agent step."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config, db
from . import orchestrator as orch
from .agents import AGENTS
from .events import bus

STATIC = Path(__file__).parent / "static"
TICKETS: dict[str, dict] = {}
LATEST_RUN: dict[str, str] = {}      # ticket id -> latest run id
_tasks: set[asyncio.Task] = set()
_counter = {"n": 2000}


def _load_samples() -> None:
    TICKETS.clear()
    LATEST_RUN.clear()
    for t in json.loads(config.TICKETS_PATH.read_text()):
        TICKETS[t["id"]] = t


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db(reset=True)
    _load_samples()
    yield


app = FastAPI(title="TriageFlow", lifespan=lifespan)


def _start(ticket: dict) -> orch.Run:
    run = orch.new_run(ticket)
    LATEST_RUN[ticket["id"]] = run.id
    task = asyncio.create_task(orch.process(run))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return run


def _ticket_view(t: dict) -> dict:
    rid = LATEST_RUN.get(t["id"])
    run = orch.RUNS.get(rid) if rid else None
    return {**t, "run_id": rid, "status": run.status if run else "new", "stage": run.stage if run else None,
            "category": (run.triage or {}).get("category") if run else None,
            "priority": (run.triage or {}).get("priority") if run else None}


# ---------------------------------------------------------------- pages


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


# ---------------------------------------------------------------- API


@app.get("/api/config")
async def get_config():
    return {
        "demo": config.DEMO_MODE,
        "agents": [{"key": a.key, "label": a.label, "role": a.role, "model": a.model,
                    "tools": a.tools} for a in AGENTS.values()],
        "limits": {"refund_auto_limit": config.REFUND_AUTO_LIMIT, "min_confidence": config.MIN_CONFIDENCE,
                   "max_risk": config.MAX_RISK, "max_revisions": config.MAX_REVISIONS},
    }


@app.get("/api/tickets")
async def list_tickets():
    return [_ticket_view(t) for t in TICKETS.values()]


@app.get("/api/customers")
async def customers():
    return db.list_customers()


class NewTicket(BaseModel):
    customer_email: str = Field(min_length=3)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)


@app.post("/api/tickets")
async def create_ticket(t: NewTicket):
    _counter["n"] += 1
    ticket = {"id": f"T-{_counter['n']}", **t.model_dump()}
    TICKETS[ticket["id"]] = ticket
    run = _start(ticket)
    return {"ticket": ticket, "run_id": run.id}


@app.post("/api/tickets/{ticket_id}/run")
async def run_ticket(ticket_id: str):
    ticket = TICKETS.get(ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    current = orch.RUNS.get(LATEST_RUN.get(ticket_id, ""))
    if current and current.status in ("queued", "running"):
        raise HTTPException(409, "This ticket is already being processed")
    return {"run_id": _start(ticket).id}


@app.post("/api/run-all")
async def run_all():
    started = []
    for t in TICKETS.values():
        if t["id"] not in LATEST_RUN:
            started.append(_start(t).id)
    return {"started": started}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run = orch.RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return {"summary": run.summary(), "events": bus.for_run(run_id)}


class Decision(BaseModel):
    approve: bool
    reply: str | None = None
    note: str = ""


@app.post("/api/runs/{run_id}/decision")
async def decide(run_id: str, d: Decision):
    if run_id not in orch.RUNS:
        raise HTTPException(404, "Run not found")
    try:
        run = orch.human_decision(run_id, d.approve, d.reply, d.note)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.summary()


@app.get("/api/stats")
async def stats():
    runs = [r for r in orch.RUNS.values() if r.status not in ("queued", "running")]
    done = len(runs)
    auto = sum(r.status == "resolved" for r in runs)
    cost = sum(r.ctx.metrics.cost_usd for r in runs if r.ctx)
    dur = [r.finished_at - r.started_at for r in runs if r.finished_at]
    return {"processed": done, "auto_resolved": auto,
            "escalated": sum(r.status in ("needs_human", "approved", "rejected") for r in runs),
            "auto_rate": round(auto / done, 3) if done else None,
            "avg_seconds": round(sum(dur) / len(dur), 2) if dur else None,
            "total_cost_usd": round(cost, 5)}


@app.post("/api/reset")
async def reset():
    if any(r.status in ("queued", "running") for r in orch.RUNS.values()):
        raise HTTPException(409, "Wait for running tickets to finish before resetting")
    orch.RUNS.clear()
    bus.history.clear()
    db.init_db(reset=True)
    _load_samples()
    bus.emit(None, "reset", None)
    return {"ok": True}


@app.get("/api/stream")
async def stream():
    """SSE: every event from every run, live."""
    queue = bus.subscribe()

    async def gen():
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
