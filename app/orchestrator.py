"""Orchestrator: a deterministic workflow around the LLM agents.

    Triage -> Research -> Resolution <-> QA (revise loop, max N) -> Policy gate -> auto-resolve | human review

The routing and the final go/no-go are plain Python on purpose: the LLMs do judgement work
(classify, research, write, critique) and code enforces the rules that must never be wrong.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import config, db
from .agent_loop import AgentError, Metrics, RunContext, run_agent
from .agents import AGENTS
from .events import bus

MONEY_ACTIONS = {"refund", "store_credit"}
REMEDY_ACTIONS = {"refund", "store_credit", "reship", "replacement"}


@dataclass
class Run:
    id: str
    ticket: dict
    status: str = "queued"   # queued | running | resolved | needs_human | approved | rejected | failed
    stage: str = "queued"
    triage: dict | None = None
    research: dict | None = None
    resolution: dict | None = None
    review: dict | None = None
    revisions: int = 0
    actions: list[dict] = field(default_factory=list)
    gate_reasons: list[str] = field(default_factory=list)
    final_reply: str | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    ctx: RunContext | None = None

    def summary(self) -> dict:
        return {
            "id": self.id, "ticket_id": self.ticket["id"], "status": self.status, "stage": self.stage,
            "category": (self.triage or {}).get("category"), "priority": (self.triage or {}).get("priority"),
            "revisions": self.revisions, "gate_reasons": self.gate_reasons, "actions": self.actions,
            "reply": self.final_reply or (self.resolution or {}).get("reply"),
            "triage": self.triage, "research": self.research, "resolution": self.resolution, "review": self.review,
            "error": self.error, "ticket": self.ticket,
            "duration_s": round((self.finished_at or time.time()) - self.started_at, 2) if self.started_at else None,
            "metrics": self.ctx.metrics.as_dict() if self.ctx else Metrics().as_dict(),
        }


RUNS: dict[str, Run] = {}
_sem: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(config.MAX_PARALLEL_RUNS)
    return _sem


def _tag(name: str, value: Any) -> str:
    return f"<{name}>\n{json.dumps(value, indent=2, default=str)}\n</{name}>"


def _set_stage(run: Run, stage: str, **extra: Any) -> None:
    run.stage = stage
    bus.emit(run.id, "stage", None, stage=stage, ticket_id=run.ticket["id"], **extra)


def _route(run: Run, frm: str, to: str, reason: str) -> None:
    bus.emit(run.id, "route", None, frm=frm, to=to, reason=reason)


# ---------------------------------------------------------------- policy gate (pure code)


def policy_gate(run: Run) -> list[str]:
    """Return reasons a human must review. Empty list = safe to auto-resolve."""
    reasons: list[str] = []
    t, r, q = run.triage or {}, run.resolution or {}, run.review or {}

    if t.get("needs_human_now"):
        reasons.append(f"Triage flagged for a human: {t.get('human_reason') or t.get('category')}")
    if t.get("category") == "safety":
        reasons.append(f"Category '{t['category']}' is always handled by a specialist")
    if float(t.get("confidence", 1)) < config.MIN_CONFIDENCE:
        reasons.append(f"Triage confidence {t.get('confidence')} is below {config.MIN_CONFIDENCE}")
    if q.get("verdict") != "approve":
        reasons.append(f"QA verdict was '{q.get('verdict')}'" + (f" after {run.revisions} revision(s)" if run.revisions else ""))
    if float(q.get("risk_score", 0)) > config.MAX_RISK:
        reasons.append(f"QA risk score {q.get('risk_score')} is above {config.MAX_RISK}")
    if float(r.get("confidence", 1)) < config.MIN_CONFIDENCE:
        reasons.append(f"Resolution confidence {r.get('confidence')} is below {config.MIN_CONFIDENCE}")

    total_money = sum(float(a.get("amount") or 0) for a in run.actions if a.get("type") in MONEY_ACTIONS)
    if total_money > config.REFUND_AUTO_LIMIT:
        reasons.append(f"Total refund/credit ${total_money:,.2f} is above the ${config.REFUND_AUTO_LIMIT:,.0f} auto-approval limit")

    elig = run.ctx.eligibility if run.ctx else {}
    for a in run.actions:
        kind, oid = a.get("type"), a.get("order_id")
        if kind == "lock_account":
            reasons.append("Account lock requires the security team")
        if kind in REMEDY_ACTIONS:
            checks = elig.get(oid or "", [])
            ok = [c for c in checks if c.get("eligible")]
            if kind == "store_credit":
                allowed = max([float(c.get("store_credit_allowed") or 0) for c in checks] or [0])
                if float(a.get("amount") or 0) > allowed:
                    reasons.append(f"Store credit ${a.get('amount')} on {oid} exceeds the ${allowed} policy allows")
            elif not ok:
                reasons.append(f"{kind} on {oid or 'unknown order'} was not confirmed eligible by the policy engine")
            elif kind == "refund":
                cap = max(float(c.get("max_refund") or 0) for c in ok)
                if float(a.get("amount") or 0) > cap + 0.01:
                    reasons.append(f"Refund ${a.get('amount')} on {oid} exceeds the policy maximum ${cap:,.2f}")
    # de-duplicate, keep order
    return list(dict.fromkeys(reasons))


# ---------------------------------------------------------------- pipeline


async def process(run: Run) -> Run:
    async with _semaphore():
        run.status, run.started_at = "running", time.time()
        run.ctx = RunContext(run_id=run.id)
        bus.emit(run.id, "run_started", None, ticket=run.ticket, demo=config.DEMO_MODE)
        try:
            await _pipeline(run)
        except AgentError as exc:
            run.status, run.error = "needs_human", str(exc)
            run.gate_reasons = [f"Agent failure: {exc}"]
            _set_stage(run, "human")
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected in the UI
            run.status, run.error = "failed", f"{type(exc).__name__}: {exc}"
            bus.emit(run.id, "error", None, message=run.error)
        run.finished_at = time.time()
        bus.emit(run.id, "run_finished", None, summary=run.summary())
        return run


async def _pipeline(run: Run) -> None:
    ticket = run.ticket
    ctx = run.ctx
    assert ctx is not None

    # 1. Triage
    _set_stage(run, "triage")
    tri = await run_agent(AGENTS["triage"], ctx, _tag("ticket", ticket))
    run.triage = tri.output
    _route(run, "triage", "research",
           "Fast lane: a human will take over, research still gathers context" if run.triage.get("needs_human_now")
           else f"Classified as {run.triage.get('category')} / {run.triage.get('priority')}")

    # 2. Research
    _set_stage(run, "research")
    res = await run_agent(AGENTS["research"], ctx, _tag("ticket", ticket) + "\n" + _tag("triage", run.triage))
    run.research = res.output
    _route(run, "research", "resolution", f"{len(run.research.get('key_facts', []))} facts, {len(run.research.get('policies', []))} policies")

    # 3 + 4. Resolution <-> QA loop
    feedback: dict | None = None
    while True:
        _set_stage(run, "resolution", attempt=run.revisions + 1)
        prompt = "\n".join([_tag("ticket", ticket), _tag("triage", run.triage), _tag("research", run.research)])
        if feedback:
            prompt += "\n" + _tag("previous_reply", run.resolution) + "\n" + _tag("qa_feedback", feedback)
        sol = await run_agent(AGENTS["resolution"], ctx, prompt, attempt=run.revisions + 1)
        run.resolution, run.actions = sol.output, sol.proposed_actions
        _route(run, "resolution", "qa", f"Draft ready with {len(run.actions)} proposed action(s)")

        _set_stage(run, "qa", attempt=run.revisions + 1)
        eligibility = {k: v for k, v in ctx.eligibility.items()}
        review_prompt = "\n".join([
            _tag("ticket", ticket), _tag("triage", run.triage), _tag("research", run.research),
            _tag("eligibility_results", eligibility), _tag("draft_reply", run.resolution.get("reply")),
            _tag("proposed_actions", run.actions),
        ])
        qa = await run_agent(AGENTS["qa"], ctx, review_prompt, attempt=run.revisions + 1)
        run.review = qa.output

        if run.review.get("verdict") == "revise" and run.revisions < config.MAX_REVISIONS:
            run.revisions += 1
            feedback = {"issues": run.review.get("issues", []), "feedback": run.review.get("feedback", "")}
            _route(run, "qa", "resolution", f"Revision {run.revisions}: " + (run.review.get("feedback") or "see issues"))
            continue
        _route(run, "qa", "gate", f"QA verdict: {run.review.get('verdict')} (risk {run.review.get('risk_score')})")
        break

    # 5. Policy gate (deterministic)
    _set_stage(run, "gate")
    run.gate_reasons = policy_gate(run)
    bus.emit(run.id, "gate", None, passed=not run.gate_reasons, reasons=run.gate_reasons,
             checks={"auto_refund_limit": config.REFUND_AUTO_LIMIT, "min_confidence": config.MIN_CONFIDENCE,
                     "max_risk": config.MAX_RISK})

    if run.gate_reasons:
        run.status = "needs_human"
        _route(run, "gate", "human", run.gate_reasons[0])
        _set_stage(run, "human")
        bus.emit(run.id, "awaiting_human", None, reasons=run.gate_reasons)
    else:
        _route(run, "gate", "auto", "All checks passed")
        _execute(run, approved_by="policy-gate", reply=run.resolution.get("reply", ""))
        run.status = "resolved"
        _set_stage(run, "done")


def _execute(run: Run, approved_by: str, reply: str) -> None:
    for action in run.actions:
        db.record_action(run.ticket["id"], run.id, action, approved_by)
        bus.emit(run.id, "action_executed", None, action=action, approved_by=approved_by)
    run.final_reply = reply
    bus.emit(run.id, "reply_sent", None, to=run.ticket.get("customer_email"), approved_by=approved_by, reply=reply)


def human_decision(run_id: str, approve: bool, reply: str | None = None, note: str = "") -> Run:
    run = RUNS[run_id]
    if run.status != "needs_human":
        raise ValueError(f"Run is '{run.status}', not waiting for a human")
    bus.emit(run.id, "human_decision", "human", approve=approve, note=note, edited=reply is not None)
    if approve:
        _execute(run, approved_by="human", reply=reply if reply is not None else (run.resolution or {}).get("reply", ""))
        run.status = "approved"
    else:
        run.status = "rejected"
    _set_stage(run, "done")
    bus.emit(run.id, "run_finished", None, summary=run.summary())
    return run


def new_run(ticket: dict) -> Run:
    run = Run(id="run_" + uuid.uuid4().hex[:8], ticket=ticket)
    RUNS[run.id] = run
    bus.emit(run.id, "run_queued", None, ticket_id=ticket["id"])
    return run
