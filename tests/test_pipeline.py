import asyncio
import json

import pytest

from app import config, db
from app import orchestrator as orch
from app.agent_loop import RunContext
from app.events import bus

TICKETS = {t["id"]: t for t in json.loads(config.TICKETS_PATH.read_text())}


def run(ticket_id: str) -> orch.Run:
    r = orch.new_run(TICKETS[ticket_id])
    asyncio.run(orch.process(r))
    return r


@pytest.mark.parametrize("tid,status", [
    ("T-1001", "resolved"), ("T-1002", "resolved"), ("T-1003", "needs_human"), ("T-1004", "needs_human"),
    ("T-1005", "resolved"), ("T-1006", "needs_human"), ("T-1007", "resolved"), ("T-1008", "resolved"),
    ("T-1009", "resolved"),
])
def test_sample_tickets_route_correctly(tid, status):
    assert run(tid).status == status


def test_revise_loop_runs_and_is_traced():
    r = run("T-1001")
    assert r.revisions == 1
    kinds = [e["type"] for e in bus.for_run(r.id)]
    assert kinds.count("agent_started") == 6  # triage, research, (resolution, qa) x2
    assert "gate" in kinds and "reply_sent" in kinds


def test_auto_resolve_executes_actions_in_db():
    run("T-1008")
    with db.session() as conn:
        refunds = conn.execute("SELECT amount FROM refunds WHERE order_id='NG-5560'").fetchall()
    assert [r[0] for r in refunds] == [149.0]


def test_escalated_ticket_changes_nothing_until_human_approves():
    r = run("T-1003")
    with db.session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 0
    orch.human_decision(r.id, approve=True, reply="Edited reply")
    assert r.status == "approved" and r.final_reply == "Edited reply"
    with db.session() as conn:
        assert conn.execute("SELECT status FROM orders WHERE id='NG-5544'").fetchone()[0] == "refunded"


def _gate_run(**kw) -> orch.Run:
    r = orch.Run(id="t", ticket={"id": "X"})
    r.ctx = RunContext("t")
    r.triage = {"category": "billing", "confidence": 0.9, "needs_human_now": False}
    r.resolution = {"confidence": 0.9}
    r.review = {"verdict": "approve", "risk_score": 0.1}
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_gate_blocks_refund_not_confirmed_by_policy_engine():
    r = _gate_run(actions=[{"type": "refund", "order_id": "NG-5530", "amount": 89}])
    assert any("not confirmed eligible" in x for x in orch.policy_gate(r))


def test_gate_blocks_refund_above_policy_max_even_if_qa_approved():
    r = _gate_run(actions=[{"type": "refund", "order_id": "NG-5530", "amount": 120}])
    r.ctx.eligibility["NG-5530"] = [{"eligible": True, "max_refund": 89.0}]
    assert any("exceeds the policy maximum" in x for x in orch.policy_gate(r))


def test_gate_passes_clean_case():
    r = _gate_run(actions=[{"type": "refund", "order_id": "NG-5530", "amount": 89}])
    r.ctx.eligibility["NG-5530"] = [{"eligible": True, "max_refund": 89.0}]
    assert orch.policy_gate(r) == []
