import time

from fastapi.testclient import TestClient

from app.server import app


def wait_done(client, run_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/api/runs/{run_id}").json()["summary"]
        if s["status"] not in ("queued", "running"):
            return s
        time.sleep(0.05)
    raise TimeoutError


def test_create_ticket_and_watch_it_finish():
    with TestClient(app) as client:
        assert client.get("/api/config").json()["demo"] is True
        r = client.post("/api/tickets", json={"customer_email": "priya.nair@example.com",
                                              "subject": "Watch won't pair", "body": "My Pulse watch won't pair with my phone"})
        run_id = r.json()["run_id"]
        s = wait_done(client, run_id)
        assert s["status"] == "resolved" and s["triage"]["category"] == "technical"
        events = client.get(f"/api/runs/{run_id}").json()["events"]
        assert {"agent_started", "tool_call", "tool_result", "gate", "reply_sent"} <= {e["type"] for e in events}


def test_human_approval_flow():
    with TestClient(app) as client:
        run_id = client.post("/api/tickets/T-1006/run").json()["run_id"]
        assert wait_done(client, run_id)["status"] == "needs_human"
        s = client.post(f"/api/runs/{run_id}/decision", json={"approve": True}).json()
        assert s["status"] == "approved"
        # a second decision is rejected
        assert client.post(f"/api/runs/{run_id}/decision", json={"approve": False}).status_code == 409
