"""Run the labelled ticket set through the full pipeline and score it.

    python -m evals.run_evals            # uses Claude if ANTHROPIC_API_KEY is set, else the demo model
    python -m evals.run_evals --demo     # force the offline keyword baseline
    python -m evals.run_evals --limit 5

What is scored (all from the pipeline's structured outputs, no LLM judge):
  - category accuracy            triage label vs expected
  - routing accuracy             auto-resolved vs sent to a human
  - escalation recall            of tickets that MUST reach a human, how many did  <- the safety metric
  - escalation precision         of tickets sent to a human, how many needed it   <- the cost metric
  - action match                 proposed action types == expected action types
Plus cost, tokens, latency and QA revisions per ticket.

Each ticket runs on a freshly seeded database so actions from one ticket don't leak into the next.
Results go to evals/results/<timestamp>.json and evals/results/<timestamp>.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use the offline scripted model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="ticket ids to run")
    args = ap.parse_args()

    # configure before importing the app
    if args.demo:
        os.environ["DEMO_MODE"] = "1"
    os.environ.setdefault("DEMO_STEP_DELAY", "0")
    os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "eval.db")
    sys.path.insert(0, str(HERE.parent))

    from app import config, db  # noqa: E402
    from app import orchestrator as orch  # noqa: E402

    cases = [json.loads(line) for line in (HERE / "dataset.jsonl").read_text().splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    if args.limit:
        cases = cases[: args.limit]

    mode = "demo (keyword baseline)" if config.DEMO_MODE else "Claude: " + ", ".join(f"{k}={v}" for k, v in config.MODELS.items())
    print(f"Running {len(cases)} tickets · {mode}\n")

    async def run_all():
        rows = []
        for c in cases:
            db.init_db(reset=True)
            ticket = {k: c[k] for k in ("id", "customer_email", "subject", "body")}
            run = orch.new_run(ticket)
            t0 = time.time()
            await orch.process(run)
            s = run.summary()
            exp = c["expected"]
            route = "auto" if s["status"] == "resolved" else "human" if s["status"] == "needs_human" else s["status"]
            got_actions = sorted(a.get("type") for a in s["actions"])
            row = {
                "id": c["id"], "subject": c["subject"],
                "expected_category": exp["category"], "category": s["category"],
                "expected_route": exp["route"], "route": route,
                "expected_actions": sorted(exp["actions"]), "actions": got_actions,
                "category_ok": s["category"] == exp["category"],
                "route_ok": route == exp["route"],
                "actions_ok": got_actions == sorted(exp["actions"]),
                "revisions": s["revisions"], "gate_reasons": s["gate_reasons"], "error": s["error"],
                "seconds": round(time.time() - t0, 2), **s["metrics"],
                "reply": s["reply"],
            }
            rows.append(row)
            flag = "✓" if row["category_ok"] and row["route_ok"] and row["actions_ok"] else "✗"
            print(f"{flag} {c['id']:7} cat {row['category']!s:16} route {route:6} actions {got_actions}  "
                  f"(${row['cost_usd']:.4f}, {row['seconds']}s)")
        return rows

    rows = asyncio.run(run_all())
    n = len(rows)
    must_human = [r for r in rows if r["expected_route"] == "human"]
    sent_human = [r for r in rows if r["route"] == "human"]
    summary = {
        "mode": mode, "tickets": n,
        "category_accuracy": sum(r["category_ok"] for r in rows) / n,
        "routing_accuracy": sum(r["route_ok"] for r in rows) / n,
        "escalation_recall": (sum(r["route"] == "human" for r in must_human) / len(must_human)) if must_human else None,
        "escalation_precision": (sum(r["expected_route"] == "human" for r in sent_human) / len(sent_human)) if sent_human else None,
        "action_match": sum(r["actions_ok"] for r in rows) / n,
        "fully_correct": sum(r["category_ok"] and r["route_ok"] and r["actions_ok"] for r in rows) / n,
        "avg_cost_usd": sum(r["cost_usd"] for r in rows) / n,
        "avg_seconds": sum(r["seconds"] for r in rows) / n,
        "avg_llm_calls": sum(r["llm_calls"] for r in rows) / n,
        "total_revisions": sum(r["revisions"] for r in rows),
        "errors": sum(bool(r["error"]) for r in rows),
    }

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("-demo" if config.DEMO_MODE else "-claude")
    (out / f"{stamp}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    (out / f"{stamp}.md").write_text(_markdown(summary, rows))

    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    print(f"\nCategory accuracy    {pct(summary['category_accuracy'])}")
    print(f"Routing accuracy     {pct(summary['routing_accuracy'])}")
    print(f"Escalation recall    {pct(summary['escalation_recall'])}   (must-escalate tickets that reached a human)")
    print(f"Escalation precision {pct(summary['escalation_precision'])}")
    print(f"Action match         {pct(summary['action_match'])}")
    print(f"Fully correct        {pct(summary['fully_correct'])}")
    print(f"Avg cost / ticket    ${summary['avg_cost_usd']:.4f}   avg {summary['avg_seconds']:.1f}s, {summary['avg_llm_calls']:.1f} model calls")
    print(f"\nReport: evals/results/{stamp}.md")


def _markdown(s: dict, rows: list[dict]) -> str:
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    lines = [
        "# Eval report", "", f"Mode: {s['mode']}  ", f"Tickets: {s['tickets']}", "",
        "| Metric | Score |", "|---|---|",
        f"| Category accuracy | {pct(s['category_accuracy'])} |",
        f"| Routing accuracy | {pct(s['routing_accuracy'])} |",
        f"| Escalation recall (safety) | {pct(s['escalation_recall'])} |",
        f"| Escalation precision | {pct(s['escalation_precision'])} |",
        f"| Action match | {pct(s['action_match'])} |",
        f"| Fully correct | {pct(s['fully_correct'])} |",
        f"| Avg cost per ticket | ${s['avg_cost_usd']:.4f} |",
        f"| Avg seconds per ticket | {s['avg_seconds']:.1f} |",
        f"| QA revisions (total) | {s['total_revisions']} |",
        "", "| Ticket | Category (expected) | Route (expected) | Actions (expected) | OK |", "|---|---|---|---|---|",
    ]
    for r in rows:
        ok = "✓" if r["category_ok"] and r["route_ok"] and r["actions_ok"] else "✗"
        lines.append(f"| {r['id']} {r['subject']} | {r['category']} ({r['expected_category']}) | {r['route']} ({r['expected_route']}) "
                     f"| {', '.join(r['actions']) or '–'} ({', '.join(r['expected_actions']) or '–'}) | {ok} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
