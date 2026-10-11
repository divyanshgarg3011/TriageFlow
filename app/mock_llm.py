"""Offline 'demo mode' LLM.

It returns the same Anthropic-style blocks (text + tool_use) as the real API, driven by simple
rules, so the full agent loop, tools, gates and UI run without an API key or network. It is a
stand-in for demos and tests, not a model: real behaviour comes from running with Claude.

The QA mock deliberately rejects the first shipping draft (missing timeline) so the
revise loop is visible in demos.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import uuid

from . import config
from .llm import LLMResponse

ORDER_RE = re.compile(r"\bNG-\d{4}\b", re.I)
DEFECT_WORDS = ("flicker", "defect", "broken", "damaged", "not working", "stopped working", "faulty")


# ---------------------------------------------------------------- helpers to read the conversation


def _tags(text: str) -> dict:
    out = {}
    for name, body in re.findall(r"<(\w+)>\n(.*?)\n</\1>", text, re.S):
        try:
            out[name] = json.loads(body)
        except json.JSONDecodeError:
            out[name] = body
    return out


def _history(messages: list[dict]) -> list[tuple[str, dict, object]]:
    """[(tool_name, tool_input, parsed_result), ...] in call order."""
    calls: dict[str, tuple[str, dict]] = {}
    out = []
    for m in messages:
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                calls[b["id"]] = (b["name"], b.get("input") or {})
            elif b.get("type") == "tool_result" and b["tool_use_id"] in calls:
                name, args = calls[b["tool_use_id"]]
                try:
                    res = json.loads(b["content"])
                except (TypeError, json.JSONDecodeError):
                    res = b["content"]
                out.append((name, args, res))
    return out


def _first(hist, name):
    for n, a, r in hist:
        if n == name:
            return r
    return None


def _tool(name: str, **args) -> dict:
    return {"type": "tool_use", "id": "toolu_mock_" + uuid.uuid4().hex[:10], "name": name, "input": args}


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


# ---------------------------------------------------------------- per-agent scripted policies


def _triage(ctx: dict, hist) -> list[dict]:
    t = ctx["ticket"]
    text = f"{t['subject']} {t['body']}".lower()
    m = ORDER_RE.search(t["body"] + " " + t["subject"])
    order_id = m.group(0).upper() if m else ""
    legal = any(w in text for w in ("lawyer", " sue", "chargeback", "legal"))
    human_reason = ""
    conf = 0.92
    if any(w in text for w in ("hot", "burn", "fire", "smoke", "swell", "injur")):
        cat, pri = "safety", "urgent"
        human_reason = "Safety incident (overheating / property damage)" + (" plus a legal threat" if legal else "")
    elif any(w in text for w in ("password", "log in", "login", "hacked", "in my account", "did not place")):
        cat, pri = "account_security", "urgent"
        human_reason = "Suspected account takeover"
    elif any(w in text for w in ("charged twice", "double charge", "charged two", "two payments")):
        cat, pri = "billing", "high"
    elif any(w in text for w in ("refund", "return")):
        cat, pri = "refund_return", "high" if any(w in text for w in DEFECT_WORDS) else "medium"
    elif any(w in text for w in ("arrive", "tracking", "where is", "nothing came", "never came", "delivery")):
        cat, pri = "shipping", "medium"
        if not order_id:
            conf = 0.78
    elif any(w in text for w in ("pair", "won't", "not working", "how do", "setup", "reset")):
        cat, pri = "technical", "low"
    else:
        cat, pri, conf = "other", "medium", 0.55
    if legal and not human_reason:
        human_reason = "Legal threat"
    sentiment = ("angry" if legal or "!!" in t["body"] or "dangerous" in text
                 else "frustrated" if any(w in text for w in ("not acceptable", "fix this", "what is going on", "never"))
                 else "neutral")
    out = {
        "category": cat, "priority": pri, "sentiment": sentiment, "order_id": order_id,
        "summary": _summary(cat, t, order_id), "needs_human_now": bool(human_reason),
        "human_reason": human_reason, "confidence": conf,
    }
    note = f"Signals: category={cat}, sentiment={sentiment}" + (f", order {order_id}" if order_id else ", no order id given")
    if human_reason:
        note += f". Flagging for a human: {human_reason.lower()}."
    return [_text(note), _tool("submit_triage", **out)]


def _summary(cat, t, oid):
    return {
        "shipping": f"Customer's parcel {oid or '(order unknown)'} has not arrived and they want it found or replaced.",
        "refund_return": f"Customer wants to return {oid or 'an item'} for a refund.",
        "billing": f"Customer reports being charged more than once for {oid or 'an order'}.",
        "technical": "Customer needs help getting a product working.",
        "account_security": "Customer reports someone else took over their account and placed an order.",
        "safety": "Customer reports a product overheating and causing damage.",
    }.get(cat, t["subject"])


def _kb_query(tri: dict, ticket: dict) -> str:
    text = (ticket["subject"] + " " + ticket["body"]).lower()
    cat = tri.get("category")
    if cat == "shipping":
        return "parcel delayed lost tracking not updated reshipment"
    if cat == "refund_return":
        if any(w in text for w in DEFECT_WORDS):
            return "defective item refund replacement " + ("laptop screen flickering" if "flicker" in text else "")
        return "standard return window refund change of mind return shipping"
    if cat == "billing":
        return "duplicate charges refund"
    if cat == "technical":
        return "Pulse smartwatch pair" if "watch" in text else ticket["subject"]
    if cat == "account_security":
        return "suspected account takeover changing account email"
    if cat == "safety":
        return "power bank overheating safety incident escalate"
    return ticket["subject"]


def _research(ctx: dict, hist) -> list[dict]:
    t, tri = ctx["ticket"], ctx["triage"]
    done = [n for n, _, _ in hist]
    if "get_customer" not in done:
        return [_text("Starting with the customer record."), _tool("get_customer", email=t["customer_email"])]
    cust = _first(hist, "get_customer") or {}
    oid = tri.get("order_id")
    if oid and "get_order" not in done:
        return [_text(f"Customer found ({cust.get('tier')}). Pulling order {oid}." if cust.get("found") else "No customer record. Checking the order anyway."),
                _tool("get_order", order_id=oid)]
    if not oid and "list_customer_orders" not in done and cust.get("found"):
        return [_text("No order id in the ticket, so I'll list this customer's orders and pick the relevant one."),
                _tool("list_customer_orders", email=t["customer_email"])]
    if "search_knowledge_base" not in done:
        q = _kb_query(tri, t)
        return [_text(f"Now the policy side: searching the knowledge base for '{q}'."), _tool("search_knowledge_base", query=q, top_k=3)]

    # compose
    facts, open_q = [], []
    if cust.get("found"):
        facts.append(f"Customer {cust['name']} is a {cust['tier']} member since {cust['member_since']}, {cust['order_count']} order(s).")
    else:
        open_q.append(f"No customer account found for {t['customer_email']}; ask for the order id or the email used at checkout.")
    order = _first(hist, "get_order")
    if not oid:
        lst = (_first(hist, "list_customer_orders") or {}).get("orders", [])
        if lst:
            text = (t["subject"] + " " + t["body"]).lower()
            match = next((o for o in lst if any(w in o["item"].lower() for w in text.split() if len(w) > 3)), lst[0])
            order, oid = match, match["id"]
            facts.append(f"Ticket has no order id; matched it to {oid} ({match['item']}) from the customer's {len(lst)} order(s).")
        elif cust.get("found"):
            open_q.append("Customer has no orders on record.")
    if order and order.get("found", True):
        facts.append(f"Order {order['id']}: {order['item']}, ${order['amount']:.2f}, status '{order['status']}', ordered {order['ordered_at']}.")
        if order.get("delivered_at"):
            facts.append(f"Delivered {order['delivered_at']} ({order['days_since_delivery']} days ago).")
        elif order.get("shipped_at"):
            facts.append(f"Shipped {order['shipped_at']} ({order['days_since_shipped']} days ago) with {order['carrier']}; "
                         f"tracking last updated {order['days_since_tracking_update']} days ago.")
        else:
            facts.append("Order has not shipped yet.")
        if order.get("charge_count", 1) > 1:
            facts.append(f"Payment was captured {order['charge_count']} times for this order.")
    policies = [{"source": r["source"], "rule": r["text"].split("\n")[0][:260]}
                for r in (_first(hist, "search_knowledge_base") or {}).get("results", [])[:2]]
    out = {
        "customer_found": bool(cust.get("found")), "customer_first_name": cust.get("first_name", ""),
        "customer_tier": cust.get("tier", "unknown"), "order_id": oid or "", "key_facts": facts,
        "policies": policies, "open_questions": open_q,
    }
    return [_text(f"Collected {len(facts)} facts and {len(policies)} relevant policies."), _tool("submit_research", **out)]


def _remedy_reason(tri: dict, ticket: dict) -> str | None:
    text = (ticket["subject"] + " " + ticket["body"]).lower()
    cat = tri.get("category")
    if cat == "shipping":
        return "not_delivered"
    if cat == "billing":
        return "duplicate_charge"
    if cat == "refund_return":
        return "defective" if any(w in text for w in DEFECT_WORDS) else "change_of_mind"
    return None


def _resolution(ctx: dict, hist) -> list[dict]:
    t, tri, res = ctx["ticket"], ctx["triage"], ctx["research"]
    feedback = ctx.get("qa_feedback")
    name = res.get("customer_first_name") or "there"
    oid = res.get("order_id")
    reason = _remedy_reason(tri, t)
    done = [n for n, _, _ in hist]
    sign = "\n\nBest regards,\nNimbus Gear Support"

    if reason and oid and "check_refund_eligibility" not in done and not tri.get("needs_human_now"):
        return [_text(f"Checking with the policy engine what {oid} qualifies for ({reason.replace('_', ' ')})."),
                _tool("check_refund_eligibility", order_id=oid, reason=reason)]
    elig = _first(hist, "check_refund_eligibility") or {}

    blocks: list[dict] = []
    if "propose_action" not in done:
        actions = []
        if tri.get("category") == "account_security":
            actions.append(dict(type="lock_account", reason="Suspected account takeover; unrecognised order " + (oid or "")))
        elif elig.get("eligible"):
            if reason == "not_delivered":
                actions.append(dict(type="reship", order_id=oid, reason=elig["reason"]))
            elif reason in ("change_of_mind", "defective", "duplicate_charge"):
                actions.append(dict(type="refund", order_id=oid, amount=elig["max_refund"], reason=elig["reason"]))
        elif elig.get("store_credit_allowed"):
            actions.append(dict(type="store_credit", order_id=oid, amount=elig["store_credit_allowed"], reason=elig["reason"]))
        if actions:
            blocks.append(_text("Proposing: " + ", ".join(a["type"] for a in actions) + ". These wait for QA and the policy gate."))
            blocks += [_tool("propose_action", **a) for a in actions]
            return blocks

    # write the reply
    cat = tri.get("category")
    conf = 0.88
    if tri.get("needs_human_now") or cat in ("safety", "account_security"):
        if cat == "account_security":
            body = ("I'm sorry to hear someone may have accessed your account. To protect you, we are locking the account now, "
                    "and the unrecognised order will not ship while our security team reviews it. For your safety we can't change "
                    "the account email over email; a security specialist will verify your identity and contact you within 24 hours.")
        else:
            body = ("Thank you for telling us, and I'm very sorry this happened. Please stop using the power bank right away and keep it "
                    "away from anything flammable. Your report has been escalated to a product safety specialist, who will contact you "
                    "within 24 hours to make this right.")
    elif reason == "defective" and elig.get("needs_manager_approval"):
        body = (f"I'm sorry your {oid} laptop screen is flickering, especially after you've already tried the driver update and safe mode. "
                "That points to a hardware defect, which is covered by our defective-item policy. Because of the order value, a support "
                "manager needs to approve the refund; they will review it and contact you within 24 hours.")
        conf = 0.8
    elif reason == "not_delivered" and elig.get("eligible"):
        timeline = "" if not feedback else " It ships within 24 hours and arrives in 3–6 business days, and you'll get a new tracking number by email."
        body = (f"I'm sorry your backpack ({oid}) hasn't arrived. Tracking hasn't updated in a long time, so we're treating the parcel "
                f"as lost and sending you a free replacement.{timeline} If you'd prefer a full refund instead, just reply to this email.")
    elif reason == "not_delivered" and oid and not elig.get("eligible"):
        body = (f"I'm sorry you haven't got your order {oid} yet. {elig.get('reason', '')} Please check with neighbours and around "
                "your entrance, as parcels are sometimes left in a safe place. If it hasn't turned up within 48 hours, reply to this "
                "email and we'll open an investigation with the carrier.")
    elif reason == "duplicate_charge" and not elig.get("eligible"):
        body = (f"Thanks for flagging this. We can only see one completed payment for {oid}. A second line on your statement is "
                "usually a temporary card authorisation, which your bank releases within 3–5 business days. If it is still there "
                "after that, reply with a screenshot of your statement and we'll look into it straight away.")
    elif reason == "change_of_mind" and elig.get("eligible"):
        fee = elig.get("return_shipping_fee", 0)
        body = (f"Thanks for reaching out about your return for {oid}. You're within our 30-day return window, so you'll get a full refund "
                f"of ${elig['max_refund']:.2f}. " + ("Return shipping is free for Gold members. " if not fee else f"A ${fee} return shipping fee applies. ")
                + "Once we receive the item, the refund goes back to your original payment method within 5–7 business days.")
    elif reason == "change_of_mind":
        body = (f"Thanks for getting in touch about {oid}. The item was delivered more than 30 days ago, which is outside our "
                "30-day return window for change-of-mind returns, so we're not able to offer a refund this time. I'm sorry I can't "
                "help more with this one. If the item has a defect, let us know, as defects are covered for 90 days after delivery.")
    elif reason == "duplicate_charge" and elig.get("eligible"):
        body = (f"Sorry about the double charge on {oid}. You were charged twice, so we're refunding the extra ${elig['max_refund']:.2f} "
                "now. It will reach your card within 5–7 business days. No need to return anything.")
    elif cat == "technical":
        body = ("Sorry the Pulse watch won't pair. Please try this: update the Nimbus app, remove the watch from your phone's Bluetooth list, "
                "and restart both devices. Then on the watch go to Settings > System > Reset pairing and pair again from inside the Nimbus app "
                "rather than your phone's Bluetooth menu. If it still won't connect, reply here and we'll send a replacement.")
    else:
        body = ("Sorry to hear your order hasn't arrived. I couldn't find an account under this email address. Could you reply with your "
                "order number (it starts with NG-) or the email you used at checkout? I'll look into it as soon as I have it.")
        conf = 0.75
    reply = f"Hi {name},\n\n{body}{sign}"
    blocks.append(_text("Draft written" + (" with QA feedback applied." if feedback else ".")))
    blocks.append(_tool("submit_resolution", reply=reply, confidence=conf,
                        rationale=f"{cat}: remedy={reason or 'none'}; policy engine said: {elig.get('reason', 'n/a')}"))
    return blocks


def _qa(ctx: dict, hist) -> list[dict]:
    tri, reply = ctx["triage"], ctx.get("draft_reply") or ""
    actions = ctx.get("proposed_actions") or []
    elig = ctx.get("eligibility_results") or {}
    done = [n for n, _, _ in hist]
    if actions and "search_knowledge_base" not in done:
        return [_text("Double-checking the approval limits before signing off."),
                _tool("search_knowledge_base", query="refund approval limits manager", top_k=2)]

    issues, verdict, risk, feedback = [], "approve", 0.12, ""
    big = [c for checks in elig.values() for c in checks if c.get("needs_manager_approval")]
    if tri.get("needs_human_now") or tri.get("category") in ("safety", "account_security"):
        verdict, risk = "escalate", 0.9
        issues.append(f"{tri.get('human_reason') or tri.get('category')}: must be handled by a specialist. Holding reply is appropriate.")
    elif big:
        verdict, risk = "escalate", 0.7
        issues.append("Refund above $500 requires manager approval. Holding reply is appropriate.")
    elif tri.get("category") == "shipping" and actions and "business days" not in reply:
        verdict, risk = "revise", 0.35
        issues.append("Reply promises a replacement but gives no delivery timeline.")
        feedback = "Add when the replacement ships and the 3–6 business day delivery estimate, and mention the new tracking number."
    if not reply.startswith("Hi "):
        verdict, risk = "revise", max(risk, 0.3)
        issues.append("Reply does not greet the customer by name.")
    note = {"approve": "Grounded, within policy, tone OK. Approving.",
            "revise": "Found fixable issues; sending back for revision.",
            "escalate": "This needs a human; recommending escalation."}[verdict]
    return [_text(note), _tool("submit_review", verdict=verdict, risk_score=risk, issues=issues, feedback=feedback)]


POLICIES = {"triage": _triage, "research": _research, "resolution": _resolution, "qa": _qa}


class MockClient:
    name = "demo"

    async def create(self, *, agent, model, system, messages, tools, max_tokens=2048) -> LLMResponse:
        ctx = _tags(messages[0]["content"])
        blocks = POLICIES[agent](ctx, _history(messages))
        if config.DEMO_STEP_DELAY:
            await asyncio.sleep(config.DEMO_STEP_DELAY * random.uniform(0.6, 1.4))
        # rough token estimate (~4 chars/token) so cost and token panels have realistic numbers
        in_tok = (len(system) + len(json.dumps(messages)) + len(json.dumps(tools))) // 4
        out_tok = max(20, len(json.dumps(blocks)) // 4)
        return LLMResponse(content=blocks, stop_reason="tool_use", input_tokens=in_tok,
                           output_tokens=out_tok, model=model, meta={"mock": True})
