"""Tools the agents can call. Each tool = a JSON schema (what Claude sees) + a Python function.

Read-only tools query the store. `propose_action` only RECORDS an intent; nothing changes in the
store until the orchestrator's policy gate (or a human) approves it. Agents propose, code disposes.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from . import config, db

# ---------------------------------------------------------------- knowledge base search


@dataclass
class KBChunk:
    source: str
    heading: str
    text: str
    tokens: Counter = field(default_factory=Counter)


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the and or of to for in on is are be it this that with as at by we our your you can not no any".split())


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


class KnowledgeBase:
    """Tiny TF-IDF search over markdown docs, chunked by '##' sections. No external deps."""

    def __init__(self) -> None:
        self.chunks: list[KBChunk] = []
        for path in sorted(config.KB_DIR.glob("*.md")):
            title = path.stem
            for part in re.split(r"\n(?=## )", path.read_text()):
                lines = part.strip().splitlines()
                if not lines or lines[0].startswith("# "):
                    lines = lines[1:]
                if not lines:
                    continue
                heading = lines[0].lstrip("# ").strip()
                body = "\n".join(lines[1:]).strip()
                chunk = KBChunk(title, heading, body)
                chunk.tokens = Counter(_tokenize(heading + " " + heading + " " + body))
                self.chunks.append(chunk)
        n = len(self.chunks)
        df: Counter = Counter()
        for c in self.chunks:
            df.update(set(c.tokens))
        self.idf = {t: math.log((n + 1) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = Counter(_tokenize(query))
        scored = []
        for c in self.chunks:
            score = sum(q[t] * c.tokens[t] * self.idf.get(t, 0) ** 2 for t in q) / math.sqrt(sum(c.tokens.values()) + 1)
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [
            {"source": f"{c.source}.md#{c.heading}", "heading": c.heading, "text": c.text, "score": round(s, 3)}
            for s, c in scored[:top_k]
        ]


_KB: KnowledgeBase | None = None


def kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


# ---------------------------------------------------------------- tool implementations


def _days_since(iso: str | None) -> int | None:
    return (date.today() - date.fromisoformat(iso)).days if iso else None


def get_customer(email: str) -> dict:
    with db.session() as conn:
        row = conn.execute("SELECT * FROM customers WHERE email = ?", (email.strip().lower(),)).fetchone()
        if not row:
            return {"found": False, "email": email, "message": "No customer with this email."}
        n = conn.execute("SELECT COUNT(*) FROM orders WHERE email = ?", (row["email"],)).fetchone()[0]
    out = db.row_to_dict(row)
    out.update(found=True, order_count=n, first_name=row["name"].split()[0])
    return out


def _order_view(row) -> dict:
    o = db.row_to_dict(row)
    o["days_since_delivery"] = _days_since(o["delivered_at"])
    o["days_since_shipped"] = _days_since(o["shipped_at"])
    o["days_since_tracking_update"] = _days_since(o["tracking_last_update"])
    return o


def get_order(order_id: str) -> dict:
    with db.session() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id.strip().upper(),)).fetchone()
        if not row:
            return {"found": False, "order_id": order_id, "message": "No order with this id."}
        refunds = [dict(r) for r in conn.execute("SELECT amount, reason, created_at FROM refunds WHERE order_id = ?", (row["id"],))]
    out = _order_view(row)
    out.update(found=True, refunds=refunds)
    return out


def list_customer_orders(email: str) -> dict:
    with db.session() as conn:
        rows = conn.execute("SELECT * FROM orders WHERE email = ? ORDER BY ordered_at DESC", (email.strip().lower(),)).fetchall()
    return {"email": email, "orders": [_order_view(r) for r in rows]}


def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    return {"query": query, "results": kb().search(query, top_k=int(top_k))}


def check_refund_eligibility(order_id: str, reason: str) -> dict:
    """Deterministic policy engine. The model asks; code decides what policy allows."""
    order = get_order(order_id)
    if not order.get("found"):
        return {"eligible": False, "reason": "Order not found."}
    cust = get_customer(order["email"])
    amount = float(order["amount"])
    days = order["days_since_delivery"]
    already = sum(r["amount"] for r in order["refunds"])
    base = {"order_id": order["id"], "order_amount": amount, "customer_tier": cust.get("tier")}

    if already >= amount:
        return {**base, "eligible": False, "reason": "Order has already been fully refunded."}
    if reason == "duplicate_charge":
        extra = order["charge_count"] - 1
        if extra < 1:
            return {**base, "eligible": False, "reason": "Only one charge found for this order."}
        return {**base, "eligible": True, "max_refund": amount * extra, "remedy": "refund",
                "reason": f"{extra} duplicate charge(s) found; refund each extra charge.",
                "needs_manager_approval": amount * extra > config.REFUND_AUTO_LIMIT}
    if reason == "not_delivered":
        stale = order["days_since_tracking_update"]
        shipped = order["days_since_shipped"]
        if order["status"] == "delivered":
            return {**base, "eligible": False, "reason": "Carrier marks this order as delivered."}
        if (stale is not None and stale >= 7) or (shipped is not None and shipped >= 10):
            return {**base, "eligible": True, "max_refund": amount, "remedy": "reship_or_refund",
                    "reason": f"Treated as lost: shipped {shipped} days ago, no tracking update for {stale} days.",
                    "needs_manager_approval": amount > config.REFUND_AUTO_LIMIT}
        return {**base, "eligible": False, "reason": "Parcel still within normal delivery time."}
    if days is None:
        return {**base, "eligible": False, "reason": "Order has not been delivered yet."}
    if reason == "defective":
        if days <= 90:
            return {**base, "eligible": True, "max_refund": amount, "remedy": "refund_or_replacement",
                    "reason": f"Defective item reported {days} days after delivery (limit 90).",
                    "needs_manager_approval": amount > config.REFUND_AUTO_LIMIT}
        return {**base, "eligible": False, "reason": f"Defect reported {days} days after delivery; limit is 90."}
    # change of mind
    if days <= 30:
        return {**base, "eligible": True, "max_refund": amount, "remedy": "refund_after_return",
                "return_shipping_fee": 0 if cust.get("tier") == "gold" else 7.99,
                "reason": f"Within 30-day return window ({days} days since delivery).",
                "needs_manager_approval": amount > config.REFUND_AUTO_LIMIT}
    if cust.get("tier") == "gold":
        return {**base, "eligible": False, "store_credit_allowed": round(amount * 0.5, 2),
                "reason": f"Outside 30-day window ({days} days). Gold member: 50% store credit allowed."}
    return {**base, "eligible": False, "reason": f"Outside 30-day return window ({days} days since delivery)."}


# ---------------------------------------------------------------- registry


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[..., Any] | None = None  # None = handled by the agent loop (e.g. propose_action)

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("get_customer", "Look up a customer profile (name, membership tier, notes) by email address.",
             _obj({"email": {"type": "string"}}, ["email"]), get_customer),
        Tool("get_order", "Get one order by id (e.g. NG-5521): item, amount, status, dates, carrier, charge count, refunds.",
             _obj({"order_id": {"type": "string"}}, ["order_id"]), get_order),
        Tool("list_customer_orders", "List all orders for a customer email. Use when the ticket has no order id.",
             _obj({"email": {"type": "string"}}, ["email"]), list_customer_orders),
        Tool("search_knowledge_base", "Search company policies and troubleshooting guides. Returns the best matching sections with sources.",
             _obj({"query": {"type": "string"}, "top_k": {"type": "integer", "default": 3}}, ["query"]), search_knowledge_base),
        Tool("check_refund_eligibility",
             "Ask the policy engine whether an order qualifies for a refund/replacement and the maximum allowed amount. Always call this before proposing a refund, reship or store credit.",
             _obj({"order_id": {"type": "string"},
                   "reason": {"type": "string", "enum": ["change_of_mind", "defective", "not_delivered", "duplicate_charge"]}},
                  ["order_id", "reason"]), check_refund_eligibility),
        Tool("propose_action",
             "Propose an account action. It is NOT executed now: it is reviewed by QA and policy gates first. Call once per action.",
             _obj({"type": {"type": "string", "enum": ["refund", "reship", "replacement", "store_credit", "lock_account", "password_reset_link"]},
                   "order_id": {"type": "string"},
                   "amount": {"type": "number", "description": "USD, for refund/store_credit"},
                   "reason": {"type": "string"}},
                  ["type", "reason"])),
    ]
}


def run_tool(name: str, args: dict) -> Any:
    tool = TOOLS[name]
    if tool.fn is None:
        raise ValueError(f"{name} is handled by the agent loop")
    return tool.fn(**args)
