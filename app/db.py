"""SQLite store for the fictional 'Nimbus Gear' shop: customers, orders, refunds, executed actions.

Dates are seeded relative to today so the return-window rules always behave the same way.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

from . import config

SCHEMA = """
CREATE TABLE customers (
    email TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    tier TEXT NOT NULL,            -- gold | standard
    member_since TEXT NOT NULL,
    lifetime_value REAL NOT NULL,
    notes TEXT
);
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL REFERENCES customers(email),
    item TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL,          -- processing | shipped | delivered | refunded | reship_requested
    ordered_at TEXT NOT NULL,
    shipped_at TEXT,
    delivered_at TEXT,
    carrier TEXT,
    tracking_last_update TEXT,
    charge_count INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    amount REAL NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE executed_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    approved_by TEXT NOT NULL,     -- 'policy-gate' (auto) or 'human'
    executed_at TEXT NOT NULL
);
"""


def _d(days_ago: int | None) -> str | None:
    if days_ago is None:
        return None
    return (date.today() - timedelta(days=days_ago)).isoformat()


CUSTOMERS = [
    ("aisha.khan@example.com", "Aisha Khan", "gold", 900, 2140.50, None),
    ("rohan.mehta@example.com", "Rohan Mehta", "standard", 400, 389.00, None),
    ("lena.fischer@example.com", "Lena Fischer", "gold", 1300, 5320.00, None),
    ("omar.haddad@example.com", "Omar Haddad", "standard", 200, 159.00, None),
    ("priya.nair@example.com", "Priya Nair", "standard", 30, 199.00, "First order"),
    ("sofia.rossi@example.com", "Sofia Rossi", "standard", 700, 612.00, None),
    ("daniel.okafor@example.com", "Daniel Okafor", "gold", 1100, 3010.00, None),
    ("mei.chen@example.com", "Mei Chen", "standard", 500, 230.00, None),
]

# id, email, item, amount, status, ordered, shipped, delivered, carrier, tracking_update, charges
ORDERS = [
    ("NG-5521", "rohan.mehta@example.com", "Trailblazer 45L backpack", 129.00, "shipped", 16, 14, None, "SwiftPost", 12, 1),
    ("NG-5530", "aisha.khan@example.com", "AeroBuds Pro earbuds", 89.00, "delivered", 15, 13, 10, "SwiftPost", 10, 1),
    ("NG-5544", "lena.fischer@example.com", "Nimbus Book 14 laptop", 1299.00, "delivered", 25, 23, 20, "AirLine Express", 20, 1),
    ("NG-5550", "omar.haddad@example.com", "Volt 20K power bank", 59.00, "delivered", 45, 43, 40, "SwiftPost", 40, 1),
    ("NG-5552", "priya.nair@example.com", "Pulse smartwatch", 199.00, "delivered", 9, 7, 5, "SwiftPost", 5, 1),
    ("NG-5539", "sofia.rossi@example.com", "Summit rain jacket", 149.00, "delivered", 66, 64, 60, "SwiftPost", 60, 1),
    ("NG-5560", "daniel.okafor@example.com", "Lumen desk lamp", 149.00, "delivered", 7, 5, 3, "AirLine Express", 3, 2),
    ("NG-5571", "aisha.khan@example.com", "Trail socks 3-pack", 24.00, "delivered", 75, 73, 70, "SwiftPost", 70, 1),
    ("NG-5575", "mei.chen@example.com", "Nimbus Book 14 laptop", 1299.00, "processing", 1, None, None, None, None, 1),
    ("NG-5580", "lena.fischer@example.com", "AeroBuds Pro earbuds", 89.00, "delivered", 30, 28, 26, "SwiftPost", 26, 1),
    ("NG-5583", "rohan.mehta@example.com", "Pulse smartwatch", 199.00, "delivered", 14, 12, 9, "SwiftPost", 9, 1),
]


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def session():
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(reset: bool = True) -> None:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if reset and config.DB_PATH.exists():
        config.DB_PATH.unlink()
    if config.DB_PATH.exists():
        return
    with session() as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO customers VALUES (?,?,?,?,?,?)",
            [(e, n, t, _d(since), ltv, notes) for e, n, t, since, ltv, notes in CUSTOMERS],
        )
        conn.executemany(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (i, e, item, amt, st, _d(o), _d(s), _d(dl), car, _d(tr), ch)
                for i, e, item, amt, st, o, s, dl, car, tr, ch in ORDERS
            ],
        )


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def record_action(ticket_id: str, run_id: str, action: dict, approved_by: str) -> None:
    """Apply an approved action to the store. This is the ONLY place state changes."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with session() as conn:
        kind = action.get("type")
        order_id = action.get("order_id")
        if kind in ("refund", "store_credit") and order_id:
            conn.execute(
                "INSERT INTO refunds (order_id, amount, reason, created_at) VALUES (?,?,?,?)",
                (order_id, float(action.get("amount") or 0), f"{kind}: {action.get('reason', '')}", now),
            )
            if kind == "refund":
                conn.execute("UPDATE orders SET status='refunded' WHERE id=?", (order_id,))
        elif kind == "reship" and order_id:
            conn.execute("UPDATE orders SET status='reship_requested' WHERE id=?", (order_id,))
        conn.execute(
            "INSERT INTO executed_actions (ticket_id, run_id, type, payload, approved_by, executed_at) VALUES (?,?,?,?,?,?)",
            (ticket_id, run_id, kind or "none", json.dumps(action), approved_by, now),
        )


def list_customers() -> list[dict]:
    with session() as conn:
        return [dict(r) for r in conn.execute("SELECT email, name, tier FROM customers ORDER BY name")]
