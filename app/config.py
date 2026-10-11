"""Central configuration. Everything is overridable with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    """Minimal .env loader (avoids an extra dependency). Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

# DEMO_MODE=1 forces the offline scripted LLM even if a key is present.
# With no key at all we fall back to demo mode automatically.
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1" or not ANTHROPIC_API_KEY

# Model per agent. Cheap/fast model for classification, stronger model for reasoning.
MODELS = {
    "triage": os.getenv("TRIAGE_MODEL", "claude-haiku-5-5"),
    "research": os.getenv("RESEARCH_MODEL", "claude-sonnet-5-5"),
    "resolution": os.getenv("RESOLUTION_MODEL", "claude-sonnet-5-5"),
    "qa": os.getenv("QA_MODEL", "claude-sonnet-5-5"),
}

# USD per million tokens (input, output). Check current prices at
# https://platform.claude.com/docs/en/about-claude/pricing and update if needed.
PRICING = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
}

# Hard business rules enforced in code, never left to the model.
REFUND_AUTO_LIMIT = float(os.getenv("REFUND_AUTO_LIMIT", "500"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.6"))
MAX_RISK = float(os.getenv("MAX_RISK", "0.6"))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
MAX_PARALLEL_RUNS = int(os.getenv("MAX_PARALLEL_RUNS", "3"))

# Demo mode adds small delays so you can watch each step happen.
DEMO_STEP_DELAY = float(os.getenv("DEMO_STEP_DELAY", "0.6"))

DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "nimbus.db"))
KB_DIR = ROOT / "app" / "kb"
TICKETS_PATH = ROOT / "data" / "sample_tickets.json"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * inp + output_tokens * out) / 1_000_000
