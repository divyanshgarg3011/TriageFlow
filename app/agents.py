"""The four specialist agents: their prompts, tools and structured-output contracts.

Each agent finishes by calling its own `submit_*` tool. Using a tool as the output channel gives us
schema-shaped JSON we can validate, instead of parsing free text.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .tools import TOOLS, Tool, _obj

COMPANY = "Nimbus Gear, an online store for outdoor gear and consumer electronics"


@dataclass
class AgentSpec:
    key: str
    label: str
    role: str                 # one line shown in the UI
    system: str
    tools: list[str]
    output: Tool
    max_tokens: int = 2048

    @property
    def model(self) -> str:
        return config.MODELS[self.key]

    def tool_specs(self) -> list[dict]:
        return [TOOLS[t].spec() for t in self.tools] + [self.output.spec()]


# ---------------------------------------------------------------- output contracts

SUBMIT_TRIAGE = Tool(
    "submit_triage",
    "Submit your final triage of the ticket. Call exactly once.",
    _obj(
        {
            "category": {"type": "string", "enum": ["shipping", "refund_return", "billing", "technical", "account_security", "safety", "other"]},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "frustrated", "angry"]},
            "order_id": {"type": "string", "description": "Order id mentioned in the ticket (e.g. NG-5521), or empty string"},
            "summary": {"type": "string", "description": "One sentence: what the customer wants"},
            "needs_human_now": {"type": "boolean", "description": "True for legal threats, safety incidents, suspected account takeover or fraud"},
            "human_reason": {"type": "string", "description": "Why a human is needed, or empty string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        ["category", "priority", "sentiment", "order_id", "summary", "needs_human_now", "human_reason", "confidence"],
    ),
)

SUBMIT_RESEARCH = Tool(
    "submit_research",
    "Submit the verified facts and policies the resolution agent needs. Call exactly once.",
    _obj(
        {
            "customer_found": {"type": "boolean"},
            "customer_first_name": {"type": "string"},
            "customer_tier": {"type": "string"},
            "order_id": {"type": "string", "description": "The order this ticket is about, or empty string"},
            "key_facts": {"type": "array", "items": {"type": "string"}, "description": "Facts copied from tool results only"},
            "policies": {"type": "array", "items": {"type": "object", "properties": {"source": {"type": "string"}, "rule": {"type": "string"}},
                                                    "required": ["source", "rule"]}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        ["customer_found", "customer_first_name", "customer_tier", "order_id", "key_facts", "policies", "open_questions"],
    ),
)

SUBMIT_RESOLUTION = Tool(
    "submit_resolution",
    "Submit the customer reply. Propose any actions with propose_action BEFORE calling this. Call exactly once.",
    _obj(
        {
            "reply": {"type": "string", "description": "The full email reply to the customer"},
            "rationale": {"type": "string", "description": "Short internal note: why this resolution"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        ["reply", "rationale", "confidence"],
    ),
)

SUBMIT_REVIEW = Tool(
    "submit_review",
    "Submit your QA verdict. Call exactly once.",
    _obj(
        {
            "verdict": {"type": "string", "enum": ["approve", "revise", "escalate"]},
            "risk_score": {"type": "number", "minimum": 0, "maximum": 1, "description": "0 = safe to send, 1 = very risky"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "feedback": {"type": "string", "description": "Concrete instructions for the resolution agent if verdict is revise"},
        },
        ["verdict", "risk_score", "issues", "feedback"],
    ),
)

# ---------------------------------------------------------------- agents

TRIAGE = AgentSpec(
    key="triage",
    label="Triage",
    role="Classifies the ticket, sets priority and flags anything a human must see",
    system=f"""You are the triage agent for the support desk of {COMPANY}.
Read one customer ticket and classify it. You do not reply to the customer.

Rules:
- category: shipping (late/lost parcels), refund_return (returns, refunds, defects where the customer wants money back or a swap),
  billing (charges, invoices, double charges), technical (how-to, setup, product not working where they want help),
  account_security (login, password, someone else using the account), safety (heat, fire, burns, injury, property damage), other.
- priority: urgent for safety, legal threats, or account takeover; high for money problems or angry customers; medium for normal issues; low for questions.
- needs_human_now = true ONLY for legal threats (lawyer, sue, chargeback), safety incidents, suspected account takeover or fraud.
- Extract the order id exactly as written (format NG-1234) or use an empty string.
- Be calibrated: confidence below 0.6 means you are genuinely unsure.
Finish by calling submit_triage.""",
    tools=[],
    output=SUBMIT_TRIAGE,
    max_tokens=1024,
)

RESEARCH = AgentSpec(
    key="research",
    label="Research",
    role="Pulls customer, order and policy facts with tools",
    system=f"""You are the research agent for the support desk of {COMPANY}.
Gather every fact the resolution agent will need. You never write to the customer.

How to work:
1. Look up the customer by email with get_customer.
2. If there is an order id, call get_order. If not, call list_customer_orders and identify the most likely order.
3. Search the knowledge base for the policies that apply (one or two focused queries).
4. Submit with submit_research.

Rules:
- key_facts must come from tool results only. Never guess dates, amounts or statuses.
- Include days since delivery / shipping when relevant; they decide policy outcomes.
- If something important is missing (no customer record, no matching order), add it to open_questions.""",
    tools=["get_customer", "get_order", "list_customer_orders", "search_knowledge_base"],
    output=SUBMIT_RESEARCH,
)

RESOLUTION = AgentSpec(
    key="resolution",
    label="Resolution",
    role="Decides the fix, proposes actions and drafts the reply",
    system=f"""You are the resolution agent for the support desk of {COMPANY}.
Using the triage and research provided, decide how to resolve the ticket and write the reply email.

How to work:
1. If any money or item remedy might apply (refund, reship, replacement, store credit), call check_refund_eligibility first.
   Only offer what it says is allowed, and never more than its max_refund.
2. Propose each concrete action with propose_action (one call per action). Actions are executed later, after review.
3. Call submit_resolution with the reply.

Reply rules:
- Greet the customer by first name. Acknowledge the problem in the first sentence.
- Say exactly what happens next and when (use the timelines from policy).
- Only state facts from the research. Never invent order details, dates or amounts.
- If the case needs a human (safety, legal, account takeover, refund above $500, or a policy you are unsure about),
  write a short holding reply that says a specialist will contact them within 24 hours, and do not promise a remedy.
  For suspected account takeover, propose lock_account.
- If you need information from the customer (e.g. order id), ask for it clearly and propose no actions.
- Plain, warm, concise. Sign off as "Nimbus Gear Support". No markdown.
If you receive QA feedback, fix every issue it lists.""",
    tools=["check_refund_eligibility", "search_knowledge_base", "propose_action"],
    output=SUBMIT_RESOLUTION,
)

QA = AgentSpec(
    key="qa",
    label="QA review",
    role="Checks the draft for made-up facts, policy breaks and tone",
    system=f"""You are the QA reviewer for the support desk of {COMPANY}.
You check a drafted reply and its proposed actions before anything reaches the customer.

Check, in order:
1. Grounding: every date, amount, status and item in the reply appears in the research facts or eligibility results. Anything else is a hallucination.
2. Policy: actions and promises match the policies and the eligibility results (amounts within max_refund, correct remedy).
3. Safety: legal threats, safety incidents, account takeover and refunds above $500 must not be auto-resolved; the reply must be a holding reply.
4. Tone and completeness: uses the first name, acknowledges the issue, clear next step, no markdown.

Verdict:
- approve: correct and safe to send as is.
- revise: fixable problems; give precise feedback. Use this for wording, missing next steps or small factual slips.
- escalate: needs a human (safety, legal, takeover, large refund, or the policy is genuinely unclear).
You can use search_knowledge_base to double-check a policy. Finish with submit_review.""",
    tools=["search_knowledge_base"],
    output=SUBMIT_REVIEW,
)

AGENTS = {a.key: a for a in (TRIAGE, RESEARCH, RESOLUTION, QA)}
