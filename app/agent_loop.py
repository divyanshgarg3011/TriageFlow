"""Generic tool-use loop shared by every agent.

    call model -> run the tools it asked for -> send results back -> repeat
    until the agent calls its submit_* tool (structured output) or runs out of steps.

Every step is emitted as an event, which is what the UI animates.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import config
from .agents import AgentSpec
from .events import bus
from .llm import get_client
from .tools import TOOLS, run_tool


class AgentError(RuntimeError):
    pass


@dataclass
class Metrics:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    llm_ms: int = 0

    def as_dict(self) -> dict:
        return {**self.__dict__, "cost_usd": round(self.cost_usd, 5)}


@dataclass
class RunContext:
    """What an agent needs from the run it belongs to."""
    run_id: str
    metrics: Metrics = field(default_factory=Metrics)
    eligibility: dict[str, list[dict]] = field(default_factory=dict)  # order_id -> policy engine answers


@dataclass
class AgentResult:
    output: dict
    proposed_actions: list[dict]
    steps: int


def _missing_keys(schema: dict, data: dict) -> list[str]:
    return [k for k in schema.get("required", []) if k not in data]


def _short(value: Any, limit: int = 4000) -> Any:
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit] + "…"


async def run_agent(spec: AgentSpec, ctx: RunContext, user_content: str, *, attempt: int = 1) -> AgentResult:
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": user_content}]
    proposed: list[dict] = []
    tool_specs = spec.tool_specs()
    out_name = spec.output.name

    bus.emit(ctx.run_id, "agent_started", spec.key, label=spec.label, model=spec.model, attempt=attempt,
             tools=[t["name"] for t in tool_specs])

    for step in range(1, config.MAX_AGENT_STEPS + 1):
        t0 = time.perf_counter()
        resp = await client.create(agent=spec.key, model=spec.model, system=spec.system,
                                   messages=messages, tools=tool_specs, max_tokens=spec.max_tokens)
        ms = int((time.perf_counter() - t0) * 1000)
        cost = config.cost_usd(spec.model, resp.input_tokens, resp.output_tokens)
        m = ctx.metrics
        m.input_tokens += resp.input_tokens
        m.output_tokens += resp.output_tokens
        m.cost_usd += cost
        m.llm_calls += 1
        m.llm_ms += ms
        bus.emit(ctx.run_id, "llm_call", spec.key, model=spec.model, step=step, ms=ms,
                 input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                 cost_usd=round(cost, 6), stop_reason=resp.stop_reason)

        messages.append({"role": "assistant", "content": resp.content})

        for block in resp.content:
            if block["type"] == "text" and block["text"].strip():
                bus.emit(ctx.run_id, "thought", spec.key, text=block["text"].strip())
            elif block["type"] == "thinking" and block.get("thinking", "").strip():
                bus.emit(ctx.run_id, "thinking", spec.key, text=block["thinking"].strip()[:1500])

        tool_uses = [b for b in resp.content if b["type"] == "tool_use"]
        if not tool_uses:
            # The model stopped without giving its structured answer: nudge it once more.
            bus.emit(ctx.run_id, "nudge", spec.key, text=f"No {out_name} call yet; asking the agent to finish.")
            messages.append({"role": "user", "content": f"Call {out_name} now with your final answer."})
            continue

        results: list[dict] = []
        final: dict | None = None
        for tu in tool_uses:
            name, args, tid = tu["name"], tu.get("input") or {}, tu["id"]

            if name == out_name:
                missing = _missing_keys(spec.output.schema, args)
                if missing:
                    bus.emit(ctx.run_id, "validation_error", spec.key, missing=missing)
                    results.append({"type": "tool_result", "tool_use_id": tid, "is_error": True,
                                    "content": f"Missing required fields: {', '.join(missing)}. Call {out_name} again with all fields."})
                    continue
                final = args
                results.append({"type": "tool_result", "tool_use_id": tid, "content": "Recorded."})
                continue

            bus.emit(ctx.run_id, "tool_call", spec.key, tool=name, args=args, call_id=tid)
            m.tool_calls += 1
            t1 = time.perf_counter()
            try:
                if name not in spec.tools:
                    raise ValueError(f"Tool {name} is not available to this agent")
                if name == "propose_action":
                    action = {k: v for k, v in args.items() if v not in (None, "")}
                    proposed.append(action)
                    result: Any = {"recorded": True, "note": "Queued for QA review and policy gates; not executed yet."}
                else:
                    result = await asyncio.to_thread(run_tool, name, args)
                    if name == "check_refund_eligibility" and isinstance(result, dict) and result.get("order_id"):
                        ctx.eligibility.setdefault(result["order_id"], []).append({"reason": args.get("reason"), **result})
                is_error = False
            except Exception as exc:  # tool errors go back to the model, they don't crash the run
                result, is_error = {"error": str(exc)}, True
            bus.emit(ctx.run_id, "tool_result", spec.key, tool=name, call_id=tid, ok=not is_error,
                     ms=int((time.perf_counter() - t1) * 1000), result=_short(result))
            results.append({"type": "tool_result", "tool_use_id": tid, "is_error": is_error,
                            "content": json.dumps(result, default=str)})

        if final is not None:
            bus.emit(ctx.run_id, "agent_finished", spec.key, label=spec.label, output=final,
                     proposed_actions=proposed, steps=step)
            return AgentResult(output=final, proposed_actions=proposed, steps=step)

        messages.append({"role": "user", "content": results})

    bus.emit(ctx.run_id, "agent_failed", spec.key, reason=f"No result after {config.MAX_AGENT_STEPS} steps")
    raise AgentError(f"{spec.label} agent did not finish within {config.MAX_AGENT_STEPS} steps")
