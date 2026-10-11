"""LLM clients behind one tiny interface so the agent loop does not care which one it talks to.

- ClaudeClient: real Anthropic Messages API with tool use.
- MockClient (mock_llm.py): offline, scripted, deterministic. Same response shape, so the whole
  pipeline + UI can be demoed and tested without an API key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from . import config


@dataclass
class LLMResponse:
    content: list[dict]                 # Anthropic-style content blocks as plain dicts
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    meta: dict = field(default_factory=dict)


class LLMClient(Protocol):
    name: str

    async def create(self, *, agent: str, model: str, system: str, messages: list[dict],
                     tools: list[dict], max_tokens: int = 2048) -> LLMResponse: ...


def _block_to_dict(block) -> dict | None:
    """Keep only the fields the API accepts back on the next turn."""
    t = block.type
    if t == "text":
        return {"type": "text", "text": block.text}
    if t == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if t == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if t == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    return None


class ClaudeClient:
    name = "claude"

    def __init__(self) -> None:
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3)

    async def create(self, *, agent: str, model: str, system: str, messages: list[dict],
                     tools: list[dict], max_tokens: int = 2048) -> LLMResponse:
        resp = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        blocks = [b for b in (_block_to_dict(x) for x in resp.content) if b]
        return LLMResponse(
            content=blocks,
            stop_reason=resp.stop_reason or "",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
        )


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        if config.DEMO_MODE:
            from .mock_llm import MockClient

            _client = MockClient()
        else:
            _client = ClaudeClient()
    return _client
