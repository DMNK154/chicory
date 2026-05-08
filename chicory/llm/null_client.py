"""Null LLM client for API-only / MCP-only use."""

from __future__ import annotations

from typing import Any

from chicory.llm.base import BaseLLMClient
from chicory.llm.types import LLMResponse, TextBlock


class NullClient(BaseLLMClient):
    """LLM client that returns a fixed message.

    Used when no LLM API key is configured, allowing the MCP server
    and orchestrator to function without an LLM backend.
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=[TextBlock(text="[No LLM configured. Use MCP tools or CLI commands directly.]")],
            stop_reason="end_turn",
        )
