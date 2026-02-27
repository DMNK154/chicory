"""Anthropic Claude API client wrapper."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from chicory.config import ChicoryConfig
from chicory.exceptions import LLMError
from chicory.llm.prompts import build_system_prompt
from chicory.orchestrator.tool_definitions import CHICORY_TOOLS


class ClaudeClient:
    """Wrapper around the Anthropic API for Claude interactions with memory tools."""

    def __init__(self, config: ChicoryConfig, active_tags: list[str] | None = None) -> None:
        self._config = config
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._active_tags = active_tags or []

    def update_active_tags(self, tags: list[str]) -> None:
        """Update the active tags shown in the system prompt."""
        self._active_tags = tags

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> anthropic.types.Message:
        """Send a message to Claude with memory tools available."""
        sys_prompt = system or build_system_prompt(self._active_tags)

        try:
            response = self._client.messages.create(
                model=self._config.llm_model,
                max_tokens=self._config.max_tokens,
                system=sys_prompt,
                messages=messages,
                tools=CHICORY_TOOLS,
            )
            return response
        except anthropic.APIError as e:
            raise LLMError(f"Claude API error: {e}") from e

    def judge_salience(self, content: str, context: str = "") -> float:
        """Ask Claude to rate the importance of a memory. Returns 0.0-1.0."""
        from chicory.llm.prompts import SALIENCE_PROMPT

        prompt = SALIENCE_PROMPT.format(content=content, context=context)
        try:
            response = self._client.messages.create(
                model=self._config.llm_model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            return max(0.0, min(1.0, float(text)))
        except (ValueError, IndexError, anthropic.APIError):
            return 0.5  # Default on failure

    def propose_tags(self, content: str, existing_tags: list[str]) -> list[str]:
        """Ask Claude to suggest tags for a memory."""
        from chicory.llm.prompts import TAG_PROPOSAL_PROMPT

        prompt = TAG_PROPOSAL_PROMPT.format(
            content=content,
            existing_tags=", ".join(existing_tags) if existing_tags else "(none)",
        )
        try:
            response = self._client.messages.create(
                model=self._config.llm_model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            tags = json.loads(text)
            if isinstance(tags, list):
                return [str(t) for t in tags]
        except (json.JSONDecodeError, IndexError, anthropic.APIError):
            pass
        return []
