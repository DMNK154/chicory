"""Anthropic Claude API client wrapper."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from chicory.config import ChicoryConfig
from chicory.exceptions import LLMError
from chicory.llm.base import BaseLLMClient
from chicory.llm.prompts import build_system_prompt
from chicory.llm.types import ContentBlock, LLMResponse, TextBlock, ToolUseBlock
from chicory.orchestrator.tool_definitions import CHICORY_TOOLS


class ClaudeClient(BaseLLMClient):
    """Wrapper around the Anthropic API for Claude interactions with memory tools."""

    def __init__(self, config: ChicoryConfig, active_tags: list[str] | None = None) -> None:
        super().__init__(config, active_tags)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> LLMResponse:
        """Send a message to Claude with memory tools available."""
        sys_prompt = system or build_system_prompt(self._active_tags)

        try:
            response = self._client.messages.create(
                model=self._config.llm_model,
                max_tokens=self._config.max_tokens,
                system=sys_prompt,
                messages=self._convert_messages(messages),
                tools=CHICORY_TOOLS,
            )
            return self._convert_response(response)
        except anthropic.APIError as e:
            raise LLMError(f"Claude API error: {e}") from e

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal messages (with our ContentBlock types) to Anthropic format.

        When ChatSession appends response.content (list of our dataclasses) into
        the message history and passes it back, we need to serialize those back
        to dicts that the Anthropic SDK expects.
        """
        result = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                converted = []
                for item in content:
                    if isinstance(item, TextBlock):
                        converted.append({"type": "text", "text": item.text})
                    elif isinstance(item, ToolUseBlock):
                        converted.append({
                            "type": "tool_use",
                            "id": item.id,
                            "name": item.name,
                            "input": item.input,
                        })
                    else:
                        # Already a dict (e.g., tool_result from ChatSession)
                        converted.append(item)
                result.append({"role": msg["role"], "content": converted})
            else:
                result.append(msg)
        return result

    def _convert_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Convert Anthropic Message to our LLMResponse."""
        content: list[ContentBlock] = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return LLMResponse(
            content=content,
            stop_reason=response.stop_reason or "end_turn",
            model=response.model or "",
            usage=usage,
        )

    # --- Anthropic-specific methods (not part of the base interface) ---

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
