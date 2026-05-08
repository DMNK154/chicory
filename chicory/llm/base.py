"""Base LLM client interface."""

from __future__ import annotations

from typing import Any

from chicory.config import ChicoryConfig
from chicory.llm.types import LLMResponse


class BaseLLMClient:
    """Base class for LLM client implementations.

    Subclasses must implement:
        - chat(messages, system) -> LLMResponse

    Optional overrides:
        - update_active_tags(tags) -- called before each chat turn
        - close() -- cleanup resources
    """

    def __init__(self, config: ChicoryConfig, active_tags: list[str] | None = None) -> None:
        self._config = config
        self._active_tags = active_tags or []

    def update_active_tags(self, tags: list[str]) -> None:
        """Update the active tags shown in the system prompt."""
        self._active_tags = tags

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM with memory tools available.

        Args:
            messages: Conversation history as list of role/content dicts.
            system: Optional system prompt override.

        Returns:
            LLMResponse with normalized content blocks and stop_reason.
        """
        raise NotImplementedError

    def propose_tags(self, content: str, existing_tags: list[str]) -> list[str]:
        """Suggest tags for content. Override in subclasses with LLM access."""
        return []

    def close(self) -> None:
        """Release any resources held by the client."""
