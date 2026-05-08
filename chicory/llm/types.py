"""Provider-agnostic LLM response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextBlock:
    """A text content block from an LLM response."""

    text: str
    type: str = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    """A tool-use content block from an LLM response."""

    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


ContentBlock = TextBlock | ToolUseBlock


@dataclass(frozen=True)
class LLMResponse:
    """Provider-agnostic LLM response.

    Attributes:
        content: List of content blocks (text and/or tool_use).
        stop_reason: Why the model stopped. Normalized values:
            - "end_turn": model finished naturally
            - "tool_use": model wants to call a tool
            - "max_tokens": response was truncated
        model: Model identifier string.
        usage: Token usage dict (provider-specific keys).
    """

    content: list[ContentBlock]
    stop_reason: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
