"""OpenAI-compatible LLM client."""

from __future__ import annotations

import json
from typing import Any

from chicory.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROK_MODEL,
    DEFAULT_OPENAI_MODEL,
    ChicoryConfig,
)
from chicory.exceptions import LLMError
from chicory.llm.base import BaseLLMClient
from chicory.llm.prompts import build_system_prompt
from chicory.llm.types import ContentBlock, LLMResponse, TextBlock, ToolUseBlock
from chicory.orchestrator.tool_definitions import CHICORY_TOOLS


def _convert_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-format tool definitions to Responses API tools."""
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
        for tool in tools
    ]


def _convert_tools_to_openai_chat(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-format tool definitions to Chat Completions tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read an attribute or dict key from OpenAI SDK objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _json_object(value: Any) -> dict[str, Any]:
    """Decode function arguments, tolerating malformed model output."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _response_text_parts(response: Any) -> list[str]:
    """Extract text output from a Responses API object or test double."""
    parts: list[str] = []
    for item in _get(response, "output", []) or []:
        item_type = _get(item, "type")
        if item_type == "message":
            for content in _get(item, "content", []) or []:
                content_type = _get(content, "type")
                if content_type in ("output_text", "text"):
                    text = _get(content, "text", "")
                    if text:
                        parts.append(text)
        elif item_type in ("output_text", "text"):
            text = _get(item, "text", "")
            if text:
                parts.append(text)

    if not parts:
        output_text = _get(response, "output_text", "")
        if output_text:
            parts.append(output_text)
    return parts


def _model_for_provider(
    config: ChicoryConfig,
    default_model: str,
    provider_model_attr: str | None = None,
    provider_prefix: str | None = None,
) -> str:
    """Resolve provider-specific model defaults while honoring explicit overrides."""
    provider_model = getattr(config, provider_model_attr, "") if provider_model_attr else ""
    if provider_prefix and config.llm_model.startswith(provider_prefix):
        return config.llm_model
    if provider_model_attr:
        return provider_model or default_model
    return config.llm_model or provider_model or default_model


def _create_openai_sdk_client(
    *,
    api_key: str,
    provider_name: str,
    base_url: str | None = None,
):
    """Create an OpenAI SDK client for OpenAI-compatible providers."""
    if not api_key:
        raise ValueError(f"{provider_name} API key is required")

    try:
        import openai
    except ImportError:
        raise ImportError(
            f"openai package is required for {provider_name} support. "
            "Install it with: pip install chicory-man[openai]"
        )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


class _ResponsesAPIClient(BaseLLMClient):
    """Base class for providers using the OpenAI-compatible Responses API."""

    _provider_name = "OpenAI"
    _default_model = DEFAULT_OPENAI_MODEL
    _provider_model_attr: str | None = None
    _api_key_attr = "openai_api_key"
    _base_url_attr: str | None = None
    _use_instructions = True

    def __init__(self, config: ChicoryConfig, active_tags: list[str] | None = None) -> None:
        super().__init__(config, active_tags)
        base_url = getattr(config, self._base_url_attr, None) if self._base_url_attr else None
        self._model = _model_for_provider(
            config,
            self._default_model,
            self._provider_model_attr,
            getattr(self, "_provider_model_prefix", None),
        )
        self._client = _create_openai_sdk_client(
            api_key=getattr(config, self._api_key_attr),
            provider_name=self._provider_name,
            base_url=base_url,
        )
        if not hasattr(self._client, "responses"):
            raise ImportError(
                "The installed openai package does not expose the Responses API. "
                "Upgrade it with: pip install -U chicory-man[openai]"
            )
        self._tools = _convert_tools_to_openai(CHICORY_TOOLS)

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages to the provider with memory tools available."""
        sys_prompt = system or build_system_prompt(self._active_tags)
        input_items = self._convert_messages(messages)
        request: dict[str, Any] = {
            "model": self._model,
            "max_output_tokens": self._config.max_tokens,
            "tools": self._tools,
        }
        if self._use_instructions:
            request["instructions"] = sys_prompt
            request["input"] = input_items
        else:
            request["input"] = [{"role": "system", "content": sys_prompt}, *input_items]

        try:
            response = self._client.responses.create(**request)
        except Exception as e:
            raise LLMError(f"{self._provider_name} API error: {e}") from e

        return self._convert_response(response)

    def propose_tags(self, content: str, existing_tags: list[str]) -> list[str]:
        """Ask OpenAI to suggest tags for a memory."""
        from chicory.llm.prompts import TAG_PROPOSAL_PROMPT

        prompt = TAG_PROPOSAL_PROMPT.format(
            content=content,
            existing_tags=", ".join(existing_tags) if existing_tags else "(none)",
        )
        try:
            response = self._client.responses.create(
                model=self._model,
                max_output_tokens=200,
                input=prompt,
            )
            text = "\n".join(_response_text_parts(response)).strip()
            tags = json.loads(text)
            if isinstance(tags, list):
                return [str(t) for t in tags]
        except (json.JSONDecodeError, IndexError, Exception):
            pass
        return []

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal message format to Responses API input items.

        Function calls are output items in the Responses API. The chat loop stores
        only provider-neutral blocks, so we reconstruct enough call/output history
        for the next model turn.
        """
        result: list[dict[str, Any]] = []

        def flush_assistant_text(text_parts: list[str]) -> None:
            if text_parts:
                result.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts),
                })
                text_parts.clear()

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "assistant" and isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        flush_assistant_text(text_parts)
                        result.append({
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        })
                flush_assistant_text(text_parts)

            elif role == "user" and isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        result.append({
                            "type": "function_call_output",
                            "call_id": item["tool_use_id"],
                            "output": item["content"],
                        })
            else:
                result.append({"role": role, "content": content})

        return result

    def _convert_response(self, response: Any) -> LLMResponse:
        """Convert an OpenAI Responses API response to our LLMResponse."""
        content: list[ContentBlock] = []

        for text in _response_text_parts(response):
            content.append(TextBlock(text=text))

        for item in _get(response, "output", []) or []:
            if _get(item, "type") == "function_call":
                content.append(ToolUseBlock(
                    id=_get(item, "call_id") or _get(item, "id", ""),
                    name=_get(item, "name", ""),
                    input=_json_object(_get(item, "arguments", "{}")),
                ))

        has_tool_use = any(block.type == "tool_use" for block in content)
        stop_reason = "tool_use" if has_tool_use else "end_turn"
        if _get(response, "status") == "incomplete":
            details = _get(response, "incomplete_details", {})
            reason = _get(details, "reason", "")
            if reason in ("max_output_tokens", "max_tokens"):
                stop_reason = "max_tokens"
            elif reason:
                stop_reason = reason

        usage: dict[str, int] = {}
        response_usage = _get(response, "usage")
        if response_usage:
            usage = {
                "input_tokens": _get(response_usage, "input_tokens", 0),
                "output_tokens": _get(response_usage, "output_tokens", 0),
            }

        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            model=_get(response, "model", "") or "",
            usage=usage,
        )


class OpenAIClient(_ResponsesAPIClient):
    """LLM client for OpenAI Responses API models."""

    _provider_name = "OpenAI"
    _default_model = DEFAULT_OPENAI_MODEL
    _api_key_attr = "openai_api_key"
    _use_instructions = True


class GrokClient(_ResponsesAPIClient):
    """LLM client for xAI Grok via the OpenAI-compatible Responses API."""

    _provider_name = "Grok"
    _default_model = DEFAULT_GROK_MODEL
    _provider_model_attr = "grok_model"
    _provider_model_prefix = "grok"
    _api_key_attr = "xai_api_key"
    _base_url_attr = "xai_base_url"
    _use_instructions = False


class GeminiClient(BaseLLMClient):
    """LLM client for Gemini via Google's OpenAI-compatible Chat Completions API."""

    def __init__(self, config: ChicoryConfig, active_tags: list[str] | None = None) -> None:
        super().__init__(config, active_tags)
        self._model = _model_for_provider(
            config,
            DEFAULT_GEMINI_MODEL,
            "gemini_model",
            "gemini",
        )
        self._client = _create_openai_sdk_client(
            api_key=config.gemini_api_key,
            provider_name="Gemini",
            base_url=config.gemini_base_url,
        )
        if not hasattr(self._client, "chat") or not hasattr(self._client.chat, "completions"):
            raise ImportError(
                "The installed openai package does not expose Chat Completions. "
                "Upgrade it with: pip install -U chicory-man[openai]"
            )
        self._tools = _convert_tools_to_openai_chat(CHICORY_TOOLS)

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages to Gemini with memory tools available."""
        sys_prompt = system or build_system_prompt(self._active_tags)
        gemini_messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
        gemini_messages.extend(self._convert_messages(messages))

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._config.max_tokens,
                messages=gemini_messages,
                tools=self._tools,
                tool_choice="auto",
            )
        except Exception as e:
            raise LLMError(f"Gemini API error: {e}") from e

        return self._convert_response(response)

    def propose_tags(self, content: str, existing_tags: list[str]) -> list[str]:
        """Ask Gemini to suggest tags for a memory."""
        from chicory.llm.prompts import TAG_PROPOSAL_PROMPT

        prompt = TAG_PROPOSAL_PROMPT.format(
            content=content,
            existing_tags=", ".join(existing_tags) if existing_tags else "(none)",
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (_get(_get(response.choices[0], "message"), "content", "") or "").strip()
            tags = json.loads(text)
            if isinstance(tags, list):
                return [str(t) for t in tags]
        except (json.JSONDecodeError, IndexError, Exception):
            pass
        return []

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal message format to OpenAI-compatible chat messages."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "assistant" and isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        })
                chat_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    chat_msg["tool_calls"] = tool_calls
                result.append(chat_msg)

            elif role == "user" and isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "tool_call_id": item["tool_use_id"],
                            "content": item["content"],
                        })
            else:
                result.append({"role": role, "content": content})

        return result

    def _convert_response(self, response: Any) -> LLMResponse:
        """Convert an OpenAI-compatible ChatCompletion to our LLMResponse."""
        choice = response.choices[0]
        message = _get(choice, "message")

        content: list[ContentBlock] = []
        message_content = _get(message, "content")
        if message_content:
            content.append(TextBlock(text=message_content))

        for tc in _get(message, "tool_calls", []) or []:
            fn = _get(tc, "function", {})
            content.append(ToolUseBlock(
                id=_get(tc, "id", ""),
                name=_get(fn, "name", ""),
                input=_json_object(_get(fn, "arguments", "{}")),
            ))

        stop_reason = _STOP_REASON_MAP.get(
            _get(choice, "finish_reason", "") or "",
            _get(choice, "finish_reason", "") or "end_turn",
        )

        usage: dict[str, int] = {}
        response_usage = _get(response, "usage")
        if response_usage:
            usage = {
                "input_tokens": _get(
                    response_usage,
                    "prompt_tokens",
                    _get(response_usage, "input_tokens", 0),
                ),
                "output_tokens": _get(
                    response_usage,
                    "completion_tokens",
                    _get(response_usage, "output_tokens", 0),
                ),
            }

        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            model=_get(response, "model", "") or "",
            usage=usage,
        )
