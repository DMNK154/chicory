"""Tests for LLM client factory and provider selection."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from chicory.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROK_MODEL,
    ChicoryConfig,
    load_config,
)
from chicory.llm.base import BaseLLMClient
from chicory.llm.factory import create_llm_client
from chicory.llm.null_client import NullClient
from chicory.llm.types import LLMResponse, TextBlock, ToolUseBlock


def _install_fake_openai(monkeypatch):
    class FakeResponses:
        def __init__(self):
            self.last_request = None

        def create(self, **kwargs):
            self.last_request = kwargs
            return SimpleNamespace(
                output=[SimpleNamespace(type="message", content=[
                    SimpleNamespace(type="output_text", text="ok"),
                ])],
                output_text="ok",
                status="completed",
                model=kwargs["model"],
                usage=None,
            )

    class FakeChatCompletions:
        def __init__(self):
            self.last_request = None

        def create(self, **kwargs):
            self.last_request = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=[]),
                    finish_reason="stop",
                )],
                model=kwargs["model"],
                usage=None,
            )

    class FakeOpenAIClient:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.responses = FakeResponses()
            self.chat = SimpleNamespace(completions=FakeChatCompletions())
            self.__class__.instances.append(self)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))
    return FakeOpenAIClient


class TestNullClient:
    def test_chat_returns_fixed_message(self):
        config = ChicoryConfig()
        client = NullClient(config)
        resp = client.chat([{"role": "user", "content": "hello"}])
        assert isinstance(resp, LLMResponse)
        assert resp.stop_reason == "end_turn"
        assert len(resp.content) == 1
        assert resp.content[0].type == "text"
        assert "No LLM configured" in resp.content[0].text

    def test_update_active_tags(self):
        config = ChicoryConfig()
        client = NullClient(config)
        client.update_active_tags(["test", "memory"])
        assert client._active_tags == ["test", "memory"]


class TestBaseLLMClient:
    def test_chat_raises_not_implemented(self):
        config = ChicoryConfig()
        client = BaseLLMClient(config)
        with pytest.raises(NotImplementedError):
            client.chat([])

    def test_close_is_noop(self):
        config = ChicoryConfig()
        client = BaseLLMClient(config)
        client.close()  # Should not raise


class TestFactory:
    def test_auto_no_keys_returns_null(self):
        config = ChicoryConfig(llm_provider="auto")
        client = create_llm_client(config)
        assert isinstance(client, NullClient)

    def test_explicit_null_provider(self):
        config = ChicoryConfig(llm_provider="null")
        client = create_llm_client(config)
        assert isinstance(client, NullClient)

    def test_auto_with_anthropic_key(self):
        config = ChicoryConfig(anthropic_api_key="test-key")
        client = create_llm_client(config)
        # Should attempt to create ClaudeClient (will have the key set)
        from chicory.llm.client import ClaudeClient
        assert isinstance(client, ClaudeClient)

    def test_explicit_anthropic_provider(self):
        config = ChicoryConfig(llm_provider="anthropic", anthropic_api_key="test-key")
        client = create_llm_client(config)
        from chicory.llm.client import ClaudeClient
        assert isinstance(client, ClaudeClient)

    def test_explicit_openai_no_package_raises(self):
        """If openai package isn't installed, should get ImportError."""
        # This test only makes sense if openai isn't installed.
        # If it is installed, we just verify the factory returns an OpenAIClient.
        config = ChicoryConfig(llm_provider="openai", openai_api_key="test-key")
        try:
            client = create_llm_client(config)
            from chicory.llm.openai_client import OpenAIClient
            assert isinstance(client, OpenAIClient)
        except ImportError:
            pass  # Expected if openai not installed

    def test_auto_with_xai_key(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        config = ChicoryConfig(llm_provider="auto", xai_api_key="test-key")
        client = create_llm_client(config)
        from chicory.llm.openai_client import GrokClient
        assert isinstance(client, GrokClient)

    def test_auto_with_gemini_key(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        config = ChicoryConfig(llm_provider="auto", gemini_api_key="test-key")
        client = create_llm_client(config)
        from chicory.llm.openai_client import GeminiClient
        assert isinstance(client, GeminiClient)

    def test_explicit_grok_provider_uses_xai_base_url_and_default_model(self, monkeypatch):
        fake_openai = _install_fake_openai(monkeypatch)
        config = ChicoryConfig(llm_provider="grok", xai_api_key="test-key")
        client = create_llm_client(config)
        from chicory.llm.openai_client import GrokClient

        assert isinstance(client, GrokClient)
        assert client._model == DEFAULT_GROK_MODEL
        assert fake_openai.instances[-1].kwargs["base_url"] == "https://api.x.ai/v1"

    def test_explicit_gemini_provider_uses_google_base_url_and_default_model(self, monkeypatch):
        fake_openai = _install_fake_openai(monkeypatch)
        config = ChicoryConfig(llm_provider="gemini", gemini_api_key="test-key")
        client = create_llm_client(config)
        from chicory.llm.openai_client import GeminiClient

        assert isinstance(client, GeminiClient)
        assert client._model == DEFAULT_GEMINI_MODEL
        assert fake_openai.instances[-1].kwargs["base_url"] == (
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def test_explicit_gemini_provider_requires_key(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        config = ChicoryConfig(llm_provider="gemini")
        with pytest.raises(ValueError, match="Gemini API key is required"):
            create_llm_client(config)

    def test_load_config_swaps_generic_model_for_provider_default(self, monkeypatch):
        monkeypatch.setenv("CHICORY_LLM_PROVIDER", "grok")
        monkeypatch.setenv("CHICORY_LLM_MODEL", "claude-sonnet-4-6")
        config = load_config(db_path=Path(":memory:"), commons_enabled=False)
        assert config.llm_model == DEFAULT_GROK_MODEL


class TestClaudeClientConversion:
    """Test ClaudeClient message conversion without making API calls."""

    def test_convert_messages_with_content_blocks(self):
        from chicory.llm.client import ClaudeClient

        config = ChicoryConfig(anthropic_api_key="test-key")
        client = ClaudeClient(config)

        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="Let me check."),
                    ToolUseBlock(id="t1", name="retrieve_memories", input={"query": "hello"}),
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": '{"memories": []}'},
                ],
            },
        ]

        converted = client._convert_messages(messages)

        # First message unchanged
        assert converted[0] == {"role": "user", "content": "hello"}

        # Assistant message: ContentBlock dataclasses → dicts
        assert converted[1]["role"] == "assistant"
        assert converted[1]["content"][0] == {"type": "text", "text": "Let me check."}
        assert converted[1]["content"][1] == {
            "type": "tool_use",
            "id": "t1",
            "name": "retrieve_memories",
            "input": {"query": "hello"},
        }

        # Tool result message passes through as-is (already dicts)
        assert converted[2]["role"] == "user"
        assert converted[2]["content"][0]["type"] == "tool_result"

    def test_convert_messages_plain_strings(self):
        from chicory.llm.client import ClaudeClient

        config = ChicoryConfig(anthropic_api_key="test-key")
        client = ClaudeClient(config)

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        converted = client._convert_messages(messages)
        assert converted == messages  # Plain strings pass through unchanged


class TestOpenAIToolConversion:
    def test_convert_tools_to_openai(self):
        from chicory.llm.openai_client import _convert_tools_to_openai

        anthropic_tools = [
            {
                "name": "store_memory",
                "description": "Store a memory",
                "input_schema": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            }
        ]

        openai_tools = _convert_tools_to_openai(anthropic_tools)

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["name"] == "store_memory"
        assert openai_tools[0]["description"] == "Store a memory"
        assert openai_tools[0]["parameters"] == anthropic_tools[0]["input_schema"]

    def test_convert_tools_to_openai_chat(self):
        from chicory.llm.openai_client import _convert_tools_to_openai_chat

        anthropic_tools = [
            {
                "name": "store_memory",
                "description": "Store a memory",
                "input_schema": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            }
        ]

        chat_tools = _convert_tools_to_openai_chat(anthropic_tools)

        assert len(chat_tools) == 1
        assert chat_tools[0]["type"] == "function"
        assert chat_tools[0]["function"]["name"] == "store_memory"
        assert chat_tools[0]["function"]["parameters"] == anthropic_tools[0]["input_schema"]


class TestOpenAIResponsesConversion:
    """Test OpenAI Responses API conversion without making API calls."""

    def test_convert_messages_with_tool_round_trip(self):
        from chicory.llm.openai_client import OpenAIClient

        client = object.__new__(OpenAIClient)
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="Let me check."),
                    ToolUseBlock(id="call_1", name="retrieve_memories", input={"query": "hello"}),
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": '{"memories": []}'},
                ],
            },
        ]

        converted = client._convert_messages(messages)

        assert converted == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Let me check."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "retrieve_memories",
                "arguments": '{"query": "hello"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '{"memories": []}',
            },
        ]

    def test_convert_response_with_text_tool_and_usage(self):
        from chicory.llm.openai_client import OpenAIClient

        client = object.__new__(OpenAIClient)
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="I found something."),
                    ],
                ),
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="retrieve_memories",
                    arguments='{"query": "hello"}',
                ),
            ],
            status="completed",
            model="gpt-5.5",
            usage=SimpleNamespace(input_tokens=12, output_tokens=8),
        )

        converted = client._convert_response(response)

        assert converted.stop_reason == "tool_use"
        assert converted.model == "gpt-5.5"
        assert converted.usage == {"input_tokens": 12, "output_tokens": 8}
        assert converted.content == [
            TextBlock(text="I found something."),
            ToolUseBlock(id="call_1", name="retrieve_memories", input={"query": "hello"}),
        ]


class TestGrokClient:
    def test_grok_puts_system_prompt_in_input_not_instructions(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        from chicory.llm.openai_client import GrokClient

        client = GrokClient(ChicoryConfig(llm_provider="grok", xai_api_key="test-key"))
        client.chat([{"role": "user", "content": "hello"}], system="system prompt")

        request = client._client.responses.last_request
        assert "instructions" not in request
        assert request["model"] == DEFAULT_GROK_MODEL
        assert request["input"][0] == {"role": "system", "content": "system prompt"}
        assert request["input"][1] == {"role": "user", "content": "hello"}


class TestGeminiClientConversion:
    """Test Gemini OpenAI-compatible conversion without making API calls."""

    def test_convert_messages_with_tool_round_trip(self):
        from chicory.llm.openai_client import GeminiClient

        client = object.__new__(GeminiClient)
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="Let me check."),
                    ToolUseBlock(id="call_1", name="retrieve_memories", input={"query": "hello"}),
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": '{"memories": []}'},
                ],
            },
        ]

        converted = client._convert_messages(messages)

        assert converted == [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "retrieve_memories",
                            "arguments": '{"query": "hello"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"memories": []}',
            },
        ]

    def test_convert_response_with_text_tool_and_usage(self):
        from chicory.llm.openai_client import GeminiClient

        client = object.__new__(GeminiClient)
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="I found something.",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="retrieve_memories",
                                    arguments='{"query": "hello"}',
                                ),
                            ),
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            model="gemini-2.5-flash",
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
        )

        converted = client._convert_response(response)

        assert converted.stop_reason == "tool_use"
        assert converted.model == "gemini-2.5-flash"
        assert converted.usage == {"input_tokens": 12, "output_tokens": 8}
        assert converted.content == [
            TextBlock(text="I found something."),
            ToolUseBlock(id="call_1", name="retrieve_memories", input={"query": "hello"}),
        ]
