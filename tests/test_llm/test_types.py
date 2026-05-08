"""Tests for provider-agnostic LLM response types."""

import pytest

from chicory.llm.types import LLMResponse, TextBlock, ToolUseBlock


class TestTextBlock:
    def test_construction(self):
        block = TextBlock(text="hello")
        assert block.text == "hello"
        assert block.type == "text"

    def test_frozen(self):
        block = TextBlock(text="hello")
        with pytest.raises(AttributeError):
            block.text = "changed"


class TestToolUseBlock:
    def test_construction(self):
        block = ToolUseBlock(id="t1", name="store_memory", input={"content": "x"})
        assert block.id == "t1"
        assert block.name == "store_memory"
        assert block.input == {"content": "x"}
        assert block.type == "tool_use"

    def test_frozen(self):
        block = ToolUseBlock(id="t1", name="store_memory", input={})
        with pytest.raises(AttributeError):
            block.name = "changed"


class TestLLMResponse:
    def test_text_only(self):
        resp = LLMResponse(
            content=[TextBlock(text="hi")],
            stop_reason="end_turn",
        )
        assert len(resp.content) == 1
        assert resp.content[0].text == "hi"
        assert resp.stop_reason == "end_turn"
        assert resp.model == ""
        assert resp.usage == {}

    def test_tool_use(self):
        resp = LLMResponse(
            content=[
                TextBlock(text="Let me store that."),
                ToolUseBlock(id="t1", name="store_memory", input={"content": "x", "tags": ["test"]}),
            ],
            stop_reason="tool_use",
            model="test-model",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert len(resp.content) == 2
        assert resp.content[0].type == "text"
        assert resp.content[1].type == "tool_use"
        assert resp.content[1].name == "store_memory"
        assert resp.stop_reason == "tool_use"

    def test_frozen(self):
        resp = LLMResponse(content=[], stop_reason="end_turn")
        with pytest.raises(AttributeError):
            resp.stop_reason = "changed"

    def test_content_iteration(self):
        """Verify the pattern ChatSession uses to iterate content blocks."""
        resp = LLMResponse(
            content=[
                TextBlock(text="thinking..."),
                ToolUseBlock(id="t1", name="retrieve_memories", input={"query": "test"}),
            ],
            stop_reason="tool_use",
        )
        has_tool_use = any(block.type == "tool_use" for block in resp.content)
        assert has_tool_use

        for block in resp.content:
            if block.type == "tool_use":
                assert block.name == "retrieve_memories"
                assert block.input == {"query": "test"}
                assert block.id == "t1"
            elif hasattr(block, "text"):
                assert block.text == "thinking..."
