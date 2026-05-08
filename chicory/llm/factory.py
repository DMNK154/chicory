"""LLM client factory."""

from __future__ import annotations

from chicory.config import ChicoryConfig
from chicory.llm.base import BaseLLMClient


def create_llm_client(config: ChicoryConfig) -> BaseLLMClient:
    """Create an LLM client based on config.

    Provider selection:
    1. Explicit llm_provider setting overrides auto-detection
    2. "auto" (default): picks based on which API key is set
    3. Falls back to NullClient if no key is configured
    """
    provider = config.llm_provider.lower()

    if provider == "anthropic" or (provider == "auto" and config.anthropic_api_key):
        from chicory.llm.client import ClaudeClient
        return ClaudeClient(config)

    if provider == "openai" or (provider == "auto" and config.openai_api_key):
        from chicory.llm.openai_client import OpenAIClient
        return OpenAIClient(config)

    if provider in ("grok", "xai") or (provider == "auto" and config.xai_api_key):
        from chicory.llm.openai_client import GrokClient
        return GrokClient(config)

    if provider in ("gemini", "google") or (provider == "auto" and config.gemini_api_key):
        from chicory.llm.openai_client import GeminiClient
        return GeminiClient(config)

    from chicory.llm.null_client import NullClient
    return NullClient(config)
