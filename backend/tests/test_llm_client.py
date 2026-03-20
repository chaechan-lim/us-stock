"""LLM client + provider tests."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.llm.client import LLMClient, LLMResponse, ToolCall
from services.llm.providers import AnthropicProvider

# ── Mock helpers ──────────────────────────────────────────────


@dataclass
class MockLLMConfig:
    enabled: bool = True
    api_key: str = "test-anthropic-key"
    model: str = "claude-haiku-4-5-20251001"
    fallback_model: str = "claude-sonnet-4-6"
    gemini_api_key: str = ""
    gemini_fallback_model: str = "gemini-2.5-flash"
    max_tokens: int = 4096
    cooldown_seconds: int = 300


def _make_anthropic_response(text="Hello", stop_reason="end_turn", tool_blocks=None):
    """Create a mock Anthropic API response."""
    content = []
    if text:
        block = MagicMock()
        block.type = "text"
        block.text = text
        content.append(block)
    if tool_blocks:
        for tb in tool_blocks:
            block = MagicMock()
            block.type = "tool_use"
            block.id = tb["id"]
            block.name = tb["name"]
            block.input = tb["input"]
            content.append(block)

    resp = MagicMock()
    resp.content = content
    resp.stop_reason = stop_reason
    return resp


def _make_client_with_mock_provider(config=None):
    """Create LLMClient with mocked AnthropicProvider (no real import)."""
    config = config or MockLLMConfig()
    with patch.object(AnthropicProvider, "__init__", return_value=None):
        client = LLMClient(config)
    # Replace with a fully controlled mock provider
    mock_provider = MagicMock()
    client._anthropic = mock_provider
    return client, mock_provider


# ── LLMClient tests ──────────────────────────────────────────


class TestLLMClientInit:
    def test_init_anthropic_only(self):
        client, _ = _make_client_with_mock_provider()
        assert client._anthropic is not None
        assert client._gemini is None

    def test_init_no_keys(self):
        config = MockLLMConfig(api_key="")
        client = LLMClient(config)
        assert client._anthropic is None
        assert client._gemini is None

    def test_init_with_gemini(self):
        config = MockLLMConfig(gemini_api_key="test-key")
        with patch.object(AnthropicProvider, "__init__", return_value=None):
            from services.llm.providers import GeminiProvider

            with patch.object(GeminiProvider, "__init__", return_value=None):
                client = LLMClient(config)
        assert client._anthropic is not None
        assert client._gemini is not None


class TestLLMClientFallbackChain:
    def test_chain_anthropic_only(self):
        client, _ = _make_client_with_mock_provider()
        chain = client._build_fallback_chain()
        assert len(chain) == 2  # haiku + sonnet
        assert chain[0][0] == "claude-haiku-4-5-20251001"
        assert chain[1][0] == "claude-sonnet-4-6"

    def test_chain_with_gemini(self):
        config = MockLLMConfig(gemini_api_key="test-key")
        client, _ = _make_client_with_mock_provider(config)
        client._gemini = MagicMock()
        chain = client._build_fallback_chain()
        assert len(chain) == 3
        # Cost-aware order: Haiku → Gemini (free) → Sonnet (expensive)
        assert chain[1][0] == "gemini-2.5-flash"
        assert chain[2][0] == "claude-sonnet-4-6"

    def test_chain_with_model_override(self):
        client, _ = _make_client_with_mock_provider()
        chain = client._build_fallback_chain(model_override="claude-sonnet-4-6")
        models = [m for m, _ in chain]
        assert models[0] == "claude-sonnet-4-6"
        # Haiku should be a fallback when override is set
        assert "claude-haiku-4-5-20251001" in models
        # No duplicates
        assert len(models) == len(set(models))

    def test_chain_with_model_override_deduplicates(self):
        """Override same as primary should not create duplicates."""
        client, _ = _make_client_with_mock_provider()
        chain = client._build_fallback_chain(
            model_override="claude-haiku-4-5-20251001",
        )
        models = [m for m, _ in chain]
        assert models.count("claude-haiku-4-5-20251001") == 1

    def test_chain_no_fallback(self):
        config = MockLLMConfig(fallback_model="")
        client, _ = _make_client_with_mock_provider(config)
        chain = client._build_fallback_chain()
        assert len(chain) == 1


class TestLLMClientGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        client, provider = _make_client_with_mock_provider()
        mock_response = LLMResponse(text="Analysis complete", model="claude-haiku-4-5-20251001")
        provider.create = AsyncMock(return_value=mock_response)

        result = await client.generate(
            messages=[{"role": "user", "content": "test"}],
        )
        assert result.text == "Analysis complete"
        provider.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_fallback_on_failure(self):
        client, provider = _make_client_with_mock_provider()

        mock_response = LLMResponse(text="fallback response", model="claude-sonnet-4-6")

        async def mock_create(**kwargs):
            if kwargs["model"] == "claude-haiku-4-5-20251001":
                raise Exception("API error")
            return mock_response

        provider.create = mock_create

        result = await client.generate(
            messages=[{"role": "user", "content": "test"}],
            retries=1,
        )
        assert result.text == "fallback response"

    @pytest.mark.asyncio
    async def test_generate_all_fail_raises(self):
        client, provider = _make_client_with_mock_provider()
        provider.create = AsyncMock(side_effect=Exception("fail"))

        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await client.generate(
                messages=[{"role": "user", "content": "test"}],
                retries=1,
            )

    @pytest.mark.asyncio
    async def test_generate_no_providers_raises(self):
        config = MockLLMConfig(api_key="", fallback_model="")
        client = LLMClient(config)

        with pytest.raises(RuntimeError, match="No LLM providers"):
            await client.generate(
                messages=[{"role": "user", "content": "test"}],
            )

    @pytest.mark.asyncio
    async def test_generate_cross_provider_fallback(self):
        """Anthropic fail -> Gemini fallback."""
        config = MockLLMConfig(gemini_api_key="test-key")
        client, anthropic_provider = _make_client_with_mock_provider(config)

        # Anthropic fails
        anthropic_provider.create = AsyncMock(side_effect=Exception("Anthropic down"))

        # Gemini succeeds
        gemini_response = LLMResponse(text="Gemini response", model="gemini-2.5-flash")
        gemini_provider = MagicMock()
        gemini_provider.create = AsyncMock(return_value=gemini_response)
        client._gemini = gemini_provider

        result = await client.generate(
            messages=[{"role": "user", "content": "test"}],
            retries=1,
        )
        assert result.text == "Gemini response"
        assert result.model == "gemini-2.5-flash"


class TestLLMClientToolUse:
    @pytest.mark.asyncio
    async def test_tool_use_response(self):
        client, provider = _make_client_with_mock_provider()

        tool_response = LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="tc_1", name="get_status", arguments={"symbol": "AAPL"})],
            stop_reason="tool_use",
            model="claude-haiku-4-5-20251001",
        )
        provider.create = AsyncMock(return_value=tool_response)

        result = await client.generate_with_tools(
            messages=[{"role": "user", "content": "check AAPL"}],
            tools=[{"name": "get_status", "description": "status", "input_schema": {}}],
        )
        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_status"


class TestFormatToolLoopMessages:
    def test_format_delegates_to_provider(self):
        client, provider = _make_client_with_mock_provider()

        response = LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="tc_1", name="test_tool", arguments={})],
            stop_reason="tool_use",
            model="claude-haiku-4-5-20251001",
        )

        provider.format_tool_loop_messages = MagicMock(
            return_value=({"role": "assistant"}, {"role": "user"})
        )

        asst, user = client.format_tool_loop_messages(
            response,
            [{"tool_call_id": "tc_1", "content": "result"}],
        )
        assert asst["role"] == "assistant"
        assert user["role"] == "user"
        provider.format_tool_loop_messages.assert_called_once()


# ── AnthropicProvider tests ───────────────────────────────────


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_create_text_response(self):
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            raw = _make_anthropic_response("test response", "end_turn")
            mock_client.messages.create = AsyncMock(return_value=raw)

            provider = AnthropicProvider(api_key="test")
            result = await provider.create(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                system=None,
                tools=None,
            )

        assert result.text == "test response"
        assert result.stop_reason == "end_turn"
        assert len(result.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_create_tool_use_response(self):
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            raw = _make_anthropic_response(
                text=None,
                stop_reason="tool_use",
                tool_blocks=[{"id": "tc_1", "name": "get_status", "input": {"ex": "a"}}],
            )
            mock_client.messages.create = AsyncMock(return_value=raw)

            provider = AnthropicProvider(api_key="test")
            result = await provider.create(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                system="test system",
                tools=[{"name": "get_status"}],
            )

        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_status"
        assert result.tool_calls[0].arguments == {"ex": "a"}

    def test_format_tool_loop_messages(self):
        with patch("anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="test")

        raw = _make_anthropic_response(
            text="thinking...",
            stop_reason="tool_use",
            tool_blocks=[{"id": "tc_1", "name": "get_status", "input": {}}],
        )

        response = LLMResponse(
            text="thinking...",
            tool_calls=[ToolCall(id="tc_1", name="get_status", arguments={})],
            stop_reason="tool_use",
            model="test",
            raw=raw,
        )

        asst_msg, user_msg = provider.format_tool_loop_messages(
            response,
            [{"tool_call_id": "tc_1", "content": '{"status": "ok"}'}],
        )

        assert asst_msg["role"] == "assistant"
        assert len(asst_msg["content"]) == 2  # text + tool_use
        assert asst_msg["content"][0]["type"] == "text"
        assert asst_msg["content"][1]["type"] == "tool_use"

        assert user_msg["role"] == "user"
        assert user_msg["content"][0]["type"] == "tool_result"
        assert user_msg["content"][0]["tool_use_id"] == "tc_1"
