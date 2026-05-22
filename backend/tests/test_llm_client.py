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


# ── GeminiProvider tests ─────────────────────────────────────────
#
# google.genai is a heavy SDK; we patch the symbols GeminiProvider
# touches lazily (`from google import genai` / `from google.genai
# import types`). Each test wires the smallest possible stub.

def _patch_genai(stub_types):
    """Return a context manager that patches sys.modules so
    `from google.genai import types` returns our stub."""
    fake_genai = MagicMock()
    fake_genai.types = stub_types
    return patch.dict("sys.modules", {
        "google": MagicMock(genai=fake_genai),
        "google.genai": fake_genai,
        "google.genai.types": stub_types,
    })


def _stub_genai_types():
    """Return a MagicMock that mimics google.genai.types just enough."""
    types = MagicMock()
    # Each types.Foo(...) call returns a recognizable mock with the
    # arguments stored so tests can assert against them later.
    def _make(name):
        def _ctor(**kw):
            obj = MagicMock(name=name)
            for k, v in kw.items():
                setattr(obj, k, v)
            return obj
        m = MagicMock(side_effect=_ctor)
        m.__name__ = name
        return m
    types.Tool = _make("Tool")
    types.FunctionDeclaration = _make("FunctionDeclaration")
    types.GenerateContentConfig = _make("GenerateContentConfig")
    types.Content = _make("Content")
    # Part has from_text / from_function_call / from_function_response classmethods
    types.Part = MagicMock()
    types.Part.from_text = MagicMock(side_effect=lambda text: ("text", text))
    types.Part.from_function_call = MagicMock(side_effect=lambda name, args: ("fc", name, args))
    types.Part.from_function_response = MagicMock(
        side_effect=lambda name, response: ("fr", name, response),
    )
    return types


def _make_gemini_response(text=None, function_calls=None, finish_reason=None):
    """Build a stub response object shaped like Gemini's."""
    candidate = MagicMock()
    content = MagicMock()
    parts = []
    if text:
        p = MagicMock()
        p.text = text
        p.function_call = None
        parts.append(p)
    for fc in (function_calls or []):
        p = MagicMock()
        p.text = None
        fc_mock = MagicMock()
        fc_mock.name = fc["name"]
        fc_mock.args = fc.get("args") or {}
        p.function_call = fc_mock
        parts.append(p)
    content.parts = parts
    candidate.content = content
    candidate.finish_reason = finish_reason
    resp = MagicMock()
    resp.candidates = [candidate]
    return resp


class TestGeminiProvider:
    def _provider(self):
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
            from services.llm.providers import GeminiProvider
            with patch.object(GeminiProvider, "__init__", return_value=None):
                provider = GeminiProvider.__new__(GeminiProvider)
            provider._client = MagicMock()
            provider._client.aio = MagicMock()
            provider._client.aio.models = MagicMock()
            provider._client.aio.models.generate_content = AsyncMock()
            provider._genai = MagicMock()
        return provider

    def test_convert_tools(self):
        provider = self._provider()
        stub_types = _stub_genai_types()
        # `from google.genai import types` resolves via the parent module's attr,
        # so we need to patch both `google.genai.types` (sys.modules) and the
        # `.types` attribute on a fake `google.genai`.
        fake_genai = MagicMock()
        fake_genai.types = stub_types
        with patch.dict("sys.modules", {
            "google": MagicMock(genai=fake_genai),
            "google.genai": fake_genai,
            "google.genai.types": stub_types,
        }):
            tools = provider._convert_tools([
                {"name": "get_status", "description": "fetch", "input_schema": {"properties": {"a": {"type": "string"}}}},
                {"name": "bare", "input_schema": {}},
            ])
        assert len(tools) == 1
        stub_types.FunctionDeclaration.assert_called()
        # Empty properties → parameters=None branch
        call_kwargs = stub_types.FunctionDeclaration.call_args_list[1].kwargs
        assert call_kwargs["parameters"] is None

    def test_convert_messages_string_content(self):
        provider = self._provider()
        stub_types = _stub_genai_types()
        with _patch_genai(stub_types):
            out = provider._convert_messages([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ])
        assert len(out) == 2
        # role mapped: assistant → model
        stub_types.Content.assert_any_call(role="user", parts=[("text", "hi")])
        stub_types.Content.assert_any_call(role="model", parts=[("text", "hello")])

    def test_convert_messages_list_content_all_block_types(self):
        provider = self._provider()
        stub_types = _stub_genai_types()
        with _patch_genai(stub_types):
            out = provider._convert_messages([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {"type": "tool_use", "name": "f", "input": {"x": 1}},
                        {"type": "tool_result", "_tool_name": "f", "content": "ok"},
                        "bare-string",
                    ],
                },
            ])
        assert len(out) == 1

    def test_convert_messages_empty_parts_skipped(self):
        provider = self._provider()
        stub_types = _stub_genai_types()
        with _patch_genai(stub_types):
            # Empty list content → no parts → message skipped
            out = provider._convert_messages([{"role": "user", "content": []}])
        assert out == []

    @pytest.mark.asyncio
    async def test_create_text_response(self):
        provider = self._provider()
        stub_types = _stub_genai_types()
        resp = _make_gemini_response(text="hello world")
        provider._client.aio.models.generate_content = AsyncMock(return_value=resp)
        with _patch_genai(stub_types):
            out = await provider.create(
                messages=[{"role": "user", "content": "hi"}],
                model="gemini-x",
                max_tokens=128,
                system="be brief",
                tools=None,
            )
        assert out.text == "hello world"
        assert out.stop_reason == "end_turn"
        assert out.tool_calls == []
        assert out.model == "gemini-x"

    @pytest.mark.asyncio
    async def test_create_tool_use_response(self):
        provider = self._provider()
        stub_types = _stub_genai_types()
        resp = _make_gemini_response(function_calls=[
            {"name": "get_status", "args": {"sym": "AAPL"}},
        ])
        provider._client.aio.models.generate_content = AsyncMock(return_value=resp)
        with _patch_genai(stub_types):
            out = await provider.create(
                messages=[{"role": "user", "content": "go"}],
                model="gemini-x",
                max_tokens=128,
                system=None,
                tools=[{"name": "get_status", "input_schema": {"properties": {}}}],
            )
        assert out.stop_reason == "tool_use"
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "get_status"
        assert out.tool_calls[0].arguments == {"sym": "AAPL"}

    @pytest.mark.asyncio
    async def test_create_max_tokens_stop(self):
        provider = self._provider()
        stub_types = _stub_genai_types()
        resp = _make_gemini_response(text="cut", finish_reason="MAX_TOKENS")
        provider._client.aio.models.generate_content = AsyncMock(return_value=resp)
        with _patch_genai(stub_types):
            out = await provider.create(
                messages=[{"role": "user", "content": "long"}],
                model="gemini-x", max_tokens=10, system=None, tools=None,
            )
        assert out.stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_create_empty_candidates(self):
        provider = self._provider()
        stub_types = _stub_genai_types()
        empty = MagicMock()
        empty.candidates = []
        provider._client.aio.models.generate_content = AsyncMock(return_value=empty)
        with _patch_genai(stub_types):
            out = await provider.create(
                messages=[{"role": "user", "content": "x"}],
                model="gemini-x", max_tokens=10, system=None, tools=None,
            )
        assert out.text is None
        assert out.tool_calls == []
        assert out.stop_reason == "end_turn"

    def test_format_tool_loop_messages(self):
        provider = self._provider()
        response = LLMResponse(
            text="thinking",
            tool_calls=[ToolCall(id="x1", name="get_status", arguments={"y": 1})],
            stop_reason="tool_use",
            model="gemini-x",
            raw=None,
        )
        asst, user = provider.format_tool_loop_messages(
            response,
            [{"tool_call_id": "x1", "content": '{"ok": true}'}],
        )
        assert asst["role"] == "assistant"
        assert any(p["type"] == "text" for p in asst["content"])
        assert any(p["type"] == "tool_use" for p in asst["content"])
        assert user["role"] == "user"
        assert user["content"][0]["type"] == "tool_result"
        assert user["content"][0]["_tool_name"] == "get_status"

    def test_format_tool_loop_messages_no_text(self):
        """response.text=None → no text part on assistant message."""
        provider = self._provider()
        response = LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="x2", name="f", arguments={})],
            stop_reason="tool_use",
            model="gemini-x",
            raw=None,
        )
        asst, _ = provider.format_tool_loop_messages(
            response,
            [{"tool_call_id": "x2", "content": ""}],
        )
        assert all(p["type"] != "text" for p in asst["content"])
