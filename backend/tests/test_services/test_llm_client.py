"""Tests for LLM client: daily budget, fallback chain, rate-limit handling, cooldown."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.llm.client import LLMClient, _is_rate_limit_error
from services.llm.providers import LLMResponse


@pytest.fixture
def llm_config():
    cfg = MagicMock()
    cfg.api_key = "test-key"
    cfg.model = "claude-haiku-4-5-20251001"
    cfg.fallback_model = "claude-sonnet-4-6"
    cfg.gemini_api_key = "gemini-key"
    cfg.gemini_fallback_model = "gemini-3-flash-preview"
    cfg.max_daily_calls = 10
    return cfg


@pytest.fixture
def mock_response():
    return LLMResponse(
        text="test response",
        model="claude-haiku-4-5-20251001",
    )


class TestDailyBudget:
    def test_budget_check_passes_when_under_limit(self, llm_config):
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            assert client._check_budget() is True

    def test_budget_check_fails_when_at_limit(self, llm_config):
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            client._daily_calls = 10
            client._daily_reset_date = time.strftime("%Y-%m-%d")
            assert client._check_budget() is False

    def test_budget_resets_on_new_day(self, llm_config):
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            client._daily_calls = 10
            client._daily_reset_date = "2020-01-01"  # old date
            assert client._check_budget() is True
            assert client._daily_calls == 0

    def test_unlimited_when_max_is_zero(self, llm_config):
        llm_config.max_daily_calls = 0
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            client._daily_calls = 999
            assert client._check_budget() is True

    def test_remaining_calls(self, llm_config):
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            assert client.daily_calls_remaining == 10
            client._increment_calls()
            client._increment_calls()
            assert client.daily_calls_remaining == 8

    def test_remaining_unlimited(self, llm_config):
        llm_config.max_daily_calls = 0
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            assert client.daily_calls_remaining == -1

    @pytest.mark.asyncio
    async def test_generate_rejected_when_budget_exhausted(self, llm_config):
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            client._daily_calls = 10
            client._daily_reset_date = time.strftime("%Y-%m-%d")

            with pytest.raises(RuntimeError, match="budget exhausted"):
                await client.generate(
                    messages=[{"role": "user", "content": "test"}],
                )

    @pytest.mark.asyncio
    async def test_generate_increments_counter(self, llm_config, mock_response):
        with (
            patch("services.llm.client.AnthropicProvider") as MockAnthropic,
            patch("services.llm.client.GeminiProvider"),
        ):
            mock_provider = MockAnthropic.return_value
            mock_provider.create = AsyncMock(return_value=mock_response)

            client = LLMClient(llm_config)
            assert client._daily_calls == 0

            await client.generate(
                messages=[{"role": "user", "content": "test"}],
            )
            assert client._daily_calls == 1


class TestFallbackChain:
    def test_chain_order_is_haiku_gemini_sonnet(self, llm_config):
        """Gemini (free) should come before Sonnet (expensive)."""
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            chain = client._build_fallback_chain()

            models = [model for model, _ in chain]
            assert models[0] == "claude-haiku-4-5-20251001"
            assert models[1] == "gemini-3-flash-preview"
            assert models[2] == "claude-sonnet-4-6"

    def test_model_override_gemini_includes_haiku_fallback(self, llm_config):
        """When Gemini is model_override, Haiku should be fallback (not duplicate Gemini)."""
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            chain = client._build_fallback_chain(model_override="gemini-3-flash-preview")

            models = [model for model, _ in chain]
            # Gemini primary → Haiku fallback → Sonnet last resort
            assert models[0] == "gemini-3-flash-preview"
            assert models[1] == "claude-haiku-4-5-20251001"
            assert models[2] == "claude-sonnet-4-6"
            # No duplicates
            assert len(models) == len(set(models))

    def test_model_override_no_duplicate_entries(self, llm_config):
        """Deduplication: same model should not appear twice in chain."""
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            chain = client._build_fallback_chain(model_override="gemini-3-flash-preview")

            models = [model for model, _ in chain]
            assert len(models) == len(set(models)), f"Duplicate models in chain: {models}"

    def test_model_override_haiku_deduplicates(self, llm_config):
        """Override with Haiku should not duplicate with default primary."""
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            chain = client._build_fallback_chain(
                model_override="claude-haiku-4-5-20251001",
            )

            models = [model for model, _ in chain]
            assert models.count("claude-haiku-4-5-20251001") == 1

    def test_chain_without_gemini(self, llm_config):
        llm_config.gemini_api_key = ""
        llm_config.gemini_fallback_model = ""
        with patch("services.llm.client.AnthropicProvider"):
            client = LLMClient(llm_config)
            chain = client._build_fallback_chain()

            models = [model for model, _ in chain]
            assert len(models) == 2
            assert models[0] == "claude-haiku-4-5-20251001"
            assert models[1] == "claude-sonnet-4-6"


class TestRateLimitDetection:
    """Tests for _is_rate_limit_error() helper."""

    def test_detects_429_status(self):
        assert _is_rate_limit_error(Exception("HTTP 429 Too Many Requests")) is True

    def test_detects_resource_exhausted(self):
        assert _is_rate_limit_error(Exception("RESOURCE_EXHAUSTED: free tier limit")) is True

    def test_detects_rate_limit_message(self):
        assert _is_rate_limit_error(Exception("Rate limit exceeded")) is True

    def test_detects_quota_message(self):
        assert (
            _is_rate_limit_error(Exception("Quota exceeded for model gemini-3-flash-preview"))
            is True
        )

    def test_detects_too_many_requests(self):
        assert _is_rate_limit_error(Exception("Too many requests")) is True

    def test_ignores_regular_errors(self):
        assert _is_rate_limit_error(Exception("Internal server error")) is False

    def test_ignores_auth_errors(self):
        assert _is_rate_limit_error(Exception("401 Unauthorized")) is False

    def test_ignores_connection_errors(self):
        assert _is_rate_limit_error(Exception("Connection refused")) is False


class TestProviderCooldown:
    """Tests for per-provider cooldown after quota-exceeded errors."""

    def test_no_cooldown_by_default(self, llm_config):
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            assert client._is_provider_cooled_down("gemini-3-flash-preview") is False

    def test_cooldown_set_marks_provider_unavailable(self, llm_config):
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            client._set_provider_cooldown("gemini-3-flash-preview", seconds=3600)
            assert client._is_provider_cooled_down("gemini-3-flash-preview") is True

    def test_cooldown_expires(self, llm_config):
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            # Set cooldown that already expired
            client._provider_cooldowns["gemini-3-flash-preview"] = time.monotonic() - 1.0
            assert client._is_provider_cooled_down("gemini-3-flash-preview") is False
            # Should be removed after expiry check
            assert "gemini-3-flash-preview" not in client._provider_cooldowns

    def test_cooldown_does_not_affect_other_providers(self, llm_config):
        with (
            patch("services.llm.client.AnthropicProvider"),
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)
            client._set_provider_cooldown("gemini-3-flash-preview", seconds=3600)
            assert client._is_provider_cooled_down("claude-haiku-4-5-20251001") is False

    @pytest.mark.asyncio
    async def test_rate_limit_error_skips_retries_and_falls_back(self, llm_config):
        """429 errors should skip remaining retries and move to next provider."""
        with (
            patch("services.llm.client.AnthropicProvider") as MockA,
            patch("services.llm.client.GeminiProvider") as MockG,
        ):
            client = LLMClient(llm_config)

            # Haiku returns rate limit error
            haiku_provider = MockA.return_value
            haiku_provider.create = AsyncMock(
                side_effect=Exception("429 Too Many Requests"),
            )

            # Gemini succeeds
            gemini_provider = MockG.return_value
            gemini_response = LLMResponse(
                text="gemini ok",
                model="gemini-3-flash-preview",
            )
            gemini_provider.create = AsyncMock(return_value=gemini_response)

            result = await client.generate(
                messages=[{"role": "user", "content": "test"}],
                retries=3,
            )

            # Should only have tried Haiku once (no retries on rate limit)
            assert haiku_provider.create.call_count == 1
            assert result.text == "gemini ok"

    @pytest.mark.asyncio
    async def test_rate_limit_sets_cooldown(self, llm_config):
        """Rate-limited provider should be cooled down for subsequent calls."""
        with (
            patch("services.llm.client.AnthropicProvider") as MockA,
            patch("services.llm.client.GeminiProvider") as MockG,
        ):
            client = LLMClient(llm_config)

            haiku_provider = MockA.return_value
            haiku_provider.create = AsyncMock(
                side_effect=Exception("RESOURCE_EXHAUSTED: quota"),
            )

            gemini_response = LLMResponse(
                text="gemini ok",
                model="gemini-3-flash-preview",
            )
            gemini_provider = MockG.return_value
            gemini_provider.create = AsyncMock(return_value=gemini_response)

            await client.generate(
                messages=[{"role": "user", "content": "test"}],
                retries=3,
            )

            # Haiku should be in cooldown
            assert client._is_provider_cooled_down("claude-haiku-4-5-20251001") is True

    @pytest.mark.asyncio
    async def test_cooled_down_provider_skipped(self, llm_config):
        """Providers in cooldown should be skipped entirely."""
        with (
            patch("services.llm.client.AnthropicProvider") as MockA,
            patch("services.llm.client.GeminiProvider"),
        ):
            client = LLMClient(llm_config)

            # Pre-set Haiku and Gemini cooldown
            client._set_provider_cooldown("claude-haiku-4-5-20251001", seconds=3600)
            client._set_provider_cooldown("gemini-3-flash-preview", seconds=3600)

            # Sonnet succeeds
            sonnet_response = LLMResponse(
                text="sonnet ok",
                model="claude-sonnet-4-6",
            )
            anthropic_provider = MockA.return_value
            anthropic_provider.create = AsyncMock(return_value=sonnet_response)

            result = await client.generate(
                messages=[{"role": "user", "content": "test"}],
            )

            # Sonnet should have been used (Haiku and Gemini were cooled down)
            assert result.text == "sonnet ok"
            # Provider.create only called for Sonnet
            assert anthropic_provider.create.call_count == 1

    @pytest.mark.asyncio
    async def test_gemini_quota_falls_back_to_anthropic(self, llm_config):
        """Simulates the exact STOCK-14 scenario: Gemini override + quota hit."""
        with (
            patch("services.llm.client.AnthropicProvider") as MockA,
            patch("services.llm.client.GeminiProvider") as MockG,
        ):
            client = LLMClient(llm_config)

            gemini_provider = MockG.return_value
            gemini_provider.create = AsyncMock(
                side_effect=Exception("429 RESOURCE_EXHAUSTED: free tier limit exceeded"),
            )

            haiku_response = LLMResponse(
                text="haiku fallback",
                model="claude-haiku-4-5-20251001",
            )
            anthropic_provider = MockA.return_value
            anthropic_provider.create = AsyncMock(return_value=haiku_response)

            # Use model_override like news_sentiment_agent does
            result = await client.generate(
                messages=[{"role": "user", "content": "analyze news"}],
                model="gemini-3-flash-preview",
                retries=3,
            )

            # Should have fallen back to Haiku
            assert result.text == "haiku fallback"
            # Gemini only tried once (rate limit → no retry)
            assert gemini_provider.create.call_count == 1
