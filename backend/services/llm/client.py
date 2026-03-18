"""Multi-provider LLM client with fallback chain, retry logic, and daily budget."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from services.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    LLMResponse,
    ToolCall,
)

logger = structlog.get_logger(__name__)

# Re-export for convenience
__all__ = ["LLMClient", "LLMResponse", "ToolCall"]

# Rate-limit / quota-exceeded error patterns
_RATE_LIMIT_PATTERNS = (
    "429",
    "resource_exhausted",
    "rate_limit",
    "rate limit",
    "quota",
    "too many requests",
)

# Default cooldown for quota-exceeded providers (1 hour in seconds)
_DEFAULT_COOLDOWN_SECONDS = 3600


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception indicates a rate-limit or quota-exceeded error.

    Detects HTTP 429, gRPC RESOURCE_EXHAUSTED, and common quota error messages
    from both Anthropic and Google Gemini APIs.
    """
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in _RATE_LIMIT_PATTERNS)


class LLMClient:
    """Provider-agnostic LLM client with automatic fallback.

    Fallback chain: primary model -> fallback model -> gemini model.
    Each model gets multiple retry attempts before moving to the next.
    Includes daily call budget to prevent runaway costs.

    Rate-limit aware: quota-exceeded errors skip retries and trigger
    provider cooldown to avoid wasting time on known-failed providers.
    """

    def __init__(self, config: Any) -> None:
        """Initialize from LLMConfig.

        Parameters
        ----------
        config : LLMConfig
            Must have: model, fallback_model, api_key,
            Optional: gemini_api_key, gemini_fallback_model
        """
        self._config = config
        self._anthropic: AnthropicProvider | None = None
        self._gemini: GeminiProvider | None = None

        # Daily call budget tracking
        self._daily_calls = 0
        self._daily_reset_date = ""
        self._max_daily_calls = getattr(config, "max_daily_calls", 0)

        # Per-provider cooldown tracking: model -> expiry timestamp
        self._provider_cooldowns: dict[str, float] = {}

        # Lazy-init providers
        if config.api_key:
            try:
                self._anthropic = AnthropicProvider(api_key=config.api_key)
            except Exception as e:
                logger.warning("anthropic_provider_init_failed", error=str(e))

        gemini_key = getattr(config, "gemini_api_key", "")
        if gemini_key:
            try:
                self._gemini = GeminiProvider(api_key=gemini_key)
            except Exception as e:
                logger.warning("gemini_provider_init_failed", error=str(e))

    def _check_budget(self) -> bool:
        """Check if we're within daily call budget. Returns True if allowed."""
        if self._max_daily_calls <= 0:
            return True  # unlimited

        today = time.strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_calls = 0
            self._daily_reset_date = today

        return self._daily_calls < self._max_daily_calls

    def _increment_calls(self) -> None:
        """Increment daily call counter."""
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_calls = 0
            self._daily_reset_date = today
        self._daily_calls += 1

    @property
    def daily_calls_remaining(self) -> int:
        """Number of calls remaining today (-1 if unlimited)."""
        if self._max_daily_calls <= 0:
            return -1
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            return self._max_daily_calls
        return max(0, self._max_daily_calls - self._daily_calls)

    def _is_provider_cooled_down(self, model: str) -> bool:
        """Check if a provider is in cooldown (quota-exceeded recently)."""
        expiry = self._provider_cooldowns.get(model)
        if expiry is None:
            return False
        if time.monotonic() >= expiry:
            # Cooldown expired, remove it
            del self._provider_cooldowns[model]
            return False
        return True

    def _set_provider_cooldown(
        self,
        model: str,
        seconds: float = _DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        """Set cooldown for a provider after quota-exceeded error."""
        self._provider_cooldowns[model] = time.monotonic() + seconds
        logger.warning(
            "llm_provider_cooldown_set",
            model=model,
            cooldown_seconds=seconds,
        )

    def _resolve_provider(self, model: str) -> LLMProvider | None:
        if model.startswith("gemini"):
            return self._gemini
        return self._anthropic

    def _build_fallback_chain(
        self,
        model_override: str | None = None,
    ) -> list[tuple[str, LLMProvider]]:
        """Build [(model_name, provider), ...] in fallback order.

        Cost-aware ordering with deduplication:
        - Default: Haiku -> Gemini (free) -> Sonnet (expensive, last resort)
        - With model_override: override -> Haiku -> Gemini -> Sonnet
          (override is primary, others are fallbacks; duplicates removed)
        """
        chain: list[tuple[str, LLMProvider]] = []
        seen_models: set[str] = set()

        def _add(model: str, provider: LLMProvider | None) -> None:
            if provider and model not in seen_models:
                chain.append((model, provider))
                seen_models.add(model)

        # Primary model
        primary = model_override or self._config.model
        _add(primary, self._resolve_provider(primary))

        # When model_override is set, add primary Anthropic model (Haiku) as fallback
        if model_override and self._config.model:
            _add(self._config.model, self._resolve_provider(self._config.model))

        # Gemini fallback (free tier)
        gemini_model = getattr(self._config, "gemini_fallback_model", "")
        if gemini_model and self._gemini:
            _add(gemini_model, self._gemini)

        # Sonnet as last resort (13x more expensive than Haiku)
        if self._config.fallback_model:
            p = self._resolve_provider(self._config.fallback_model)
            if p:
                _add(self._config.fallback_model, p)

        return chain

    async def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        system: str | None = None,
        model: str | None = None,
        retries: int = 3,
    ) -> LLMResponse:
        """Text generation with retry + fallback.

        Raises RuntimeError if all providers fail or budget exhausted.
        """
        return await self._call_with_fallback(
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            tools=None,
            model_override=model,
            retries=retries,
        )

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        max_tokens: int = 2048,
        system: str | None = None,
        model: str | None = None,
        retries: int = 2,
    ) -> LLMResponse:
        """Tool-use generation with retry + fallback.

        Raises RuntimeError if all providers fail or budget exhausted.
        """
        return await self._call_with_fallback(
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            model_override=model,
            retries=retries,
        )

    def format_tool_loop_messages(
        self,
        response: LLMResponse,
        tool_results: list[dict],
    ) -> tuple[dict, dict]:
        """Build (assistant_msg, user_msg) for multi-turn tool loop.

        Delegates to the provider that produced the response.
        """
        provider = self._resolve_provider(response.model)
        if not provider:
            raise RuntimeError(f"No provider for model: {response.model}")
        return provider.format_tool_loop_messages(response, tool_results)

    async def _call_with_fallback(
        self,
        messages: list[dict],
        max_tokens: int,
        system: str | None,
        tools: list[dict] | None,
        model_override: str | None,
        retries: int,
    ) -> LLMResponse:
        # Budget check
        if not self._check_budget():
            logger.warning(
                "llm_daily_budget_exhausted",
                daily_calls=self._daily_calls,
                max_daily=self._max_daily_calls,
            )
            raise RuntimeError(
                f"Daily LLM call budget exhausted ({self._daily_calls}/{self._max_daily_calls})"
            )

        chain = self._build_fallback_chain(model_override)
        if not chain:
            raise RuntimeError("No LLM providers configured")

        last_error: Exception | None = None

        for model, provider in chain:
            # Skip providers in cooldown (quota-exceeded recently)
            if self._is_provider_cooled_down(model):
                logger.debug(
                    "llm_provider_in_cooldown",
                    model=model,
                )
                continue

            for attempt in range(retries):
                try:
                    response = await provider.create(
                        messages=messages,
                        model=model,
                        max_tokens=max_tokens,
                        system=system,
                        tools=tools,
                    )
                    self._increment_calls()
                    logger.debug(
                        "llm_call_success",
                        model=model,
                        attempt=attempt + 1,
                        has_tools=bool(tools),
                        daily_calls=self._daily_calls,
                    )
                    return response
                except Exception as e:
                    last_error = e

                    # Rate-limit / quota errors: skip retries, set cooldown
                    if _is_rate_limit_error(e):
                        logger.warning(
                            "llm_rate_limit_hit",
                            model=model,
                            error=str(e)[:200],
                        )
                        self._set_provider_cooldown(model)
                        break  # Skip remaining retries, move to next provider

                    wait = 2**attempt * 2  # 2s, 4s, 8s
                    logger.warning(
                        "llm_call_failed",
                        model=model,
                        attempt=attempt + 1,
                        error=str(e)[:200],
                        retry_in=wait if attempt < retries - 1 else None,
                    )
                    if attempt < retries - 1:
                        await asyncio.sleep(wait)

            logger.warning("llm_model_exhausted", model=model)

        raise RuntimeError(f"All LLM providers failed: {last_error}")
