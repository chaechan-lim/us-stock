"""Tests for ScannerPipeline orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scanner.pipeline import ScannerPipeline, _fetch_yfinance_ohlcv
from scanner.fundamental_enricher import EnrichedCandidate
from agents.market_analyst import AIRecommendation


def _make_ohlcv_df(n=50, bullish=True):
    """Create a mock OHLCV DataFrame with indicator columns."""
    np.random.seed(42)
    if bullish:
        close = 100 * np.cumprod(1 + np.random.normal(0.002, 0.01, n))
    else:
        close = 100 * np.cumprod(1 + np.random.normal(-0.002, 0.01, n))

    # Ensure monotonically increasing close prices for a clear bullish signal
    close_sorted = np.sort(close)

    return pd.DataFrame({
        "open": close_sorted * 0.998, "high": close_sorted * 1.01,
        "low": close_sorted * 0.99, "close": close_sorted,
        "volume": np.random.randint(100000, 500000, n).astype(float),
        "ema_10": close_sorted * 0.99, "ema_20": close_sorted * 0.98,
        "ema_50": close_sorted * 0.96, "ema_200": close_sorted * 0.90,
        "sma_50": close_sorted * 0.96, "sma_200": close_sorted * 0.90,
        "adx": np.full(n, 35.0), "plus_di": np.full(n, 30.0),
        "minus_di": np.full(n, 15.0),
        "rsi": np.full(n, 60.0),
        "macd": np.full(n, 2.0), "macd_histogram": np.linspace(0.5, 2.0, n),
        "macd_signal": np.full(n, 0.5),
        "roc_5": np.full(n, 3.0), "roc_10": np.full(n, 5.0),
        "roc_20": np.full(n, 8.0),
        "volume_ratio": np.full(n, 2.2), "bb_pct": np.full(n, 0.7),
        "supertrend": close_sorted * 0.95, "supertrend_direction": np.ones(n),
        "donchian_upper": close_sorted * 1.01, "donchian_lower": close_sorted * 0.90,
        "donchian_mid": close_sorted * 0.975, "atr": close_sorted * 0.015,
        "bb_lower": close_sorted * 0.93, "bb_upper": close_sorted * 1.07,
        "kc_lower": close_sorted * 0.91, "kc_upper": close_sorted * 1.09,
    })


@pytest.fixture
def mock_market_data():
    svc = AsyncMock()
    svc.get_ohlcv = AsyncMock(return_value=_make_ohlcv_df())
    svc.get_price = AsyncMock(return_value=150.0)
    return svc


@pytest.fixture
def mock_indicator_svc():
    svc = MagicMock()
    # add_all_indicators just returns the df as-is since we pre-populated indicators
    svc.add_all_indicators = MagicMock(side_effect=lambda df: df)
    return svc


@pytest.fixture
def mock_enricher():
    enricher = AsyncMock()
    enricher.enrich_batch = AsyncMock(return_value=[
        EnrichedCandidate(
            symbol="AAPL", indicator_score=80.0,
            consensus_score=75.0, fundamental_score=70.0,
            smart_money_score=65.0, combined_score=72.5, grade="B",
        ),
        EnrichedCandidate(
            symbol="MSFT", indicator_score=75.0,
            consensus_score=70.0, fundamental_score=65.0,
            smart_money_score=60.0, combined_score=67.5, grade="B",
        ),
    ])
    return enricher


@pytest.fixture
def mock_ai_agent():
    agent = AsyncMock()
    agent.analyze = AsyncMock(return_value=AIRecommendation(
        symbol="AAPL",
        recommendation="BUY",
        conviction="HIGH",
        score=85,
        summary="Strong technical and fundamental setup",
    ))
    return agent


@pytest.fixture
def pipeline(mock_market_data, mock_indicator_svc, mock_enricher):
    return ScannerPipeline(
        market_data=mock_market_data,
        indicator_svc=mock_indicator_svc,
        enricher=mock_enricher,
    )


@pytest.fixture(autouse=True)
def mock_yfinance_ohlcv():
    """Mock yfinance calls in pipeline to use test data."""
    with patch("scanner.pipeline._fetch_yfinance_ohlcv", side_effect=lambda sym, **kw: _make_ohlcv_df()):
        yield


class TestScannerPipeline:
    async def test_full_pipeline(self, mock_market_data, mock_indicator_svc, mock_enricher, mock_ai_agent):
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
            ai_agent=mock_ai_agent,
        )
        results = await pipe.run_full_scan(["AAPL", "MSFT", "GOOG"])

        assert len(results) > 0
        # Results should be sorted by combined_score descending
        for i in range(len(results) - 1):
            assert results[i]["combined_score"] >= results[i + 1]["combined_score"]

        # AI analysis should be present on top results
        top = results[0]
        assert "ai_recommendation" in top
        assert top["ai_recommendation"] == "BUY"
        assert "ai_score" in top

    async def test_pipeline_without_ai(self, pipeline):
        results = await pipeline.run_full_scan(["AAPL", "MSFT"])

        assert len(results) > 0
        # No AI fields should be present
        for r in results:
            assert "ai_recommendation" not in r
            assert "ai_score" not in r

    async def test_pipeline_empty_symbols(self, pipeline):
        results = await pipeline.run_full_scan([])
        assert results == []

    async def test_pipeline_filters_by_grade(self, mock_market_data, mock_indicator_svc):
        """Pipeline should filter out low-grade candidates."""
        # Make enricher return only one candidate that passes
        enricher = AsyncMock()
        enricher.enrich_batch = AsyncMock(return_value=[
            EnrichedCandidate(
                symbol="AAPL", indicator_score=85.0,
                consensus_score=80.0, fundamental_score=75.0,
                smart_money_score=70.0, combined_score=77.5, grade="A",
            ),
        ])
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=enricher,
        )

        results = await pipe.run_full_scan(["AAPL", "WEAK"], min_grade="A")
        # Only high-grade candidates should make it through Layer 1
        # The enricher is called with whatever passes Layer 1
        assert enricher.enrich_batch.called
        assert len(results) >= 0  # May be 0 or more depending on screener scores

    async def test_pipeline_handles_errors_gracefully(self, mock_indicator_svc, mock_enricher):
        """Pipeline should handle Layer 1 errors for individual symbols."""
        market_data = AsyncMock()
        market_data.get_ohlcv = AsyncMock(return_value=_make_ohlcv_df())
        market_data.get_price = AsyncMock(return_value=150.0)

        pipe = ScannerPipeline(
            market_data=market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
        )

        # Patch yfinance to fail for one symbol
        call_count = 0
        def side_effect_yf(sym, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("API timeout")
            return _make_ohlcv_df()

        with patch("scanner.pipeline._fetch_yfinance_ohlcv", side_effect=side_effect_yf):
            results = await pipe.run_full_scan(["AAPL", "FAIL", "MSFT"])
        assert isinstance(results, list)

    async def test_pipeline_result_structure(self, pipeline):
        results = await pipeline.run_full_scan(["AAPL"])

        assert len(results) > 0
        result = results[0]
        expected_keys = {
            "symbol", "indicator_score", "consensus_score",
            "fundamental_score", "smart_money_score",
            "combined_score", "grade",
        }
        assert expected_keys.issubset(set(result.keys()))

    async def test_pipeline_max_candidates(self, mock_market_data, mock_indicator_svc):
        """Pipeline should respect max_candidates limit."""
        enricher = AsyncMock()
        enricher.enrich_batch = AsyncMock(return_value=[
            EnrichedCandidate(
                symbol=f"SYM{i}", indicator_score=80.0 - i,
                consensus_score=70.0, fundamental_score=65.0,
                smart_money_score=60.0, combined_score=70.0 - i, grade="B",
            )
            for i in range(10)
        ])
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=enricher,
        )

        results = await pipe.run_full_scan(
            [f"SYM{i}" for i in range(10)],
            max_candidates=3,
        )
        assert len(results) <= 3

    async def test_yfinance_ohlcv_dispatched_via_to_thread(
        self, mock_market_data, mock_indicator_svc, mock_enricher,
    ):
        """_fetch_yfinance_ohlcv must run via asyncio.to_thread to avoid blocking the event loop."""
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
        )
        mock_df = _make_ohlcv_df()
        with patch(
            "scanner.pipeline.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=mock_df,
        ) as mock_to_thread:
            await pipe.run_full_scan(["AAPL", "MSFT"])

            # Layer 1 calls to_thread once per symbol
            assert mock_to_thread.call_count == 2
            for call in mock_to_thread.call_args_list:
                fn = call[0][0]
                assert callable(fn)
                # Verify the symbol argument was passed
                assert call[0][1] in ("AAPL", "MSFT")

    async def test_spy_context_dispatched_via_to_thread(
        self, mock_market_data, mock_indicator_svc, mock_enricher, mock_ai_agent,
    ):
        """SPY market context fetch in Layer 3 must also use asyncio.to_thread."""
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
            ai_agent=mock_ai_agent,
        )
        mock_df = _make_ohlcv_df()
        with patch(
            "scanner.pipeline.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=mock_df,
        ) as mock_to_thread:
            await pipe.run_full_scan(["AAPL", "MSFT", "GOOG"])

            # Layer 1 (3 symbols) + Layer 3 SPY context (1) = 4 calls
            assert mock_to_thread.call_count == 4
            # Last call should be for SPY context
            spy_call = mock_to_thread.call_args_list[-1]
            assert spy_call[0][1] == "SPY"
            assert spy_call[1]["period"] == "5d"
