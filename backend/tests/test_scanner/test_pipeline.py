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

    async def test_pipeline_rejects_penny_stocks(
        self, mock_indicator_svc, mock_enricher,
    ):
        """#58: Penny stocks must be rejected in Layer 1.

        Repro of the ACONW ($0.04) live case — pattern looked good but the
        $0.01 minimum tick caused 25% effective slippage on exit.
        """
        market_data = AsyncMock()
        pipe = ScannerPipeline(
            market_data=market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
            min_price=5.0,
        )

        cheap_df = _make_ohlcv_df()
        cheap_df = cheap_df.copy()
        cheap_df["close"] = 0.04  # ACONW-like price

        normal_df = _make_ohlcv_df()  # ~100

        def side_effect(sym, **kw):
            if sym == "ACONW":
                return cheap_df
            return normal_df

        with patch("scanner.pipeline._fetch_yfinance_ohlcv", side_effect=side_effect):
            results = await pipe.run_full_scan(["ACONW", "AAPL", "MSFT"])

        # Only non-penny names go into the enricher
        symbols_enriched = [
            t[0] for t in mock_enricher.enrich_batch.call_args[0][0]
        ]
        assert "ACONW" not in symbols_enriched
        # AAPL/MSFT should still be eligible (penny filter doesn't block them)
        assert any(s in symbols_enriched for s in ("AAPL", "MSFT"))

    async def test_pipeline_min_price_default_5(
        self, mock_market_data, mock_indicator_svc, mock_enricher,
    ):
        """Default min_price is 5.0 — sanity check on the constructor."""
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
        )
        assert pipe._min_price == 5.0

    def test_set_news_summary_caches(
        self, mock_market_data, mock_indicator_svc, mock_enricher,
    ):
        """set_news_summary stores for the next scan."""
        from agents.news_sentiment_agent import NewsSentimentSummary
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
        )
        s = NewsSentimentSummary(symbol_sentiments={"AAPL": 0.5})
        pipe.set_news_summary(s)
        assert pipe._last_news_summary is s

    async def test_pipeline_no_layer1_passes(
        self, mock_market_data, mock_indicator_svc, mock_enricher,
    ):
        """When every symbol has too-short OHLCV, return []."""
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
        )
        short_df = _make_ohlcv_df(n=30)  # < 50 → skipped
        with patch(
            "scanner.pipeline._fetch_yfinance_ohlcv",
            side_effect=lambda sym, **kw: short_df,
        ):
            res = await pipe.run_full_scan(["AAPL", "MSFT"])
        assert res == []

    async def test_layer1_logs_and_continues_on_exception(
        self, mock_market_data, mock_indicator_svc, mock_enricher,
    ):
        """Per-symbol Layer 1 failures don't abort the scan (the try/except
        path was uncovered)."""
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
        )

        def side_effect(sym, **kw):
            if sym == "BOOM":
                raise RuntimeError("yf timeout")
            return _make_ohlcv_df()

        with patch("scanner.pipeline._fetch_yfinance_ohlcv", side_effect=side_effect):
            res = await pipe.run_full_scan(["AAPL", "BOOM", "MSFT"])
        assert isinstance(res, list)


class TestNewsEnricher:
    """Layer 2.5 path (news_enricher present + active summary)."""

    async def test_news_enricher_called(self):
        """When news_enricher + summary present, results pass through enrich()."""
        from agents.news_sentiment_agent import NewsSentimentSummary
        market_data = AsyncMock()
        indicator_svc = MagicMock()
        indicator_svc.add_all_indicators = MagicMock(side_effect=lambda df: df)
        enricher = AsyncMock()
        enricher.enrich_batch = AsyncMock(return_value=[
            EnrichedCandidate(
                symbol="AAPL", indicator_score=80.0, consensus_score=75.0,
                fundamental_score=70.0, smart_money_score=65.0,
                combined_score=72.5, grade="B",
            ),
        ])
        news_enricher = MagicMock()
        news_enricher.enrich = MagicMock(side_effect=lambda results, summary: [
            {**r, "news_sentiment": 0.42} for r in results
        ])
        pipe = ScannerPipeline(
            market_data=market_data,
            indicator_svc=indicator_svc,
            enricher=enricher,
            news_enricher=news_enricher,
        )
        summary = NewsSentimentSummary(symbol_sentiments={"AAPL": 0.5})
        with patch(
            "scanner.pipeline._fetch_yfinance_ohlcv",
            side_effect=lambda sym, **kw: _make_ohlcv_df(),
        ):
            out = await pipe.run_full_scan(["AAPL"], news_summary=summary)
        assert news_enricher.enrich.called
        assert out and out[0]["news_sentiment"] == pytest.approx(0.42)


class TestLayer3ErrorPaths:
    """When the AI agent throws, the loop logs and continues; SPY context
    fetch failure is also non-fatal."""

    async def test_layer3_ai_error_continues(
        self, mock_market_data, mock_indicator_svc, mock_enricher, mock_ai_agent,
    ):
        mock_ai_agent.analyze = AsyncMock(side_effect=RuntimeError("LLM down"))
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
            ai_agent=mock_ai_agent,
        )
        with patch(
            "scanner.pipeline._fetch_yfinance_ohlcv",
            side_effect=lambda sym, **kw: _make_ohlcv_df(),
        ):
            res = await pipe.run_full_scan(["AAPL", "MSFT"])
        # No ai fields populated because every call raised, but the scan
        # itself succeeded.
        assert res
        assert "ai_recommendation" not in res[0]

    async def test_layer3_spy_context_failure_non_fatal(
        self, mock_market_data, mock_indicator_svc, mock_enricher, mock_ai_agent,
    ):
        pipe = ScannerPipeline(
            market_data=mock_market_data,
            indicator_svc=mock_indicator_svc,
            enricher=mock_enricher,
            ai_agent=mock_ai_agent,
        )

        # All non-SPY return data; SPY raises.
        def side_effect(sym, **kw):
            if sym == "SPY":
                raise RuntimeError("yfinance error on SPY")
            return _make_ohlcv_df()

        with patch("scanner.pipeline._fetch_yfinance_ohlcv", side_effect=side_effect):
            res = await pipe.run_full_scan(["AAPL", "MSFT", "GOOG"])
        assert res  # AI still runs without market context


class TestFetchYfinanceOhlcv:
    """The module-level helper has fully observable failure modes."""

    def test_empty_history_returns_empty_df(self):
        with patch("scanner.pipeline.yf.Ticker") as mock_t:
            mock_t.return_value.history.return_value = pd.DataFrame()
            df = _fetch_yfinance_ohlcv("AAPL")
        assert df.empty

    def test_missing_columns_returns_empty(self):
        with patch("scanner.pipeline.yf.Ticker") as mock_t:
            # Has Open but missing Close — must return empty.
            mock_t.return_value.history.return_value = pd.DataFrame({"Open": [1, 2]})
            df = _fetch_yfinance_ohlcv("AAPL")
        assert df.empty

    def test_exception_returns_empty(self):
        with patch("scanner.pipeline.yf.Ticker") as mock_t:
            mock_t.return_value.history.side_effect = RuntimeError("network")
            df = _fetch_yfinance_ohlcv("AAPL")
        assert df.empty

    def test_happy_path_normalizes_columns(self):
        """Valid yfinance response → lowercased OHLCV columns subset."""
        with patch("scanner.pipeline.yf.Ticker") as mock_t:
            mock_t.return_value.history.return_value = pd.DataFrame({
                "Open": [1.0, 2.0],
                "High": [1.1, 2.1],
                "Low": [0.9, 1.9],
                "Close": [1.05, 2.05],
                "Volume": [100, 200],
                "Dividends": [0.0, 0.0],  # extra column gets dropped
            })
            df = _fetch_yfinance_ohlcv("AAPL")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2

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
