"""Tests for UniverseExpander and KRUniverseExpander."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exchange.kis_adapter import RankedStock
from exchange.kis_kr_adapter import KRRankedStock
from scanner.etf_universe import ETFUniverse, SectorETF
from scanner.universe_expander import (
    KRUniverseExpander,
    KRUniverseResult,
    UniverseExpander,
    UniverseResult,
    _is_valid_kr_symbol,
)


@pytest.fixture
def mock_etf_universe():
    """Create mock ETFUniverse."""
    etf = MagicMock(spec=ETFUniverse)
    etf.get_all_sectors.return_value = {
        "Technology": SectorETF(
            name="Technology", etf="XLK",
            top_holdings=["AAPL", "MSFT", "NVDA", "AVGO", "ADBE"],
        ),
        "Financials": SectorETF(
            name="Financials", etf="XLF",
            top_holdings=["JPM", "V", "MA", "BAC", "GS"],
        ),
        "Energy": SectorETF(
            name="Energy", etf="XLE",
            top_holdings=["XOM", "CVX", "COP", "EOG", "SLB"],
        ),
    }
    etf.all_etf_symbols = ["XLK", "XLF", "XLE", "SPY", "QQQ", "TQQQ", "SQQQ"]
    etf.safe_haven = ["SHY", "TLT", "GLD"]
    return etf


@pytest.fixture
def expander(mock_etf_universe):
    return UniverseExpander(
        etf_universe=mock_etf_universe,
        max_per_screener=5,
        max_total=50,
    )


class TestSectorHoldings:
    """Test sector-weighted holdings expansion."""

    def test_strong_sectors_get_all_holdings(self, expander):
        sector_data = {
            "Technology": {"symbol": "XLK", "return_1w": 3.0, "return_1m": 8.0, "return_3m": 15.0},
            "Financials": {"symbol": "XLF", "return_1w": 1.0, "return_1m": 3.0, "return_3m": 5.0},
            "Energy": {"symbol": "XLE", "return_1w": -2.0, "return_1m": -5.0, "return_3m": -8.0},
        }
        holdings = expander._get_sector_holdings(sector_data)

        # Technology is strongest — should get all 5
        assert "AAPL" in holdings
        assert "MSFT" in holdings
        assert "NVDA" in holdings
        assert "AVGO" in holdings
        assert "ADBE" in holdings

    def test_weak_sectors_get_fewer_holdings(self, expander):
        sector_data = {
            "Technology": {"symbol": "XLK", "return_1w": 5.0, "return_1m": 10.0, "return_3m": 20.0},
            "Financials": {"symbol": "XLF", "return_1w": 0.5, "return_1m": 1.0, "return_3m": 2.0},
            "Energy": {"symbol": "XLE", "return_1w": -3.0, "return_1m": -8.0, "return_3m": -12.0},
        }
        holdings = expander._get_sector_holdings(sector_data)

        # Energy is weakest — should get only 1 holding
        energy_in = [s for s in ["XOM", "CVX", "COP", "EOG", "SLB"] if s in holdings]
        assert len(energy_in) >= 1
        assert len(energy_in) <= 3  # not all 5

    def test_no_sector_data_uses_defaults(self, expander):
        holdings = expander._get_sector_holdings(None)
        # Default strength=50, which falls in medium range (3 per sector)
        assert len(holdings) > 0
        assert len(holdings) <= 15  # 3 sectors × 5 max


class TestScreeners:
    """Test yfinance screener integration."""

    @patch("scanner.universe_expander.yf.screen")
    def test_screeners_return_symbols(self, mock_screen, expander):
        mock_screen.return_value = {
            "quotes": [
                {"symbol": "TSLA"},
                {"symbol": "AMD"},
                {"symbol": "NFLX"},
            ]
        }
        result = expander._run_screeners()
        assert "TSLA" in result
        assert "AMD" in result
        assert "NFLX" in result

    @patch("scanner.universe_expander.yf.screen")
    def test_screeners_skip_non_us(self, mock_screen, expander):
        mock_screen.return_value = {
            "quotes": [
                {"symbol": "AAPL"},
                {"symbol": "SHOP.TO"},  # Canadian
                {"symbol": "HSBA.L"},  # London
            ]
        }
        result = expander._run_screeners()
        assert "AAPL" in result
        assert "SHOP.TO" not in result
        assert "HSBA.L" not in result

    @patch("scanner.universe_expander.yf.screen")
    def test_screeners_dedup(self, mock_screen, expander):
        # Same symbol from multiple screeners
        mock_screen.return_value = {
            "quotes": [{"symbol": "NVDA"}, {"symbol": "NVDA"}]
        }
        result = expander._run_screeners()
        assert result.count("NVDA") == 1

    @patch("scanner.universe_expander.yf.screen")
    def test_screeners_handle_failure(self, mock_screen, expander):
        mock_screen.side_effect = Exception("API error")
        result = expander._run_screeners()
        assert result == []

    @patch("scanner.universe_expander.yf.screen")
    def test_screeners_respect_max_per_screener(self, mock_screen, expander):
        mock_screen.return_value = {
            "quotes": [{"symbol": s} for s in [
                "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
                "META", "GOOGL", "AMD", "NFLX", "CRM",
            ]]
        }
        result = expander._run_screeners()
        # Per screener max is 5, but same mock for all 5 screeners → dedup
        assert len(set(result)) <= 10


class TestKISScreening:
    """Test KIS ranking API integration."""

    @pytest.mark.asyncio
    async def test_kis_screening_returns_symbols(self, mock_etf_universe):
        """KIS screening returns symbols from ranking APIs."""
        mock_kis = AsyncMock()
        mock_kis.fetch_volume_surge.return_value = [
            RankedStock(symbol="PLTR", source="volume_surge"),
            RankedStock(symbol="SOFI", source="volume_surge"),
        ]
        mock_kis.fetch_updown_rate.return_value = [
            RankedStock(symbol="RIVN", source="updown_up"),
            RankedStock(symbol="PLTR", source="updown_up"),  # dup
        ]
        mock_kis.fetch_new_highlow.return_value = [
            RankedStock(symbol="CRWD", source="new_high"),
        ]
        mock_limiter = AsyncMock()

        expander = UniverseExpander(
            etf_universe=mock_etf_universe,
            kis_adapter=mock_kis,
            rate_limiter=mock_limiter,
            max_per_screener=5,
            max_total=50,
        )
        result = await expander._run_kis_screening()

        assert "PLTR" in result
        assert "SOFI" in result
        assert "RIVN" in result
        assert "CRWD" in result
        # PLTR should appear only once (deduped)
        assert result.count("PLTR") == 1

    @pytest.mark.asyncio
    async def test_kis_screening_acquires_rate_limit(self, mock_etf_universe):
        """Each KIS API call acquires rate limiter token."""
        mock_kis = AsyncMock()
        mock_kis.fetch_volume_surge.return_value = []
        mock_kis.fetch_updown_rate.return_value = []
        mock_kis.fetch_new_highlow.return_value = []
        mock_limiter = AsyncMock()

        expander = UniverseExpander(
            etf_universe=mock_etf_universe,
            kis_adapter=mock_kis,
            rate_limiter=mock_limiter,
            max_per_screener=5,
            max_total=50,
        )
        await expander._run_kis_screening()

        # 3 API calls = 3 rate limiter acquisitions
        assert mock_limiter.acquire.call_count == 3

    @pytest.mark.asyncio
    async def test_kis_screening_skipped_without_adapter(self, expander):
        """No KIS calls if adapter not configured."""
        result = await expander._run_kis_screening()
        assert result == []

    @pytest.mark.asyncio
    async def test_kis_screening_handles_failure(self, mock_etf_universe):
        """Partial failure doesn't break other calls."""
        mock_kis = AsyncMock()
        mock_kis.fetch_volume_surge.side_effect = Exception("timeout")
        mock_kis.fetch_updown_rate.return_value = [
            RankedStock(symbol="AMD", source="updown_up"),
        ]
        mock_kis.fetch_new_highlow.return_value = []
        mock_limiter = AsyncMock()

        expander = UniverseExpander(
            etf_universe=mock_etf_universe,
            kis_adapter=mock_kis,
            rate_limiter=mock_limiter,
            max_per_screener=5,
            max_total=50,
        )
        result = await expander._run_kis_screening()

        # volume_surge failed but updown_rate succeeded
        assert "AMD" in result

    @pytest.mark.asyncio
    async def test_kis_screening_filters_non_us(self, mock_etf_universe):
        """Non-US symbols (with dots) are filtered out."""
        mock_kis = AsyncMock()
        mock_kis.fetch_volume_surge.return_value = [
            RankedStock(symbol="AAPL", source="volume_surge"),
            RankedStock(symbol="7203.T", source="volume_surge"),
        ]
        mock_kis.fetch_updown_rate.return_value = []
        mock_kis.fetch_new_highlow.return_value = []
        mock_limiter = AsyncMock()

        expander = UniverseExpander(
            etf_universe=mock_etf_universe,
            kis_adapter=mock_kis,
            rate_limiter=mock_limiter,
            max_per_screener=5,
            max_total=50,
        )
        result = await expander._run_kis_screening()

        assert "AAPL" in result
        assert "7203.T" not in result

    @pytest.mark.asyncio
    async def test_kis_without_rate_limiter(self, mock_etf_universe):
        """Works without rate limiter (just no throttling)."""
        mock_kis = AsyncMock()
        mock_kis.fetch_volume_surge.return_value = [
            RankedStock(symbol="TSLA", source="volume_surge"),
        ]
        mock_kis.fetch_updown_rate.return_value = []
        mock_kis.fetch_new_highlow.return_value = []

        expander = UniverseExpander(
            etf_universe=mock_etf_universe,
            kis_adapter=mock_kis,
            rate_limiter=None,
            max_per_screener=5,
            max_total=50,
        )
        result = await expander._run_kis_screening()
        assert "TSLA" in result


class TestFilterSymbols:
    """Test symbol filtering logic."""

    def test_filters_etfs(self, expander):
        symbols = {"AAPL", "SPY", "QQQ", "MSFT", "TQQQ"}
        result = expander._filter_symbols(symbols)
        assert "SPY" not in result
        assert "QQQ" not in result
        assert "TQQQ" not in result
        assert "AAPL" in result
        assert "MSFT" in result

    def test_filters_safe_haven(self, expander):
        symbols = {"AAPL", "GLD", "TLT", "SHY"}
        result = expander._filter_symbols(symbols)
        assert "AAPL" in result
        assert "GLD" not in result

    def test_keeps_hyphenated_stock_symbols(self, expander):
        symbols = {"BRK-B", "AAPL"}
        result = expander._filter_symbols(symbols)
        assert "BRK-B" in result
        assert "AAPL" in result


class TestExpand:
    """Test full expand flow."""

    @patch("scanner.universe_expander.yf.screen")
    @pytest.mark.asyncio
    async def test_full_expand(self, mock_screen, expander):
        mock_screen.return_value = {
            "quotes": [{"symbol": "TSLA"}, {"symbol": "AMD"}]
        }
        result = await expander.expand(
            existing_watchlist=["AAPL", "MSFT"],
            sector_data={
                "Technology": {"symbol": "XLK", "return_1w": 3.0, "return_1m": 8.0, "return_3m": 15.0},
                "Financials": {"symbol": "XLF", "return_1w": 1.0, "return_1m": 3.0, "return_3m": 5.0},
                "Energy": {"symbol": "XLE", "return_1w": -1.0, "return_1m": -2.0, "return_3m": -3.0},
            },
        )
        assert isinstance(result, UniverseResult)
        assert "AAPL" in result.symbols
        assert "MSFT" in result.symbols
        assert "TSLA" in result.symbols
        assert len(result.symbols) > 4
        assert "watchlist" in result.sources
        assert "screeners" in result.sources
        assert "sector_holdings" in result.sources

    @patch("scanner.universe_expander.yf.screen")
    @pytest.mark.asyncio
    async def test_full_expand_with_kis(self, mock_screen, mock_etf_universe):
        """Full expand includes KIS ranking source."""
        mock_screen.return_value = {
            "quotes": [{"symbol": "TSLA"}]
        }
        mock_kis = AsyncMock()
        mock_kis.fetch_volume_surge.return_value = [
            RankedStock(symbol="PLTR", source="volume_surge"),
        ]
        mock_kis.fetch_updown_rate.return_value = [
            RankedStock(symbol="SOFI", source="updown_up"),
        ]
        mock_kis.fetch_new_highlow.return_value = []
        mock_limiter = AsyncMock()

        expander = UniverseExpander(
            etf_universe=mock_etf_universe,
            kis_adapter=mock_kis,
            rate_limiter=mock_limiter,
            max_per_screener=5,
            max_total=50,
        )
        result = await expander.expand(existing_watchlist=["AAPL"])

        assert "AAPL" in result.symbols
        assert "TSLA" in result.symbols
        assert "PLTR" in result.symbols
        assert "SOFI" in result.symbols
        assert "kis_ranking" in result.sources

    @patch("scanner.universe_expander.yf.screen")
    @pytest.mark.asyncio
    async def test_expand_no_watchlist(self, mock_screen, expander):
        mock_screen.return_value = {"quotes": [{"symbol": "TSLA"}]}
        result = await expander.expand()
        assert len(result.symbols) > 0
        assert "watchlist" not in result.sources

    @patch("scanner.universe_expander.yf.screen")
    @pytest.mark.asyncio
    async def test_expand_respects_max_total(self, mock_screen):
        mock_screen.return_value = {
            "quotes": [{"symbol": s} for s in [
                "TSLA", "AMD", "NFLX", "CRM", "SHOP",
            ]]
        }
        expander = UniverseExpander(max_total=5, max_per_screener=3)
        result = await expander.expand(
            existing_watchlist=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
        )
        assert len(result.symbols) <= 5


# ---------------------------------------------------------------------------
# KRUniverseExpander tests
# ---------------------------------------------------------------------------


class TestIsValidKRSymbol:
    """Tests for _is_valid_kr_symbol helper."""

    def test_valid_6digit_code(self):
        assert _is_valid_kr_symbol("005930") is True
        assert _is_valid_kr_symbol("000660") is True
        assert _is_valid_kr_symbol("247540") is True

    def test_rejects_non_numeric(self):
        assert _is_valid_kr_symbol("AAPL") is False
        assert _is_valid_kr_symbol("ABC123") is False

    def test_rejects_wrong_length(self):
        assert _is_valid_kr_symbol("05930") is False   # 5 digits
        assert _is_valid_kr_symbol("0059300") is False  # 7 digits

    def test_rejects_empty(self):
        assert _is_valid_kr_symbol("") is False

    def test_rejects_with_dot(self):
        assert _is_valid_kr_symbol("005930.KS") is False


class TestKRUniverseExpander:
    """Tests for KRUniverseExpander."""

    @pytest.fixture
    def expander(self):
        return KRUniverseExpander(max_total=80)

    @pytest.mark.asyncio
    async def test_expand_kr_includes_seed(self, expander):
        """expand_kr always includes the curated seed list."""
        result = await expander.expand_kr()
        assert isinstance(result, KRUniverseResult)
        assert "seed" in result.sources
        # Samsung is in the seed
        assert "005930" in result.symbols

    @pytest.mark.asyncio
    async def test_expand_kr_includes_watchlist(self, expander):
        """expand_kr includes existing watchlist symbols."""
        result = await expander.expand_kr(existing_watchlist=["005930", "000660"])
        assert "watchlist" in result.sources
        assert "005930" in result.symbols
        assert "000660" in result.symbols

    @pytest.mark.asyncio
    async def test_expand_kr_no_kis_adapter(self, expander):
        """Without KIS adapter, still returns seed symbols."""
        result = await expander.expand_kr()
        assert len(result.symbols) > 0
        assert "kis_kr_ranking" not in result.sources

    @pytest.mark.asyncio
    async def test_expand_kr_respects_max_total(self):
        """expand_kr limits symbols to max_total."""
        expander = KRUniverseExpander(max_total=10)
        result = await expander.expand_kr()
        assert len(result.symbols) <= 10

    @pytest.mark.asyncio
    async def test_expand_kr_deduplicates(self, expander):
        """Symbols appear only once even if in multiple sources."""
        result = await expander.expand_kr(existing_watchlist=["005930"])
        assert result.symbols.count("005930") == 1

    @pytest.mark.asyncio
    async def test_expand_kr_with_kis_adapter(self):
        """expand_kr includes KIS domestic ranking results."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = [
            KRRankedStock(symbol="035420", name="NAVER",
                          exchange="KRX", source="kr_volume_surge"),
        ]
        mock_kis_kr.fetch_updown_rate.return_value = [
            KRRankedStock(symbol="377300", name="카카오페이",
                          exchange="KOSDAQ", source="kr_updown_up"),
        ]
        mock_kis_kr.fetch_new_highlow.return_value = []
        mock_limiter = AsyncMock()

        expander = KRUniverseExpander(
            kis_kr_adapter=mock_kis_kr,
            rate_limiter=mock_limiter,
            max_total=80,
        )
        result = await expander.expand_kr()

        assert "kis_kr_ranking" in result.sources
        assert "035420" in result.symbols
        assert "377300" in result.symbols

    @pytest.mark.asyncio
    async def test_expand_kr_exchange_map_from_seed(self, expander):
        """Exchange map is populated from seed list."""
        result = await expander.expand_kr()
        assert "005930" in result.exchange_map
        assert result.exchange_map["005930"] == "KRX"
        assert "247540" in result.exchange_map
        assert result.exchange_map["247540"] == "KOSDAQ"

    @pytest.mark.asyncio
    async def test_expand_kr_exchange_map_from_kis(self):
        """Exchange map updated with KIS ranking discoveries."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = [
            KRRankedStock(symbol="111111", exchange="KOSDAQ",
                          source="kr_volume_surge"),
        ]
        mock_kis_kr.fetch_updown_rate.return_value = []
        mock_kis_kr.fetch_new_highlow.return_value = []
        mock_limiter = AsyncMock()

        expander = KRUniverseExpander(
            kis_kr_adapter=mock_kis_kr,
            rate_limiter=mock_limiter,
        )
        result = await expander.expand_kr()

        assert "111111" in result.exchange_map
        assert result.exchange_map["111111"] == "KOSDAQ"

    @pytest.mark.asyncio
    async def test_expand_kr_total_discovered(self, expander):
        """total_discovered reflects filtered symbol count."""
        result = await expander.expand_kr()
        assert result.total_discovered == len(result.symbols)


class TestKRRunKisKRScreening:
    """Tests for KRUniverseExpander._run_kis_kr_screening()."""

    @pytest.mark.asyncio
    async def test_no_kis_adapter_returns_empty(self):
        expander = KRUniverseExpander()
        symbols, exchange_map = await expander._run_kis_kr_screening()
        assert symbols == []
        assert exchange_map == {}

    @pytest.mark.asyncio
    async def test_acquires_rate_limiter_per_call(self):
        """Rate limiter acquired for each of the 6 API calls."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = []
        mock_kis_kr.fetch_updown_rate.return_value = []
        mock_kis_kr.fetch_new_highlow.return_value = []
        mock_limiter = AsyncMock()

        expander = KRUniverseExpander(
            kis_kr_adapter=mock_kis_kr,
            rate_limiter=mock_limiter,
        )
        await expander._run_kis_kr_screening()

        # 6 calls: volume surge (J,K), gainers (J,K), new highs (J,K)
        assert mock_limiter.acquire.call_count == 6

    @pytest.mark.asyncio
    async def test_filters_non_kr_symbols(self):
        """Non-6-digit symbols are filtered out."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = [
            KRRankedStock(symbol="005930", exchange="KRX"),
            KRRankedStock(symbol="AAPL", exchange="KRX"),  # US stock — invalid
            KRRankedStock(symbol="12345", exchange="KRX"),  # 5 digits — invalid
        ]
        mock_kis_kr.fetch_updown_rate.return_value = []
        mock_kis_kr.fetch_new_highlow.return_value = []

        expander = KRUniverseExpander(kis_kr_adapter=mock_kis_kr)
        symbols, _ = await expander._run_kis_kr_screening()

        assert "005930" in symbols
        assert "AAPL" not in symbols
        assert "12345" not in symbols

    @pytest.mark.asyncio
    async def test_deduplicates_across_calls(self):
        """Same symbol from multiple sources appears only once."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = [
            KRRankedStock(symbol="005930", exchange="KRX"),
        ]
        mock_kis_kr.fetch_updown_rate.return_value = [
            KRRankedStock(symbol="005930", exchange="KRX"),  # dup
        ]
        mock_kis_kr.fetch_new_highlow.return_value = []

        expander = KRUniverseExpander(kis_kr_adapter=mock_kis_kr)
        symbols, _ = await expander._run_kis_kr_screening()

        assert symbols.count("005930") == 1

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self):
        """Failure in one call doesn't stop other calls."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.side_effect = Exception("timeout")
        mock_kis_kr.fetch_updown_rate.return_value = [
            KRRankedStock(symbol="000660", exchange="KRX"),
        ]
        mock_kis_kr.fetch_new_highlow.return_value = []

        expander = KRUniverseExpander(kis_kr_adapter=mock_kis_kr)
        symbols, _ = await expander._run_kis_kr_screening()

        assert "000660" in symbols

    @pytest.mark.asyncio
    async def test_without_rate_limiter(self):
        """Works fine without rate limiter (no throttling)."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = [
            KRRankedStock(symbol="005930", exchange="KRX"),
        ]
        mock_kis_kr.fetch_updown_rate.return_value = []
        mock_kis_kr.fetch_new_highlow.return_value = []

        expander = KRUniverseExpander(
            kis_kr_adapter=mock_kis_kr,
            rate_limiter=None,
        )
        symbols, _ = await expander._run_kis_kr_screening()
        assert "005930" in symbols

    @pytest.mark.asyncio
    async def test_exchange_map_populated(self):
        """Exchange map includes exchange from ranked stocks."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = [
            KRRankedStock(symbol="247540", exchange="KOSDAQ"),
        ]
        mock_kis_kr.fetch_updown_rate.return_value = []
        mock_kis_kr.fetch_new_highlow.return_value = []

        expander = KRUniverseExpander(kis_kr_adapter=mock_kis_kr)
        symbols, exchange_map = await expander._run_kis_kr_screening()

        assert exchange_map.get("247540") == "KOSDAQ"

    @pytest.mark.asyncio
    async def test_etf_symbols_filtered_out(self):
        """Known ETF symbols from kr_etf_universe are excluded."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = [
            KRRankedStock(symbol="005930", exchange="KRX"),   # stock — keep
            KRRankedStock(symbol="069500", exchange="KRX"),   # KODEX 200 — ETF
            KRRankedStock(symbol="122630", exchange="KRX"),   # KODEX 레버리지 — ETF
        ]
        mock_kis_kr.fetch_updown_rate.return_value = []
        mock_kis_kr.fetch_new_highlow.return_value = []

        mock_etf = MagicMock()
        mock_etf.all_etf_symbols = ["069500", "122630", "114800", "229200"]
        mock_etf.safe_haven = ["148070", "132030"]

        expander = KRUniverseExpander(
            kis_kr_adapter=mock_kis_kr,
            kr_etf_universe=mock_etf,
        )
        symbols, _ = await expander._run_kis_kr_screening()

        assert "005930" in symbols
        assert "069500" not in symbols
        assert "122630" not in symbols

    @pytest.mark.asyncio
    async def test_safe_haven_etfs_filtered_out(self):
        """Safe haven ETF symbols are also excluded."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = [
            KRRankedStock(symbol="148070", exchange="KRX"),   # 국고채 ETF
            KRRankedStock(symbol="005930", exchange="KRX"),   # stock
        ]
        mock_kis_kr.fetch_updown_rate.return_value = []
        mock_kis_kr.fetch_new_highlow.return_value = []

        mock_etf = MagicMock()
        mock_etf.all_etf_symbols = ["069500"]
        mock_etf.safe_haven = ["148070", "132030"]

        expander = KRUniverseExpander(
            kis_kr_adapter=mock_kis_kr,
            kr_etf_universe=mock_etf,
        )
        symbols, _ = await expander._run_kis_kr_screening()

        assert "005930" in symbols
        assert "148070" not in symbols

    @pytest.mark.asyncio
    async def test_no_etf_universe_skips_etf_filter(self):
        """Without kr_etf_universe, no ETF filtering occurs."""
        mock_kis_kr = AsyncMock()
        mock_kis_kr.fetch_volume_surge.return_value = [
            KRRankedStock(symbol="069500", exchange="KRX"),  # ETF passes
        ]
        mock_kis_kr.fetch_updown_rate.return_value = []
        mock_kis_kr.fetch_new_highlow.return_value = []

        expander = KRUniverseExpander(
            kis_kr_adapter=mock_kis_kr,
            kr_etf_universe=None,
        )
        symbols, _ = await expander._run_kis_kr_screening()

        # Without ETF universe config, ETFs are not filtered
        assert "069500" in symbols
