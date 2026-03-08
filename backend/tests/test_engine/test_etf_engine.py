"""Tests for ETF Engine."""

import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engine.etf_engine import ETFEngine, ETFPosition, ETFRiskParams
from data.market_state import MarketState, MarketRegime
from scanner.etf_universe import ETFUniverse, LeveragedPair, SectorETF, ETFRiskRules
from scanner.sector_analyzer import SectorAnalyzer, SectorScore


@pytest.fixture
def mock_market_data():
    md = AsyncMock()
    md.get_positions.return_value = []
    md.get_balance.return_value = MagicMock(total=100000, available=50000)
    md.get_ohlcv.return_value = MagicMock(
        empty=False,
        iloc=MagicMock(__getitem__=lambda self, idx: {"close": 50.0}),
    )
    return md


@pytest.fixture
def mock_order_manager():
    om = AsyncMock()
    om.place_buy = AsyncMock()
    om.place_sell = AsyncMock()
    return om


@pytest.fixture
def mock_etf_universe():
    etf = MagicMock(spec=ETFUniverse)
    etf.risk_rules = ETFRiskRules()
    etf.get_regime_etfs.side_effect = lambda regime: {
        "bull": ["TQQQ", "SOXL"],
        "bear": ["SQQQ", "SOXS"],
    }.get(regime, [])
    etf.get_all_sectors.return_value = {
        "Technology": SectorETF(name="Technology", etf="XLK", top_holdings=["AAPL", "MSFT"]),
        "Energy": SectorETF(name="Energy", etf="XLE", top_holdings=["XOM", "CVX"]),
        "Financials": SectorETF(name="Financials", etf="XLF", top_holdings=["JPM", "V"]),
    }
    etf.is_leveraged.side_effect = lambda s: s in ["TQQQ", "SQQQ", "SOXL", "SOXS"]
    etf.all_etf_symbols = ["TQQQ", "SQQQ", "SOXL", "SOXS", "XLK", "XLE", "XLF"]
    return etf


@pytest.fixture
def mock_notification():
    return AsyncMock()


@pytest.fixture
def engine(mock_market_data, mock_order_manager, mock_etf_universe, mock_notification):
    return ETFEngine(
        market_data=mock_market_data,
        order_manager=mock_order_manager,
        etf_universe=mock_etf_universe,
        notification=mock_notification,
        max_regime_etfs=2,
        max_sector_etfs=2,
    )


def _make_ohlcv_mock(price=50.0):
    """Create a mock DataFrame with proper iloc behavior."""
    import pandas as pd
    df = pd.DataFrame({"close": [price]})
    return df


class TestRegimeSwitch:
    """Test leveraged pair switching based on market regime."""

    @pytest.mark.asyncio
    async def test_bull_regime_buys_bull_etfs(self, engine, mock_order_manager, mock_market_data):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)

        state = MarketState(regime=MarketRegime.UPTREND, spy_price=500, spy_sma200=480, confidence=0.8)
        actions = await engine._manage_regime_etfs(state)

        assert len(actions) >= 1
        assert any("BUY" in a for a in actions)
        assert mock_order_manager.place_buy.called

    @pytest.mark.asyncio
    async def test_bear_regime_exits_to_cash_when_not_qualified(self, engine, mock_order_manager, mock_market_data):
        # First set bull regime
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        bull_state = MarketState(regime=MarketRegime.UPTREND)
        await engine._manage_regime_etfs(bull_state)

        # Switch to bear with SPY barely below SMA200 (not qualified for bear entry)
        pos_tqqq = MagicMock(symbol="TQQQ", quantity=100, current_price=50.0)
        mock_market_data.get_positions.return_value = [pos_tqqq]

        # spy_distance_pct=-1.0 doesn't meet -5.0 threshold → exit to cash
        bear_state = MarketState(
            regime=MarketRegime.DOWNTREND,
            spy_distance_pct=-1.0, confidence=0.5,
        )
        actions = await engine._manage_regime_etfs(bear_state)

        assert any("SELL" in a and "TQQQ" in a for a in actions)
        assert not any("BUY" in a and "SQQQ" in a for a in actions)

    @pytest.mark.asyncio
    async def test_bear_regime_buys_bear_when_qualified(self, engine, mock_order_manager, mock_market_data):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)

        # Set bull then switch to qualified bear
        await engine._manage_regime_etfs(MarketState(regime=MarketRegime.UPTREND))
        mock_market_data.get_positions.return_value = []

        # SPY -6% below SMA200, high confidence → qualified for bear entry
        bear_state = MarketState(
            regime=MarketRegime.DOWNTREND,
            spy_distance_pct=-6.0, confidence=0.85,
        )
        actions = await engine._manage_regime_etfs(bear_state)
        assert any("BUY" in a for a in actions)
        assert any("half-size" in a for a in actions)

    @pytest.mark.asyncio
    async def test_sideways_exits_all_leveraged(self, engine, mock_order_manager, mock_market_data):
        # First enter bull regime
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        await engine._manage_regime_etfs(MarketState(regime=MarketRegime.UPTREND))

        # Simulate holding TQQQ
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0)
        mock_market_data.get_positions.return_value = [pos]

        # Switch to sideways
        actions = await engine._manage_regime_etfs(MarketState(regime=MarketRegime.SIDEWAYS))
        assert any("SELL" in a for a in actions)

    @pytest.mark.asyncio
    async def test_no_action_on_same_regime(self, engine, mock_market_data):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        state = MarketState(regime=MarketRegime.UPTREND)

        await engine._manage_regime_etfs(state)
        actions = await engine._manage_regime_etfs(state)
        assert actions == []

    @pytest.mark.asyncio
    async def test_regime_switch_sends_notification(self, engine, mock_market_data, mock_notification):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        state = MarketState(regime=MarketRegime.UPTREND)
        await engine._manage_regime_etfs(state)

        mock_notification.notify_system_event.assert_called_once()
        call_args = mock_notification.notify_system_event.call_args
        assert call_args[0][0] == "etf_regime_switch"


class TestSectorRotation:
    """Test sector ETF rotation based on sector strength."""

    @pytest.mark.asyncio
    async def test_buys_top_sector_etfs(self, engine, mock_order_manager, mock_market_data):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(150.0)

        sector_data = {
            "Technology": {"symbol": "XLK", "return_1w": 3.0, "return_1m": 8.0, "return_3m": 15.0},
            "Energy": {"symbol": "XLE", "return_1w": -2.0, "return_1m": -5.0, "return_3m": -8.0},
            "Financials": {"symbol": "XLF", "return_1w": 1.0, "return_1m": 4.0, "return_3m": 10.0},
        }
        actions = await engine._manage_sector_etfs(sector_data)
        assert len(actions) >= 1
        assert mock_order_manager.place_buy.called

    @pytest.mark.asyncio
    async def test_sells_weak_sector_etfs(self, engine, mock_order_manager, mock_market_data):
        # Simulate holding XLE (weak sector)
        pos = MagicMock(symbol="XLE", quantity=50, current_price=80.0)
        mock_market_data.get_positions.return_value = [pos]

        sector_data = {
            "Technology": {"symbol": "XLK", "return_1w": 3.0, "return_1m": 8.0, "return_3m": 15.0},
            "Energy": {"symbol": "XLE", "return_1w": -3.0, "return_1m": -6.0, "return_3m": -10.0},
            "Financials": {"symbol": "XLF", "return_1w": 2.0, "return_1m": 5.0, "return_3m": 12.0},
        }
        actions = await engine._manage_sector_etfs(sector_data)
        # XLE should be sold as bottom sector
        sell_actions = [a for a in actions if "SELL" in a]
        assert len(sell_actions) >= 1

    @pytest.mark.asyncio
    async def test_no_action_when_sectors_unchanged(self, engine, mock_market_data):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(150.0)

        sector_data = {
            "Technology": {"symbol": "XLK", "return_1w": 3.0, "return_1m": 8.0, "return_3m": 15.0},
            "Energy": {"symbol": "XLE", "return_1w": -2.0, "return_1m": -5.0, "return_3m": -8.0},
            "Financials": {"symbol": "XLF", "return_1w": 1.0, "return_1m": 4.0, "return_3m": 10.0},
        }
        await engine._manage_sector_etfs(sector_data)
        actions = await engine._manage_sector_etfs(sector_data)
        assert actions == []


class TestHoldLimits:
    """Test max hold days enforcement for leveraged ETFs."""

    @pytest.mark.asyncio
    async def test_sells_expired_leveraged_position(self, engine, mock_order_manager, mock_market_data):
        # Add a position held for 15 days (exceeds 10-day limit)
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 15 * 86400,
            reason="regime_bull",
        )

        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0)
        mock_market_data.get_positions.return_value = [pos]

        actions = await engine._check_hold_limits()
        assert len(actions) == 1
        assert "TQQQ" in actions[0]
        assert mock_order_manager.place_sell.called

    @pytest.mark.asyncio
    async def test_keeps_non_expired_position(self, engine, mock_market_data):
        # Position held for 5 days (within 10-day limit)
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 5 * 86400,
            reason="regime_bull",
        )

        actions = await engine._check_hold_limits()
        assert actions == []

    @pytest.mark.asyncio
    async def test_non_leveraged_etf_exempt_from_hold_limit(self, engine, mock_market_data):
        # XLK is not leveraged — should not trigger hold limit
        engine._managed_positions["XLK"] = ETFPosition(
            symbol="XLK",
            entry_date=time.time() - 30 * 86400,
            reason="sector_rotation",
        )

        actions = await engine._check_hold_limits()
        assert actions == []


class TestExposureLimits:
    """Test portfolio exposure limit checks."""

    @pytest.mark.asyncio
    async def test_warns_on_high_etf_exposure(self, engine, mock_market_data):
        # Total portfolio = 100k, ETF positions = 40k (40% > 30% limit)
        pos1 = MagicMock(symbol="TQQQ", quantity=200, current_price=100.0)
        pos2 = MagicMock(symbol="XLK", quantity=200, current_price=100.0)
        mock_market_data.get_positions.return_value = [pos1, pos2]

        actions = await engine._check_exposure_limits()
        assert len(actions) == 1
        assert "exceeds" in actions[0]

    @pytest.mark.asyncio
    async def test_no_warning_within_limits(self, engine, mock_market_data):
        # Total = 100k, ETF = 10k (10% < 30% limit)
        pos = MagicMock(symbol="TQQQ", quantity=20, current_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        actions = await engine._check_exposure_limits()
        assert actions == []


class TestFullEvaluation:
    """Test full evaluate() flow."""

    @pytest.mark.asyncio
    async def test_full_evaluate(self, engine, mock_market_data):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)

        state = MarketState(regime=MarketRegime.UPTREND, spy_price=500, confidence=0.8)
        sector_data = {
            "Technology": {"symbol": "XLK", "return_1w": 3.0, "return_1m": 8.0, "return_3m": 15.0},
            "Energy": {"symbol": "XLE", "return_1w": -2.0, "return_1m": -5.0, "return_3m": -8.0},
            "Financials": {"symbol": "XLF", "return_1w": 1.0, "return_1m": 4.0, "return_3m": 10.0},
        }

        result = await engine.evaluate(state, sector_data)
        assert "regime" in result
        assert "sector" in result
        assert "risk" in result


class TestStatus:
    """Test get_status()."""

    def test_initial_status(self, engine):
        status = engine.get_status()
        assert status["last_regime"] is None
        assert status["top_sectors"] == []
        assert status["managed_positions"] == {}

    @pytest.mark.asyncio
    async def test_status_after_regime_switch(self, engine, mock_market_data):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        await engine._manage_regime_etfs(MarketState(regime=MarketRegime.UPTREND))

        status = engine.get_status()
        assert status["last_regime"] == "uptrend"
