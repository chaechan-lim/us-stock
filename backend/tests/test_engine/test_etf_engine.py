"""Tests for ETF Engine."""

import time
from datetime import UTC, datetime, timedelta

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
        engine._last_regime = MarketRegime.SIDEWAYS  # simulate prior regime

        state = MarketState(regime=MarketRegime.UPTREND, spy_price=500, spy_sma200=480, confidence=0.8)
        actions = await engine._manage_regime_etfs(state)

        assert len(actions) >= 1
        assert any("BUY" in a for a in actions)
        assert mock_order_manager.place_buy.called

    @pytest.mark.asyncio
    async def test_bear_regime_exits_to_cash_when_not_qualified(self, engine, mock_order_manager, mock_market_data):
        # First set bull regime
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        engine._last_regime = MarketRegime.SIDEWAYS  # simulate prior regime
        bull_state = MarketState(regime=MarketRegime.UPTREND)
        await engine._manage_regime_etfs(bull_state)

        # Manually set TQQQ entry to be old enough for min_hold (4h)
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 5 * 3600,  # 5 hours ago
            reason="regime_bull",
        )

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
        engine._last_regime = MarketRegime.SIDEWAYS  # simulate prior regime

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
        engine._last_regime = MarketRegime.SIDEWAYS  # simulate prior regime
        await engine._manage_regime_etfs(MarketState(regime=MarketRegime.UPTREND))

        # Manually set TQQQ entry to be old enough for min_hold (4h)
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 5 * 3600,  # 5 hours ago
            reason="regime_bull",
        )

        # Simulate holding TQQQ
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0)
        mock_market_data.get_positions.return_value = [pos]

        # Switch to sideways
        actions = await engine._manage_regime_etfs(MarketState(regime=MarketRegime.SIDEWAYS))
        assert any("SELL" in a for a in actions)

    @pytest.mark.asyncio
    async def test_no_action_on_same_regime(self, engine, mock_market_data):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        engine._last_regime = MarketRegime.SIDEWAYS  # simulate prior regime
        state = MarketState(regime=MarketRegime.UPTREND)

        await engine._manage_regime_etfs(state)
        actions = await engine._manage_regime_etfs(state)
        assert actions == []

    @pytest.mark.asyncio
    async def test_regime_switch_sends_notification(self, engine, mock_market_data, mock_notification):
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        engine._last_regime = MarketRegime.SIDEWAYS  # simulate prior regime
        state = MarketState(regime=MarketRegime.UPTREND)
        await engine._manage_regime_etfs(state)

        mock_notification.notify_system_event.assert_called_once()
        call_args = mock_notification.notify_system_event.call_args
        assert call_args[0][0] == "etf_regime_switch"


class TestMutualExclusivity:
    """Test 1x/2x ETF mutual exclusivity enforcement."""

    @pytest.mark.asyncio
    async def test_sells_base_etf_before_buying_leveraged(
        self, engine, mock_order_manager, mock_market_data, mock_etf_universe,
    ):
        """When buying TQQQ (bull), sell QQQ (base) if held."""
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        engine._last_regime = MarketRegime.SIDEWAYS  # simulate prior regime

        # Simulate holding QQQ (base ETF)
        pos_qqq = MagicMock(symbol="QQQ", quantity=50, current_price=400.0)
        mock_market_data.get_positions.return_value = [pos_qqq]

        # Configure get_pair_siblings: TQQQ → [QQQ, SQQQ]
        mock_etf_universe.get_pair_siblings.side_effect = lambda s: {
            "TQQQ": ["QQQ", "SQQQ"],
            "SOXL": ["SOXX", "SOXS"],
            "QQQ": ["TQQQ", "SQQQ"],
        }.get(s, [])

        state = MarketState(
            regime=MarketRegime.UPTREND, spy_price=500,
            spy_sma200=480, confidence=0.8,
        )
        actions = await engine._manage_regime_etfs(state)

        # QQQ should be sold before buying TQQQ
        sell_calls = [
            c for c in mock_order_manager.place_sell.call_args_list
            if c.kwargs.get("symbol") == "QQQ" or (c.args and c.args[0] == "QQQ")
        ]
        assert len(sell_calls) >= 1, "Should sell QQQ (sibling) before buying TQQQ"
        assert any("mutual exclusivity" in a for a in actions)

    @pytest.mark.asyncio
    async def test_no_double_sell_for_already_exited_sibling(
        self, engine, mock_order_manager, mock_market_data, mock_etf_universe,
    ):
        """If sibling is already in exit_etfs, don't sell twice."""
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)
        engine._last_regime = MarketRegime.SIDEWAYS  # simulate prior regime

        # First set bull regime
        mock_etf_universe.get_pair_siblings.return_value = []
        await engine._manage_regime_etfs(
            MarketState(regime=MarketRegime.UPTREND)
        )

        # Manually set TQQQ entry to be old enough for min_hold (4h)
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 5 * 3600,  # 5 hours ago
            reason="regime_bull",
        )

        # Now switch to bear: SQQQ siblings include TQQQ
        pos_tqqq = MagicMock(symbol="TQQQ", quantity=100, current_price=50.0)
        mock_market_data.get_positions.return_value = [pos_tqqq]
        mock_etf_universe.get_pair_siblings.side_effect = lambda s: {
            "SQQQ": ["QQQ", "TQQQ"],
            "SOXS": ["SOXX", "SOXL"],
        }.get(s, [])

        # TQQQ is already in exit_etfs (bull→bear transition), so shouldn't
        # be sold again via mutual exclusivity
        bear_state = MarketState(
            regime=MarketRegime.DOWNTREND,
            spy_distance_pct=-6.0, confidence=0.85,
        )
        actions = await engine._manage_regime_etfs(bear_state)

        # TQQQ sell should only appear once (from exit_etfs, not mutual exclusivity)
        tqqq_sells = [a for a in actions if "SELL" in a and "TQQQ" in a]
        assert len(tqqq_sells) == 1

        # After TQQQ is cleared, SQQQ buy should proceed (exit sell must not block it)
        sqqq_buys = [a for a in actions if "BUY" in a and "SQQQ" in a]
        assert len(sqqq_buys) == 1, "SQQQ should be bought after TQQQ sibling is sold"

    @pytest.mark.asyncio
    async def test_skip_buy_when_sibling_min_hold_blocks_sell(
        self, engine, mock_order_manager, mock_market_data, mock_etf_universe, caplog,
    ):
        """If sibling min_hold prevents its sell, the target buy must also be skipped.

        Without this guard, the engine can simultaneously hold TQQQ + SQQQ (both bull
        and bear), violating mutual exclusivity and amplifying decay losses.
        """
        import logging

        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)

        # TQQQ held for only 1 hour — min_hold (4h) not satisfied
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 1 * 3600,  # 1 hour, too new to sell
            reason="regime_bull",
        )
        engine._last_regime = MarketRegime.UPTREND

        pos_tqqq = MagicMock(symbol="TQQQ", quantity=100, current_price=50.0)
        mock_market_data.get_positions.return_value = [pos_tqqq]

        # Bear switch: wants to buy SQQQ, but must sell TQQQ first (sibling)
        mock_etf_universe.get_pair_siblings.side_effect = lambda s: {
            "SQQQ": ["QQQ", "TQQQ"],
            "SOXS": ["SOXX", "SOXL"],
        }.get(s, [])

        bear_state = MarketState(
            regime=MarketRegime.DOWNTREND,
            spy_distance_pct=-6.0, confidence=0.85,
        )
        with caplog.at_level(logging.INFO):
            actions = await engine._manage_regime_etfs(bear_state)

        # TQQQ sell blocked by min_hold, SQQQ buy must also be skipped.
        # (SOXS may still be bought since it has no conflicting sibling held.)
        assert not any("SELL" in a and "TQQQ" in a for a in actions), (
            "TQQQ should not be sold — min_hold constraint not satisfied"
        )
        assert not any("BUY" in a and "SQQQ" in a for a in actions), (
            "SQQQ should not be bought — sibling TQQQ could not be cleared first"
        )
        sqqq_buy_calls = [
            c for c in mock_order_manager.place_buy.call_args_list
            if c.kwargs.get("symbol") == "SQQQ" or (c.args and c.args[0] == "SQQQ")
        ]
        assert len(sqqq_buy_calls) == 0, "place_buy should not have been called for SQQQ"
        # Positive control: the guard code path must have been reached.
        # Without this, the assertion above could pass vacuously if SQQQ was never
        # a buy candidate in the first place (e.g. missing from the DOWNTREND list).
        assert any(
            "SKIP BUY SQQQ" in r.getMessage() and "TQQQ" in r.getMessage()
            for r in caplog.records
        ), "Expected 'SKIP BUY SQQQ' log — proves mutual-exclusivity guard was reached"


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


class TestMinHoldConstraint:
    """Test minimum hold duration enforcement before selling."""

    @pytest.fixture(autouse=True)
    def configure_is_leveraged(self, mock_etf_universe):
        """Explicitly configure is_leveraged so 4h (leveraged) vs 2h (sector) threshold
        dispatch is self-documenting and independent of changes to the shared fixture."""
        mock_etf_universe.is_leveraged.side_effect = (
            lambda s: s in {"TQQQ", "SQQQ", "SOXL", "SOXS"}
        )

    @pytest.mark.asyncio
    async def test_skip_sell_if_min_hold_not_satisfied_leveraged(
        self, engine, mock_order_manager, mock_market_data,
    ):
        """Leveraged ETF held < min_hold_leveraged_hours should not be sold."""
        # Add TQQQ held for only 1 hour (min is 4h)
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 1 * 3600,  # 1 hour ago
            reason="regime_bull",
        )

        pos_tqqq = MagicMock(symbol="TQQQ", quantity=100, current_price=50.0, avg_price=48.0)
        mock_market_data.get_positions.return_value = [pos_tqqq]
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)

        engine._last_regime = MarketRegime.UPTREND
        # Switch to sideways → should exit TQQQ but will be blocked by min_hold
        state = MarketState(regime=MarketRegime.SIDEWAYS)
        actions = await engine._manage_regime_etfs(state)

        # TQQQ should NOT be sold because it's too new
        assert not any("SELL" in a and "TQQQ" in a for a in actions)
        assert mock_order_manager.place_sell.call_count == 0

    @pytest.mark.asyncio
    async def test_sell_allowed_after_min_hold_satisfied_leveraged(
        self, engine, mock_order_manager, mock_market_data,
    ):
        """Leveraged ETF held >= min_hold_leveraged_hours should be sold."""
        # Add TQQQ held for 5 hours (min is 4h) ✓
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 5 * 3600,  # 5 hours ago
            reason="regime_bull",
        )

        pos_tqqq = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=48.0)
        mock_market_data.get_positions.return_value = [pos_tqqq]
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)

        engine._last_regime = MarketRegime.UPTREND
        state = MarketState(regime=MarketRegime.SIDEWAYS)
        actions = await engine._manage_regime_etfs(state)

        # TQQQ should be sold because min_hold is satisfied
        assert any("SELL" in a and "TQQQ" in a for a in actions)
        assert mock_order_manager.place_sell.called

    @pytest.mark.asyncio
    async def test_skip_sell_if_min_hold_not_satisfied_sector(
        self, engine, mock_order_manager, mock_market_data,
    ):
        """Sector ETF held < min_hold_sector_hours should not be sold."""
        # Add XLE (sector) held for only 1 hour (min is 2h)
        engine._managed_positions["XLE"] = ETFPosition(
            symbol="XLE",
            entry_date=time.time() - 1 * 3600,  # 1 hour ago
            reason="sector_rotation",
            sector="Energy",
        )

        pos_xle = MagicMock(symbol="XLE", quantity=50, current_price=80.0, avg_price=78.0)
        mock_market_data.get_positions.return_value = [pos_xle]
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(150.0)

        sector_data = {
            "Technology": {"symbol": "XLK", "return_1w": 3.0, "return_1m": 8.0, "return_3m": 15.0},
            "Energy": {"symbol": "XLE", "return_1w": -3.0, "return_1m": -6.0, "return_3m": -10.0},
            "Financials": {"symbol": "XLF", "return_1w": 2.0, "return_1m": 5.0, "return_3m": 12.0},
        }
        actions = await engine._manage_sector_etfs(sector_data)

        # XLE should NOT be sold because it's too new, even though it's in bottom
        assert not any("SELL" in a and "XLE" in a for a in actions)

    @pytest.mark.asyncio
    async def test_sell_allowed_after_min_hold_satisfied_sector(
        self, engine, mock_order_manager, mock_market_data,
    ):
        """Sector ETF held >= min_hold_sector_hours should be sold."""
        # Add XLE (sector) held for 3 hours (min is 2h) ✓
        engine._managed_positions["XLE"] = ETFPosition(
            symbol="XLE",
            entry_date=time.time() - 3 * 3600,  # 3 hours ago
            reason="sector_rotation",
            sector="Energy",
        )

        pos_xle = MagicMock(symbol="XLE", quantity=50, current_price=80.0, avg_price=78.0)
        mock_market_data.get_positions.return_value = [pos_xle]
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(150.0)

        sector_data = {
            "Technology": {"symbol": "XLK", "return_1w": 3.0, "return_1m": 8.0, "return_3m": 15.0},
            "Energy": {"symbol": "XLE", "return_1w": -3.0, "return_1m": -6.0, "return_3m": -10.0},
            "Financials": {"symbol": "XLF", "return_1w": 2.0, "return_1m": 5.0, "return_3m": 12.0},
        }
        before_sell = time.time()
        actions = await engine._manage_sector_etfs(sector_data)

        # XLE should be sold because min_hold is satisfied
        assert any("SELL" in a and "XLE" in a for a in actions)
        assert mock_order_manager.place_sell.called
        assert "XLE" in engine._last_sell_times, "sector ETF sell must record cooldown timestamp"
        assert engine._last_sell_times["XLE"] >= before_sell


class TestSellCooldownConstraint:
    """Test sell cooldown enforcement to prevent rapid rebuy."""

    @pytest.mark.asyncio
    async def test_skip_buy_if_cooldown_not_satisfied(
        self, engine, mock_order_manager, mock_market_data,
    ):
        """ETF sold < sell_cooldown_hours ago should not be bought."""
        # Simulate TQQQ sold 1 hour ago (cooldown is 4h)
        engine._last_sell_times["TQQQ"] = time.time() - 1 * 3600

        mock_market_data.get_positions.return_value = []
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)

        engine._last_regime = MarketRegime.SIDEWAYS
        state = MarketState(regime=MarketRegime.UPTREND, spy_price=500, confidence=0.8)
        actions = await engine._manage_regime_etfs(state)

        # TQQQ should NOT be bought because cooldown period not satisfied
        assert not any("BUY" in a and "TQQQ" in a for a in actions)

    @pytest.mark.asyncio
    async def test_buy_allowed_after_cooldown_satisfied(
        self, engine, mock_order_manager, mock_market_data,
    ):
        """ETF sold >= sell_cooldown_hours ago should be buyable."""
        # Simulate TQQQ sold 5 hours ago (cooldown is 4h) ✓
        engine._last_sell_times["TQQQ"] = time.time() - 5 * 3600

        mock_market_data.get_positions.return_value = []
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)

        engine._last_regime = MarketRegime.SIDEWAYS
        state = MarketState(regime=MarketRegime.UPTREND, spy_price=500, confidence=0.8)
        actions = await engine._manage_regime_etfs(state)

        # TQQQ should be bought because cooldown is satisfied
        assert any("BUY" in a and "TQQQ" in a for a in actions)
        call_symbols = [
            c.kwargs.get("symbol") or (c.args[0] if c.args else None)
            for c in mock_order_manager.place_buy.call_args_list
        ]
        assert "TQQQ" in call_symbols, f"Expected TQQQ buy call, got: {call_symbols}"

    @pytest.mark.asyncio
    async def test_sell_records_cooldown_timer(
        self, engine, mock_order_manager, mock_market_data,
    ):
        """Successful sell should record time for cooldown tracking."""
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 5 * 3600,  # 5 hours, min_hold satisfied
            reason="regime_bull",
        )

        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=48.0)
        mock_market_data.get_positions.return_value = [pos]
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(50.0)

        engine._last_regime = MarketRegime.UPTREND
        before_sell = time.time()
        await engine._manage_regime_etfs(MarketState(regime=MarketRegime.SIDEWAYS))
        after_sell = time.time()

        # Check that sell time was recorded
        assert "TQQQ" in engine._last_sell_times
        sell_time = engine._last_sell_times["TQQQ"]
        assert before_sell <= sell_time <= after_sell

    @pytest.mark.asyncio
    async def test_sector_etf_cooldown_constraint(
        self, engine, mock_order_manager, mock_market_data,
    ):
        """Sector ETF cooldown should prevent rapid rebuy."""
        # Simulate XLE sold 1 hour ago (cooldown is 4h)
        engine._last_sell_times["XLE"] = time.time() - 1 * 3600

        mock_market_data.get_positions.return_value = []
        mock_market_data.get_ohlcv.return_value = _make_ohlcv_mock(150.0)

        sector_data = {
            "Technology": {"symbol": "XLK", "return_1w": 1.0, "return_1m": 4.0, "return_3m": 10.0},
            "Energy": {"symbol": "XLE", "return_1w": 3.0, "return_1m": 8.0, "return_3m": 15.0},
            "Financials": {"symbol": "XLF", "return_1w": 2.0, "return_1m": 5.0, "return_3m": 12.0},
        }
        actions = await engine._manage_sector_etfs(sector_data)

        # XLE should NOT be bought even though it's in top because cooldown not satisfied
        assert not any("BUY" in a and "XLE" in a for a in actions)


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

    @pytest.mark.asyncio
    async def test_sell_failure_retains_tracking(
        self, engine, mock_order_manager, mock_market_data
    ):
        """When place_sell returns None, tracking must NOT be removed so retry is possible."""
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 15 * 86400,
            reason="regime_bull",
        )

        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0)
        mock_market_data.get_positions.return_value = [pos]
        # Simulate sell order failure
        mock_order_manager.place_sell.return_value = None

        actions = await engine._check_hold_limits()

        # Sell was attempted
        assert mock_order_manager.place_sell.called
        # No action logged (sell did not complete)
        assert actions == []
        # Tracking must still be present for retry
        assert "TQQQ" in engine._managed_positions

    @pytest.mark.asyncio
    async def test_sell_success_removes_tracking(
        self, engine, mock_order_manager, mock_market_data
    ):
        """When place_sell succeeds, tracking should be removed normally."""
        engine._managed_positions["TQQQ"] = ETFPosition(
            symbol="TQQQ",
            entry_date=time.time() - 15 * 86400,
            reason="regime_bull",
        )

        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0)
        mock_market_data.get_positions.return_value = [pos]
        # Simulate successful sell order
        mock_order_manager.place_sell.return_value = MagicMock()

        before_sell = time.time()
        actions = await engine._check_hold_limits()

        assert mock_order_manager.place_sell.called
        assert len(actions) == 1
        assert "TQQQ" in actions[0]
        # Tracking removed after successful sell
        assert "TQQQ" not in engine._managed_positions
        # Cooldown timestamp must be recorded so rapid rebuy is blocked
        assert "TQQQ" in engine._last_sell_times
        assert engine._last_sell_times["TQQQ"] >= before_sell


class TestExposureLimits:
    """Test portfolio exposure limit checks."""

    def test_warns_on_high_etf_exposure(self, engine, mock_market_data):
        # Total portfolio = 100k, ETF positions = 40k (40% > 30% limit)
        pos1 = MagicMock(symbol="TQQQ", quantity=200, current_price=100.0)
        pos2 = MagicMock(symbol="XLK", quantity=200, current_price=100.0)
        positions = [pos1, pos2]
        balance = MagicMock(total=100000, available=50000)

        actions = engine._check_exposure_limits(positions, balance)
        assert len(actions) == 1
        assert "exceeds" in actions[0]

    def test_no_warning_within_limits(self, engine, mock_market_data):
        # Total = 100k, ETF = 10k (10% < 30% limit)
        pos = MagicMock(symbol="TQQQ", quantity=20, current_price=50.0)
        positions = [pos]
        balance = MagicMock(total=100000, available=50000)

        actions = engine._check_exposure_limits(positions, balance)
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


# --- Helper for DB mock ---

def _mock_session_factory(orders=None):
    """Create a mock async session factory that returns given orders.

    Uses SQLAlchemy statement introspection instead of brittle string matching
    to extract the symbol from WHERE clauses and validate filter conditions.

    Args:
        orders: dict mapping symbol -> mock Order, or None for no results.
    """
    orders = orders or {}

    class FakeResult:
        def __init__(self, order):
            self._order = order

        def scalar_one_or_none(self):
            return self._order

    class FakeSession:
        async def execute(self, stmt):
            # Extract bind parameters from the compiled statement.
            # This avoids brittle string matching and properly inspects
            # the SQLAlchemy query's WHERE clause parameters.
            compiled = stmt.compile(compile_kwargs={"literal_binds": True})
            stmt_str = str(compiled)

            # Validate is_paper filter is present (critical for live/paper isolation)
            assert "is_paper" in stmt_str, (
                "Query missing is_paper filter — live orders must filter "
                "out paper trades. See position_tracker.restore_from_exchange() "
                "for the correct pattern."
            )

            for sym, order in orders.items():
                if f"'{sym}'" in stmt_str:
                    return FakeResult(order)
            return FakeResult(None)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    def factory():
        return FakeSession()

    return factory


def _make_order_mock(symbol: str, strategy_name: str, created_at: datetime | None = None):
    """Create a mock Order for DB restore tests."""
    order = MagicMock()
    order.symbol = symbol
    order.strategy_name = strategy_name
    order.created_at = created_at or datetime.now(UTC)
    order.side = "BUY"
    order.status = "filled"
    return order


def _make_sell_order_mock(symbol: str, created_at: datetime | None = None):
    """Create a mock SELL Order for cooldown-seeding tests."""
    order = MagicMock()
    order.symbol = symbol
    order.created_at = created_at or datetime.now(UTC)
    order.side = "SELL"
    order.status = "filled"
    return order


def _mock_sell_session_factory(sell_orders=None):
    """Create a mock async session factory for cooldown-seeding tests.

    Discriminates between the BUY-entry query and the SELL-cooldown query by
    inspecting the compiled SQL for the `side` literal value.  SQL inspection
    is more stable than call-count discrimination: it survives refactors that
    change how many sessions are opened per block without silently misrouting
    results.

    BUY query  → scalar_one_or_none() = None (no entry order on record)
    SELL query → scalars().all()      = sell_orders
    """
    sell_orders = sell_orders or []

    class _FakeScalars:
        def __init__(self, orders):
            self._orders = orders

        def all(self):
            return self._orders

    class _BuyResult:
        def scalar_one_or_none(self):
            return None  # no BUY order on record → restore uses inferred defaults

    class _SellResult:
        def scalars(self):
            return _FakeScalars(sell_orders)

    class _Session:
        async def execute(self, stmt):
            sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
            assert "is_paper" in sql, (
                "Query missing is_paper filter — live/paper isolation required."
            )
            if "'SELL'" in sql:
                return _SellResult()
            return _BuyResult()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    def factory():
        return _Session()

    return factory


class TestRestoreManagedPositions:
    """Test restore_managed_positions() for server restart recovery."""

    @pytest.mark.asyncio
    async def test_restore_leveraged_etf_from_broker(self, engine, mock_market_data):
        """Leveraged ETF on broker → restored as regime position."""
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        restored = await engine.restore_managed_positions()

        assert len(restored) == 1
        assert restored[0]["symbol"] == "TQQQ"
        assert restored[0]["reason"] == "regime_restored"
        assert "TQQQ" in engine._managed_positions
        assert engine._managed_positions["TQQQ"].reason == "regime_restored"

    @pytest.mark.asyncio
    async def test_restore_sector_etf_from_broker(self, engine, mock_market_data):
        """Sector ETF on broker → restored as sector_rotation with sector name."""
        pos = MagicMock(symbol="XLK", quantity=50, current_price=180.0, avg_price=170.0)
        mock_market_data.get_positions.return_value = [pos]

        restored = await engine.restore_managed_positions()

        assert len(restored) == 1
        assert restored[0]["symbol"] == "XLK"
        assert restored[0]["reason"] == "sector_rotation"
        assert restored[0]["sector"] == "Technology"
        assert engine._managed_positions["XLK"].sector == "Technology"

    @pytest.mark.asyncio
    async def test_restore_with_db_order_metadata(self, engine, mock_market_data):
        """When DB has order record, use strategy_name and created_at."""
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        buy_time = datetime.now(UTC) - timedelta(days=5)
        order = _make_order_mock("TQQQ", "etf_engine_regime", buy_time)
        session_factory = _mock_session_factory({"TQQQ": order})

        restored = await engine.restore_managed_positions(session_factory)

        assert len(restored) == 1
        assert restored[0]["source"] == "exchange"
        # Entry date should match the order's created_at (within 1 second)
        entry_ts = engine._managed_positions["TQQQ"].entry_date
        assert abs(entry_ts - buy_time.timestamp()) < 1.0

    @pytest.mark.asyncio
    async def test_restore_sector_with_db_order(self, engine, mock_market_data):
        """Sector ETF with DB order uses sector strategy mapping."""
        pos = MagicMock(symbol="XLE", quantity=30, current_price=90.0, avg_price=85.0)
        mock_market_data.get_positions.return_value = [pos]

        order = _make_order_mock("XLE", "etf_engine_sector")
        session_factory = _mock_session_factory({"XLE": order})

        restored = await engine.restore_managed_positions(session_factory)

        assert len(restored) == 1
        assert restored[0]["reason"] == "sector_rotation"
        assert engine._managed_positions["XLE"].reason == "sector_rotation"

    @pytest.mark.asyncio
    async def test_restore_multiple_positions(self, engine, mock_market_data):
        """Multiple ETF positions restored simultaneously."""
        pos1 = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        pos2 = MagicMock(symbol="XLK", quantity=50, current_price=180.0, avg_price=170.0)
        pos3 = MagicMock(symbol="XLE", quantity=30, current_price=90.0, avg_price=85.0)
        mock_market_data.get_positions.return_value = [pos1, pos2, pos3]

        restored = await engine.restore_managed_positions()

        assert len(restored) == 3
        symbols = {r["symbol"] for r in restored}
        assert symbols == {"TQQQ", "XLK", "XLE"}
        assert len(engine._managed_positions) == 3

    @pytest.mark.asyncio
    async def test_restore_ignores_non_etf_positions(self, engine, mock_market_data):
        """Non-ETF positions (regular stocks) are ignored."""
        pos_etf = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        pos_stock = MagicMock(symbol="AAPL", quantity=10, current_price=180.0, avg_price=170.0)
        mock_market_data.get_positions.return_value = [pos_etf, pos_stock]

        restored = await engine.restore_managed_positions()

        assert len(restored) == 1
        assert restored[0]["symbol"] == "TQQQ"
        assert "AAPL" not in engine._managed_positions

    @pytest.mark.asyncio
    async def test_restore_idempotent(self, engine, mock_market_data):
        """Calling restore twice doesn't duplicate positions."""
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        restored1 = await engine.restore_managed_positions()
        restored2 = await engine.restore_managed_positions()

        assert len(restored1) == 1
        assert len(restored2) == 1
        assert restored2[0]["source"] == "already_tracked"
        assert len(engine._managed_positions) == 1

    @pytest.mark.asyncio
    async def test_restore_empty_broker(self, engine, mock_market_data):
        """No positions on broker → empty restore."""
        mock_market_data.get_positions.return_value = []

        restored = await engine.restore_managed_positions()

        assert restored == []
        assert len(engine._managed_positions) == 0

    @pytest.mark.asyncio
    async def test_restore_broker_error_graceful(self, engine, mock_market_data):
        """Broker API error → graceful empty return."""
        mock_market_data.get_positions.side_effect = Exception("API timeout")

        restored = await engine.restore_managed_positions()

        assert restored == []
        assert len(engine._managed_positions) == 0

    @pytest.mark.asyncio
    async def test_restored_position_hold_limit_works(
        self, engine, mock_market_data, mock_order_manager,
    ):
        """Restored leveraged position with old entry_date triggers hold limit sell."""
        # Restore a position that was bought 15 days ago
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        buy_time = datetime.now(UTC) - timedelta(days=15)
        order = _make_order_mock("TQQQ", "etf_engine_regime", buy_time)
        session_factory = _mock_session_factory({"TQQQ": order})

        await engine.restore_managed_positions(session_factory)

        # Now check hold limits — should trigger sell (15 days > 10 day limit)
        actions = await engine._check_hold_limits()
        assert len(actions) == 1
        assert "TQQQ" in actions[0]
        assert mock_order_manager.place_sell.called

    @pytest.mark.asyncio
    async def test_restored_position_in_status(self, engine, mock_market_data):
        """Restored positions appear in get_status()."""
        pos = MagicMock(symbol="XLE", quantity=30, current_price=90.0, avg_price=85.0)
        mock_market_data.get_positions.return_value = [pos]

        await engine.restore_managed_positions()

        status = engine.get_status()
        assert "XLE" in status["managed_positions"]
        assert status["managed_positions"]["XLE"]["reason"] == "sector_rotation"
        assert status["managed_positions"]["XLE"]["sector"] == "Energy"

    @pytest.mark.asyncio
    async def test_restore_zero_quantity_skipped(self, engine, mock_market_data):
        """Positions with 0 quantity are skipped."""
        pos = MagicMock(symbol="TQQQ", quantity=0, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        restored = await engine.restore_managed_positions()

        assert restored == []
        assert len(engine._managed_positions) == 0

    @pytest.mark.asyncio
    async def test_restore_with_db_failure_falls_back_to_inference(
        self, engine, mock_market_data,
    ):
        """DB exception doesn't crash restore — falls back to ETF type inference."""
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        # Session factory that raises on execute
        class FailingSession:
            async def execute(self, stmt):
                raise RuntimeError("DB connection lost")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        def failing_factory():
            return FailingSession()

        restored = await engine.restore_managed_positions(failing_factory)

        # Should still restore via inference (leveraged → regime_restored)
        assert len(restored) == 1
        assert restored[0]["symbol"] == "TQQQ"
        assert restored[0]["reason"] == "regime_restored"
        assert restored[0]["source"] == "inferred"
        assert "TQQQ" in engine._managed_positions

    @pytest.mark.asyncio
    async def test_restore_naive_datetime_treated_as_utc(self, engine, mock_market_data):
        """Naive datetime from DB is treated as UTC, not local timezone."""
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        # Create a naive datetime (no tzinfo) — simulates some DB drivers
        naive_dt = datetime(2026, 3, 10, 14, 0, 0)  # no tzinfo
        order = _make_order_mock("TQQQ", "etf_engine_regime", naive_dt)
        # Override created_at to be truly naive (MagicMock default uses UTC)
        order.created_at = naive_dt
        session_factory = _mock_session_factory({"TQQQ": order})

        restored = await engine.restore_managed_positions(session_factory)

        assert len(restored) == 1
        # Verify the entry_date matches the naive datetime interpreted as UTC
        from datetime import timezone as tz
        expected_ts = naive_dt.replace(tzinfo=tz.utc).timestamp()
        actual_ts = engine._managed_positions["TQQQ"].entry_date
        assert abs(actual_ts - expected_ts) < 1.0, (
            f"Naive datetime should be treated as UTC. "
            f"Expected ~{expected_ts}, got {actual_ts}"
        )

    @pytest.mark.asyncio
    async def test_restore_idempotent_returns_consistent_dict_shape(
        self, engine, mock_market_data,
    ):
        """Second restore call returns dicts with same keys as first call."""
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        restored1 = await engine.restore_managed_positions()
        restored2 = await engine.restore_managed_positions()

        # Both should have the same dict keys
        assert set(restored1[0].keys()) == set(restored2[0].keys()), (
            f"Dict shape mismatch: first={set(restored1[0].keys())}, "
            f"second={set(restored2[0].keys())}"
        )
        # Specifically verify quantity and sector are present in already_tracked
        assert "quantity" in restored2[0]
        assert "sector" in restored2[0]
        assert restored2[0]["source"] == "already_tracked"


class TestRestoreSellCooldownSeeding:
    """Test _last_sell_times seeding from DB SELL orders in restore_managed_positions."""

    @pytest.mark.asyncio
    async def test_sell_within_window_seeds_last_sell_times(
        self, engine, mock_market_data,
    ):
        """SELL order within cooldown window is seeded into _last_sell_times on restore."""
        # Need a broker position so the function doesn't return early before seeding
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        sell_time = datetime.now(UTC) - timedelta(hours=2)  # 2h ago, within 4h cooldown
        sell_order = _make_sell_order_mock("TQQQ", sell_time)
        session_factory = _mock_sell_session_factory([sell_order])

        await engine.restore_managed_positions(session_factory)

        assert "TQQQ" in engine._last_sell_times
        assert abs(engine._last_sell_times["TQQQ"] - sell_time.timestamp()) < 1.0

    @pytest.mark.asyncio
    async def test_no_recent_sell_leaves_last_sell_times_empty(
        self, engine, mock_market_data,
    ):
        """No SELL orders returned (DB cutoff filtered) → _last_sell_times stays empty."""
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        # Factory returns empty list — simulates DB returning nothing within the cooldown window
        session_factory = _mock_sell_session_factory([])

        await engine.restore_managed_positions(session_factory)

        assert "TQQQ" not in engine._last_sell_times

    @pytest.mark.asyncio
    async def test_db_failure_during_seeding_is_non_fatal(
        self, engine, mock_market_data,
    ):
        """DB failure in cooldown-seeding block is non-fatal; _last_sell_times stays empty."""
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        class _FailingSession:
            async def execute(self, stmt):
                raise RuntimeError("DB connection lost")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        def failing_factory():
            return _FailingSession()

        await engine.restore_managed_positions(failing_factory)

        assert engine._last_sell_times == {}

    @pytest.mark.asyncio
    async def test_cutoff_datetime_is_timezone_naive(
        self, engine, mock_market_data,
    ):
        """Cutoff datetime passed to DB query must be timezone-naive (offset-naive).

        Regression test for STOCK-64:
            invalid input for query argument $3:
            can't subtract offset-naive and offset-aware datetimes

        The DB column ``created_at`` is ``timestamp without time zone`` and
        stores UTC-naive values via ``datetime.utcnow()``.  The cutoff must
        also be naive (``datetime.utcnow()``) so asyncpg can compare them.
        """
        pos = MagicMock(symbol="TQQQ", quantity=100, current_price=55.0, avg_price=50.0)
        mock_market_data.get_positions.return_value = [pos]

        captured_cutoffs: list = []

        class _CutoffCapturingSession:
            """Captures the cutoff literal from the compiled SELL query."""

            async def execute(self, stmt):
                sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
                if "'SELL'" in sql:
                    # Extract the bound parameters from the statement before
                    # compilation so we can inspect the raw Python datetime object.
                    for param in stmt.whereclause.clauses:
                        try:
                            right = param.right
                            if hasattr(right, "value") and isinstance(right.value, datetime):
                                captured_cutoffs.append(right.value)
                        except AttributeError:
                            pass

                    class _FakeScalars:
                        def all(self_inner):
                            return []

                    class _SellResult:
                        def scalars(self_inner):
                            return _FakeScalars()

                    return _SellResult()

                class _BuyResult:
                    def scalar_one_or_none(self_inner):
                        return None

                return _BuyResult()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        def capturing_factory():
            return _CutoffCapturingSession()

        await engine.restore_managed_positions(capturing_factory)

        # At least one cutoff datetime must have been captured from the SELL query.
        # If none were captured the test is inconclusive — warn rather than false-pass.
        assert captured_cutoffs, (
            "No cutoff datetime was captured from the SELL query. "
            "Check that the factory correctly intercepts the statement."
        )
        for cutoff in captured_cutoffs:
            assert cutoff.tzinfo is None, (
                f"Cutoff datetime {cutoff!r} is offset-aware. "
                "Use datetime.utcnow() to produce an offset-naive cutoff "
                "matching the 'timestamp without time zone' DB column."
            )
