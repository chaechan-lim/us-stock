"""Scenario 3: Market regime switch (BULL -> BEAR).

1. BULL regime -> holding TQQQ
2. Market drops below SMA200, conditions deteriorate
3. Regime detected as BEAR after confirmation days
4. TQQQ liquidated
5. Weight profile switches to downtrend
6. Strategy weights change confirmed
"""

import pytest
import pandas as pd
import numpy as np

from exchange.paper_adapter import PaperAdapter
from data.indicator_service import IndicatorService
from strategies.regime_switch import RegimeSwitchStrategy
from strategies.combiner import SignalCombiner
from strategies.config_loader import StrategyConfigLoader
from core.enums import SignalType
from engine.risk_manager import RiskManager
from engine.order_manager import OrderManager
from tests.scenarios.conftest import make_ohlcv


@pytest.fixture
def bear_df():
    """300-day data that ends well below SMA200 for clear bear signal."""
    svc = IndicatorService()
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    # First 200 days: price around 400, then last 100 days: crash to 300
    prices = np.concatenate([
        np.linspace(380, 420, 200),  # stable/up
        np.linspace(420, 300, 100),  # strong decline
    ])
    df = pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)
    return svc.add_all_indicators(df)


@pytest.mark.asyncio
async def test_regime_switch_bull_to_bear(bear_df):
    """Strategy detects bear regime -> SELL signal."""
    strategy = RegimeSwitchStrategy(params={"confirmation_days": 2})
    signal = await strategy.analyze(bear_df, "TQQQ")

    # Price is at 300, SMA200 is around 380 -> clearly below -> SELL
    assert signal.signal_type == SignalType.SELL
    assert "Bear" in signal.reason


@pytest.mark.asyncio
async def test_liquidate_leveraged_etf_on_regime_change():
    """Full flow: holding TQQQ -> regime switches -> liquidate."""
    adapter = PaperAdapter(initial_balance_usd=100_000)
    await adapter.initialize()

    risk = RiskManager()
    om = OrderManager(adapter=adapter, risk_manager=risk)

    # Start with TQQQ position from bull regime
    adapter.set_price("TQQQ", 50.0)
    buy = await om.place_buy(
        symbol="TQQQ", price=50.0,
        portfolio_value=100_000, cash_available=100_000,
        current_positions=0, strategy_name="regime_switch",
    )
    assert buy is not None

    positions = await adapter.fetch_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "TQQQ"

    # Price crashes (regime change)
    crash_price = 35.0
    adapter.set_price("TQQQ", crash_price)

    # Sell on bear signal
    sell = await om.place_sell(
        symbol="TQQQ", quantity=buy.quantity,
        price=crash_price, strategy_name="regime_switch",
    )
    assert sell.status == "filled"

    positions = await adapter.fetch_positions()
    assert len(positions) == 0

    # Loss incurred
    balance = await adapter.fetch_balance()
    assert balance.total < 100_000


@pytest.mark.asyncio
async def test_weight_profile_changes_on_regime():
    """Config loader returns different weights for uptrend vs downtrend."""
    loader = StrategyConfigLoader()

    uptrend_w = loader.get_profile_weights("uptrend")
    downtrend_w = loader.get_profile_weights("downtrend")

    # Uptrend should have trend_following with meaningful weight
    assert uptrend_w.get("trend_following", 0) >= 0.10

    # Downtrend should suppress trend_following
    assert downtrend_w.get("trend_following", 0) <= 0.10

    # Downtrend should have rsi_divergence active
    assert downtrend_w.get("rsi_divergence", 0) > 0


@pytest.mark.asyncio
async def test_combiner_uses_profile_weights():
    """Signal combiner respects different weight profiles."""
    from strategies.base import Signal

    combiner = SignalCombiner()

    buy_signal = Signal(
        signal_type=SignalType.BUY, confidence=0.8,
        strategy_name="trend_following", reason="test",
    )
    sell_signal = Signal(
        signal_type=SignalType.SELL, confidence=0.8,
        strategy_name="rsi_divergence", reason="test",
    )

    # Uptrend weights: trend_following dominant
    uptrend_w = {"trend_following": 0.30, "rsi_divergence": 0.05}
    result = combiner.combine([buy_signal, sell_signal], uptrend_w)
    assert result.signal_type == SignalType.BUY

    # Downtrend weights: rsi_divergence dominant
    downtrend_w = {"trend_following": 0.05, "rsi_divergence": 0.30}
    result = combiner.combine([buy_signal, sell_signal], downtrend_w)
    assert result.signal_type == SignalType.SELL
