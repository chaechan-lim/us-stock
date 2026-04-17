"""Tests for portfolio total_equity calculation — STOCK-1 / STOCK-12 / STOCK-59.

Validates:
- total_equity = kr_tot_evlu_krw + us_position_value_krw (통합증거금)
- Fallback: krw_total + usd_total * rate
- No double-counting of shared deposit
- available_cash = KIS 주문가능예수금 (통합증거금) or total_equity - total_position_value
- Exchange rate fallback chain
"""

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.router import api_router
from data.market_data_service import MarketDataService
from exchange.base import Balance, Position
from services.rate_limiter import RateLimiter


def _make_app(
    *,
    kr_balance: Balance | None = None,
    us_balance: Balance | None = None,
    kr_positions: list[Position] | None = None,
    us_positions: list[Position] | None = None,
    tot_asst_krw: float | None = None,
    tot_dncl_krw: float = 0,
    usd_deposit_krw: float = 0,
    last_exchange_rate: float = 1400.0,
    adapter_exchange_rate: float = 0.0,
    full_account_usd: float = 0,
    full_available_usd: float = 0,
    kr_tot_evlu_amt: float = 0,
    us_position_value_krw: float = 0,
    withdrawable_total_krw: float = 0,
    kr_deposit_krw: float = 0,
    kr_stock_eval_krw: float = 0,
    integrated_total_asset: float = 0,
) -> FastAPI:
    """Create test app with configurable mock state."""
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")

    # US adapter mock
    us_adapter = AsyncMock()
    us_adapter.fetch_balance = AsyncMock(
        return_value=us_balance or Balance(currency="USD", total=5000, available=3000, locked=2000),
    )
    us_adapter.fetch_positions = AsyncMock(return_value=us_positions or [])
    us_adapter._tot_asst_krw = tot_asst_krw
    us_adapter._tot_dncl_krw = tot_dncl_krw
    us_adapter._usd_deposit_krw = usd_deposit_krw
    us_adapter._last_exchange_rate = last_exchange_rate
    us_adapter._full_account_usd = full_account_usd
    us_adapter._full_available_usd = full_available_usd
    us_adapter._us_position_value_krw = us_position_value_krw
    us_adapter._withdrawable_total_krw = withdrawable_total_krw
    us_adapter._fetch_exchange_rate = AsyncMock(return_value=adapter_exchange_rate)

    # KR adapter mock
    kr_adapter = AsyncMock()
    kr_adapter.fetch_balance = AsyncMock(
        return_value=kr_balance
        or Balance(currency="KRW", total=5_000_000, available=3_000_000, locked=2_000_000),
    )
    kr_adapter.fetch_positions = AsyncMock(return_value=kr_positions or [])
    kr_adapter._tot_evlu_amt = kr_tot_evlu_amt
    kr_adapter._dnca_tot_amt = kr_deposit_krw
    kr_adapter._scts_evlu_amt = kr_stock_eval_krw
    kr_adapter._integrated_total_asset = integrated_total_asset

    us_rl = RateLimiter(max_per_second=100)
    kr_rl = RateLimiter(max_per_second=100)

    app.state.adapter = us_adapter
    app.state.kr_adapter = kr_adapter
    app.state.market_data = MarketDataService(adapter=us_adapter, rate_limiter=us_rl)
    app.state.kr_market_data = MarketDataService(adapter=kr_adapter, rate_limiter=kr_rl)

    return app


class TestTotalEquityIntegratedMargin:
    """통합증거금: total = 예수금 + 국내주식평가 + 해외주식평가.

    2026-04-15: KIS 앱 총자산 필드 직접 확인 후 수정.
    kr_tot_evlu_amt는 해외증거금 차감이 섞여서 부정확.
    개별 3필드 합산이 KIS 앱과 가장 근접 (시차 ~1만원).
    """

    def test_combined_formula(self):
        """Primary: CTRP6548R tot_asst_amt used directly.

        2026-04-17: No calculation — KIS 통합총자산 API 값 그대로.
        """
        app = _make_app(
            integrated_total_asset=17_770_319,
            us_balance=Balance(currency="USD", total=10_000, available=3000, locked=7000),
            kr_stock_eval_krw=2_316_380,
            kr_tot_evlu_amt=9_602_699,
            us_position_value_krw=9_561_545,
            last_exchange_rate=1400.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()

        assert abs(data["total_equity"] - 17_770_319) < 1.0
        assert "CTRP6548R" in data["equity_breakdown"]["formula"]

    def test_no_shared_deposit_skips_dedup(self):
        """Without US position KRW value, falls back to prior branch."""
        app = _make_app(
            tot_asst_krw=14_544_921,
            tot_dncl_krw=0,  # no shared deposit info
            kr_tot_evlu_amt=10_721_738,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # Falls to second branch: krw_total + us_tot_asst
        kr_default_total = 5_000_000  # from default KR balance
        expected = kr_default_total + 14_544_921
        assert abs(data["total_equity"] - expected) < 1.0

    def test_no_kr_tot_evlu_uses_us_plus_kr(self):
        """When KR raw total is missing, use KR balance total + US eval."""
        app = _make_app(
            tot_asst_krw=14_544_921,
            tot_dncl_krw=6_048_888,
            us_position_value_krw=9_561_545,
            kr_tot_evlu_amt=0,  # not available
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # Integrated branch: krw_total + us_position_value_krw
        kr_default_total = 5_000_000
        expected = kr_default_total + 9_561_545
        assert abs(data["total_equity"] - expected) < 1.0


class TestTotalEquityFallback:
    """CTRP6504R not available — uses KR total + US total * rate."""

    def test_fallback_uses_krw_plus_usd_times_rate(self):
        kr_bal = Balance(currency="KRW", total=5_000_000, available=3_000_000, locked=2_000_000)
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            tot_asst_krw=None,
            last_exchange_rate=1400.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        expected = 5_000_000 + 5000 * 1400
        assert data["total_equity"] == expected

    def test_fallback_when_tot_asst_krw_zero(self):
        app = _make_app(
            tot_asst_krw=0,
            last_exchange_rate=1400.0,
            adapter_exchange_rate=1400.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        expected = 5_000_000 + 5000 * 1400
        assert data["total_equity"] == expected

    def test_fallback_includes_usd_cash(self):
        kr_bal = Balance(currency="KRW", total=1_000_000, available=1_000_000, locked=0)
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            tot_asst_krw=None,
            last_exchange_rate=1400.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        expected = 1_000_000 + 5000 * 1400
        assert data["total_equity"] == expected

    def test_fallback_no_double_count_with_positions(self):
        us_positions = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10,
                avg_price=150.0, current_price=160.0,
                unrealized_pnl=100.0, unrealized_pnl_pct=6.67,
            ),
        ]
        us_bal = Balance(currency="USD", total=3600, available=2000, locked=1600)
        kr_bal = Balance(currency="KRW", total=2_000_000, available=2_000_000, locked=0)
        app = _make_app(
            kr_balance=kr_bal, us_balance=us_bal,
            us_positions=us_positions,
            tot_asst_krw=None, last_exchange_rate=1400.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        expected = 2_000_000 + 3600 * 1400
        assert data["total_equity"] == expected


class TestExchangeRate:
    """Exchange rate fallback chain: MarketDataService → adapter cache → 1450."""

    def test_rate_from_adapter_cache(self):
        app = _make_app(
            tot_asst_krw=None,
            last_exchange_rate=1380.0,
            adapter_exchange_rate=0.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["exchange_rate"] == 1380.0

    def test_rate_from_fetch_exchange_rate(self):
        app = _make_app(
            tot_asst_krw=None,
            last_exchange_rate=1380.0,
            adapter_exchange_rate=1395.5,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["exchange_rate"] == 1395.5


class TestResponseStructure:
    """Validate API response has all required fields."""

    def test_combined_summary_fields(self):
        app = _make_app(tot_asst_krw=50_000_000, tot_dncl_krw=6_000_000, kr_tot_evlu_amt=10_000_000)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["market"] == "ALL"
        for key in ["balance", "usd_balance", "exchange_rate", "total_equity",
                     "available_cash", "positions_count", "total_unrealized_pnl",
                     "total_unrealized_pnl_usd", "equity_breakdown", "cash_breakdown"]:
            assert key in data

    def test_combined_summary_breakdown_fields(self):
        app = _make_app(
            kr_balance=Balance(currency="KRW", total=8_654_115, available=4_183_180, locked=4_470_935),
            kr_tot_evlu_amt=9_602_699,
            us_position_value_krw=9_561_545,
            withdrawable_total_krw=4_183_090,
            kr_deposit_krw=4_183_180,
            kr_stock_eval_krw=2_316_380,
            integrated_total_asset=17_770_319,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert "CTRP6548R" in data["equity_breakdown"]["formula"]
        assert data["cash_breakdown"]["combined_cash_krw"] == data["available_cash"]

    def test_usd_balance_fields(self):
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        app = _make_app(us_balance=us_bal, tot_asst_krw=50_000_000, tot_dncl_krw=6_000_000, kr_tot_evlu_amt=10_000_000)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["usd_balance"]["total"] == 5000
        assert data["usd_balance"]["available"] == 3000

    def test_no_us_market_data(self):
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        kr_adapter = AsyncMock()
        kr_adapter.fetch_balance = AsyncMock(
            return_value=Balance(currency="KRW", total=5_000_000, available=3_000_000),
        )
        kr_adapter.fetch_positions = AsyncMock(return_value=[])
        kr_adapter._tot_evlu_amt = 0
        kr_rl = RateLimiter(max_per_second=100)
        app.state.kr_market_data = MarketDataService(adapter=kr_adapter, rate_limiter=kr_rl)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["total_equity"] == 5_000_000

    def test_no_kr_market_data(self):
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        us_adapter = AsyncMock()
        us_adapter.fetch_balance = AsyncMock(
            return_value=Balance(currency="USD", total=5000, available=3000),
        )
        us_adapter.fetch_positions = AsyncMock(return_value=[])
        us_adapter._tot_asst_krw = None
        us_adapter._tot_dncl_krw = 0
        us_adapter._last_exchange_rate = 1400.0
        us_adapter._full_account_usd = 0
        us_adapter._full_available_usd = 0
        us_adapter._fetch_exchange_rate = AsyncMock(return_value=1400.0)
        us_rl = RateLimiter(max_per_second=100)
        app.state.adapter = us_adapter
        app.state.market_data = MarketDataService(adapter=us_adapter, rate_limiter=us_rl)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["total_equity"] == 5000 * 1400


class TestUnifiedMarginAvailableCash:
    """available_cash = KIS 주문가능예수금 (통합증거금) or equity - positions."""

    def test_available_cash_equals_equity_minus_positions(self):
        """No positions + no integrated raw fields → available_cash == total_equity."""
        kr_bal = Balance(currency="KRW", total=10_000_000, available=8_000_000, locked=2_000_000)
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        app = _make_app(
            kr_balance=kr_bal, us_balance=us_bal,
            last_exchange_rate=1400.0,
            full_account_usd=11000, full_available_usd=9000,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # No positions, so available_cash == total_equity
        assert data["available_cash"] == data["total_equity"]

    def test_positions_reduce_available_cash(self):
        """Fallback path: positions reduce available_cash."""
        kr_bal = Balance(currency="KRW", total=10_000_000, available=6_000_000, locked=4_000_000)
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        kr_positions = [
            Position(symbol="005930", exchange="KRX", quantity=10,
                     avg_price=70000, current_price=72000, unrealized_pnl=20000,
                     unrealized_pnl_pct=2.86),
        ]
        us_positions = [
            Position(symbol="AAPL", exchange="NASD", quantity=5,
                     avg_price=180, current_price=190, unrealized_pnl=50,
                     unrealized_pnl_pct=5.56),
        ]
        app = _make_app(
            kr_balance=kr_bal, us_balance=us_bal,
            kr_positions=kr_positions, us_positions=us_positions,
            last_exchange_rate=1400.0,
            full_account_usd=5000, full_available_usd=3000,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        total_equity = 10_000_000 + 5000 * 1400
        kr_pos_val = 10 * 72000  # 720,000
        us_pos_val = 5 * 190  # 950 USD → 950 * 1400 = 1,330,000
        expected = total_equity - kr_pos_val - us_pos_val * 1400
        assert abs(data["available_cash"] - expected) < 1.0

    def test_integrated_margin_available_cash_uses_orderable_cash(self):
        kr_bal = Balance(currency="KRW", total=8_654_115, available=4_183_180, locked=4_470_935)
        us_bal = Balance(currency="USD", total=9152.06, available=2653.16, locked=6498.9)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            kr_tot_evlu_amt=9_602_699,
            us_position_value_krw=9_561_545,
            withdrawable_total_krw=4_183_090,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["available_cash"] == 4_183_090

    def test_available_cash_never_negative(self):
        """available_cash is floored at 0."""
        kr_bal = Balance(currency="KRW", total=1_000_000, available=0, locked=1_000_000)
        us_bal = Balance(currency="USD", total=2000, available=0, locked=2000)
        # Positions worth more than equity (underwater)
        us_positions = [
            Position(symbol="AAPL", exchange="NASD", quantity=100,
                     avg_price=200, current_price=200, unrealized_pnl=0,
                     unrealized_pnl_pct=0),
        ]
        app = _make_app(
            kr_balance=kr_bal, us_balance=us_bal,
            us_positions=us_positions,
            last_exchange_rate=1400.0,
            full_account_usd=0, full_available_usd=0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["available_cash"] >= 0
