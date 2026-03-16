"""Tests for portfolio total_equity calculation — STOCK-1.

Validates:
- CTRP6504R primary path returns tot_asst_krw directly
- Fallback path: no double-counting, USD cash included
- available_cash with/without 통합증거금
- Exchange rate fallback chain
- Frontend-aligned API response structure
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
    last_exchange_rate: float = 1400.0,
    adapter_exchange_rate: float = 0.0,  # from _fetch_exchange_rate
) -> FastAPI:
    """Create test app with configurable mock state."""
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")

    # US adapter mock
    us_adapter = AsyncMock()
    us_adapter.fetch_balance = AsyncMock(
        return_value=us_balance
        or Balance(currency="USD", total=5000, available=3000, locked=2000),
    )
    us_adapter.fetch_positions = AsyncMock(return_value=us_positions or [])
    us_adapter._tot_asst_krw = tot_asst_krw
    us_adapter._last_exchange_rate = last_exchange_rate

    async def _mock_rate():
        return adapter_exchange_rate

    us_adapter._fetch_exchange_rate = _mock_rate

    # KR adapter mock
    kr_adapter = AsyncMock()
    kr_adapter.fetch_balance = AsyncMock(
        return_value=kr_balance
        or Balance(currency="KRW", total=5_000_000, available=3_000_000, locked=2_000_000),
    )
    kr_adapter.fetch_positions = AsyncMock(return_value=kr_positions or [])

    us_rl = RateLimiter(max_per_second=100)
    kr_rl = RateLimiter(max_per_second=100)

    app.state.adapter = us_adapter
    app.state.market_data = MarketDataService(adapter=us_adapter, rate_limiter=us_rl)
    app.state.kr_market_data = MarketDataService(adapter=kr_adapter, rate_limiter=kr_rl)

    return app


class TestTotalEquityPrimary:
    """CTRP6504R (통합증거금) is available — uses tot_asst_krw directly."""

    def test_uses_tot_asst_krw_when_available(self):
        """total_equity should exactly equal _tot_asst_krw."""
        app = _make_app(tot_asst_krw=50_000_000, last_exchange_rate=1400.0)
        client = TestClient(app)
        resp = client.get("/api/v1/portfolio/summary")
        data = resp.json()
        assert data["total_equity"] == 50_000_000

    def test_available_cash_is_krw_only_with_integrated(self):
        """통합증거금 active → available_cash = krw_available (no USD double-count)."""
        kr_bal = Balance(currency="KRW", total=10_000_000, available=6_000_000, locked=4_000_000)
        us_bal = Balance(currency="USD", total=8000, available=5000, locked=3000)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            tot_asst_krw=50_000_000,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # Should be KR available only — 통합증거금 already includes USD
        assert data["available_cash"] == 6_000_000

    def test_exchange_rate_included(self):
        """Response includes exchange_rate field."""
        app = _make_app(tot_asst_krw=50_000_000, last_exchange_rate=1380.0)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert "exchange_rate" in data
        assert data["exchange_rate"] > 0


class TestTotalEquityFallback:
    """CTRP6504R not available — uses KR total + US total * rate."""

    def test_fallback_uses_krw_plus_usd_times_rate(self):
        """Fallback: krw_total + usd_total * rate — no double-counting."""
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
        # total_equity = 5_000_000 + 5000 * 1400 = 12_000_000
        expected = 5_000_000 + 5000 * 1400
        assert data["total_equity"] == expected

    def test_fallback_includes_usd_cash(self):
        """USD cash (available) is included in fallback total_equity.

        This was the original bug — fallback only counted US positions, not cash.
        """
        kr_bal = Balance(currency="KRW", total=1_000_000, available=1_000_000, locked=0)
        # US: $3000 available cash + $2000 in positions = $5000 total
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            tot_asst_krw=None,
            last_exchange_rate=1400.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # total should include full USD total (cash + positions)
        # = 1_000_000 + 5000 * 1400 = 8_000_000
        expected = 1_000_000 + 5000 * 1400
        assert data["total_equity"] == expected

    def test_fallback_no_double_count_with_positions(self):
        """US positions should NOT be counted twice in fallback.

        US adapter's total already includes position value.
        """
        us_positions = [
            Position(
                symbol="AAPL",
                exchange="NASD",
                quantity=10,
                avg_price=150.0,
                current_price=160.0,
                unrealized_pnl=100.0,
                unrealized_pnl_pct=6.67,
            ),
        ]
        # US total = cash ($2000) + positions ($1600) = $3600
        us_bal = Balance(currency="USD", total=3600, available=2000, locked=1600)
        kr_bal = Balance(currency="KRW", total=2_000_000, available=2_000_000, locked=0)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            us_positions=us_positions,
            tot_asst_krw=None,
            last_exchange_rate=1400.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # Should be: 2_000_000 + 3600 * 1400 = 7_040_000
        # NOT: 2_000_000 + 3600*1400 + 1600*1400 (double-count)
        expected = 2_000_000 + 3600 * 1400
        assert data["total_equity"] == expected

    def test_fallback_available_cash_includes_usd(self):
        """Fallback: available_cash = krw_available + usd_available * rate."""
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
        # available_cash = 3_000_000 + 3000 * 1400 = 7_200_000
        expected = 3_000_000 + 3000 * 1400
        assert data["available_cash"] == expected

    def test_fallback_zero_asst_krw(self):
        """_tot_asst_krw = 0 should trigger fallback."""
        kr_bal = Balance(currency="KRW", total=1_000_000, available=1_000_000, locked=0)
        us_bal = Balance(currency="USD", total=2000, available=2000, locked=0)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            tot_asst_krw=0,
            last_exchange_rate=1400.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        expected = 1_000_000 + 2000 * 1400
        assert data["total_equity"] == expected


class TestExchangeRate:
    """Exchange rate fallback chain: MarketDataService → adapter cache → 1450."""

    def test_rate_from_adapter_cache(self):
        """Uses adapter's _last_exchange_rate when _fetch_exchange_rate returns 0."""
        app = _make_app(
            tot_asst_krw=None,
            last_exchange_rate=1380.0,
            adapter_exchange_rate=0.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # _fetch_exchange_rate returns 0, so fall back to _last_exchange_rate
        assert data["exchange_rate"] == 1380.0

    def test_rate_from_fetch_exchange_rate(self):
        """Uses _fetch_exchange_rate when it returns positive value."""
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
        """ALL summary has balance, usd_balance, exchange_rate, total_equity, available_cash."""
        app = _make_app(tot_asst_krw=50_000_000)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["market"] == "ALL"
        assert "balance" in data
        assert "usd_balance" in data
        assert "exchange_rate" in data
        assert "total_equity" in data
        assert "available_cash" in data
        assert "positions_count" in data
        assert "total_unrealized_pnl" in data
        assert "total_unrealized_pnl_usd" in data

    def test_usd_balance_fields(self):
        """usd_balance has total and available."""
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        app = _make_app(us_balance=us_bal, tot_asst_krw=50_000_000)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["usd_balance"]["total"] == 5000
        assert data["usd_balance"]["available"] == 3000

    def test_no_us_market_data(self):
        """Works even when US market data service is unavailable."""
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")

        kr_adapter = AsyncMock()
        kr_adapter.fetch_balance = AsyncMock(
            return_value=Balance(currency="KRW", total=5_000_000, available=3_000_000),
        )
        kr_adapter.fetch_positions = AsyncMock(return_value=[])
        kr_rl = RateLimiter(max_per_second=100)
        app.state.kr_market_data = MarketDataService(adapter=kr_adapter, rate_limiter=kr_rl)
        # No US market data or adapter

        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["total_equity"] == 5_000_000
        assert data["usd_balance"]["total"] == 0

    def test_no_kr_market_data(self):
        """Works even when KR market data service is unavailable."""
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")

        us_adapter = AsyncMock()
        us_adapter.fetch_balance = AsyncMock(
            return_value=Balance(currency="USD", total=5000, available=3000),
        )
        us_adapter.fetch_positions = AsyncMock(return_value=[])
        us_adapter._tot_asst_krw = None
        us_adapter._last_exchange_rate = 1400.0

        async def _mock_rate():
            return 1400.0

        us_adapter._fetch_exchange_rate = _mock_rate

        us_rl = RateLimiter(max_per_second=100)
        app.state.adapter = us_adapter
        app.state.market_data = MarketDataService(adapter=us_adapter, rate_limiter=us_rl)
        # No KR market data

        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # total_equity = 0 (kr) + 5000 * 1400 = 7_000_000
        assert data["total_equity"] == 5000 * 1400
