"""Tests for portfolio total_equity calculation — STOCK-1 / STOCK-12 / STOCK-59.

Validates:
- total_equity = kr_tot_evlu_amt + us_tot_asst_amt - shared_deposit (통합증거금)
- Fallback: krw_total + usd_total * rate
- No double-counting of shared deposit
- available_cash = full_available_usd * rate (통합증거금)
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
    us_adapter._fetch_exchange_rate = AsyncMock(return_value=adapter_exchange_rate)

    # KR adapter mock
    kr_adapter = AsyncMock()
    kr_adapter.fetch_balance = AsyncMock(
        return_value=kr_balance
        or Balance(currency="KRW", total=5_000_000, available=3_000_000, locked=2_000_000),
    )
    kr_adapter.fetch_positions = AsyncMock(return_value=kr_positions or [])
    kr_adapter._tot_evlu_amt = kr_tot_evlu_amt

    us_rl = RateLimiter(max_per_second=100)
    kr_rl = RateLimiter(max_per_second=100)

    app.state.adapter = us_adapter
    app.state.kr_adapter = kr_adapter
    app.state.market_data = MarketDataService(adapter=us_adapter, rate_limiter=us_rl)
    app.state.kr_market_data = MarketDataService(adapter=kr_adapter, rate_limiter=kr_rl)

    return app


class TestTotalEquityIntegratedMargin:
    """통합증거금: total = kr_tot_evlu + us_tot_asst - shared_deposit."""

    def test_combined_formula(self):
        """Primary path: kr_tot_evlu + us_tot_asst - shared_deposit."""
        # Real-world scenario: deposit 6M in both, KR stocks 2.7M, US stocks ~7.4M
        kr_tot_evlu = 10_721_738  # deposit + KR stocks + some overseas
        us_tot_asst = 14_544_921  # deposit + US stocks at broker rate
        shared_deposit = 6_048_888

        app = _make_app(
            tot_asst_krw=us_tot_asst,
            tot_dncl_krw=shared_deposit,
            kr_tot_evlu_amt=kr_tot_evlu,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()

        expected = kr_tot_evlu + us_tot_asst - shared_deposit
        assert abs(data["total_equity"] - expected) < 1.0

    def test_no_shared_deposit_skips_dedup(self):
        """When tot_dncl_krw=0, falls back (no 통합증거금 detection)."""
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
        """When kr_tot_evlu_amt=0, falls to us_tot_asst + kr branch."""
        app = _make_app(
            tot_asst_krw=14_544_921,
            tot_dncl_krw=6_048_888,
            kr_tot_evlu_amt=0,  # not available
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # Second branch: krw_total + us_tot_asst
        kr_default_total = 5_000_000
        expected = kr_default_total + 14_544_921
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
                     "total_unrealized_pnl_usd"]:
            assert key in data

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
    """STOCK-42: available_cash must not double-count KRW in 통합증거금 accounts."""

    def test_unified_margin_uses_full_available_usd(self):
        kr_bal = Balance(currency="KRW", total=10_000_000, available=8_000_000, locked=2_000_000)
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        app = _make_app(
            kr_balance=kr_bal, us_balance=us_bal,
            last_exchange_rate=1400.0,
            full_account_usd=11000, full_available_usd=9000,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        expected = 9000 * 1400
        assert data["available_cash"] == expected

    def test_available_cash_capped_at_total_equity(self):
        kr_bal = Balance(currency="KRW", total=1_000_000, available=500_000, locked=500_000)
        us_bal = Balance(currency="USD", total=2000, available=1500, locked=500)
        app = _make_app(
            kr_balance=kr_bal, us_balance=us_bal,
            last_exchange_rate=1400.0,
            full_account_usd=5000, full_available_usd=15000,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["available_cash"] <= data["total_equity"]

    def test_fallback_path_unchanged(self):
        kr_bal = Balance(currency="KRW", total=5_000_000, available=3_000_000, locked=2_000_000)
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        app = _make_app(
            kr_balance=kr_bal, us_balance=us_bal,
            last_exchange_rate=1400.0,
            full_account_usd=0, full_available_usd=0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        expected = 3_000_000 + 3000 * 1400
        assert data["available_cash"] == expected
