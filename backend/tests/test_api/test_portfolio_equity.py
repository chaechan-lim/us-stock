"""Tests for portfolio total_equity calculation — STOCK-1 / STOCK-12.

Validates:
- total_equity = krw_total + usd_total * rate (unified path)
- No double-counting of USD cash or positions
- available_cash = krw_available + usd_available * rate
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
    usd_deposit_krw: float = 0,
    last_exchange_rate: float = 1400.0,
    adapter_exchange_rate: float = 0.0,  # from _fetch_exchange_rate
    full_account_usd: float = 0,
    full_available_usd: float = 0,
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
    us_adapter._usd_deposit_krw = usd_deposit_krw
    us_adapter._last_exchange_rate = last_exchange_rate
    us_adapter._full_account_usd = full_account_usd
    us_adapter._full_available_usd = full_available_usd

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


class TestTotalEquityWithTotAsstKrw:
    """STOCK-59: tot_asst_krw drives total_equity when available.

    CTRP6504R tot_asst_amt (= _tot_asst_krw) contains the full account value in
    KRW at the broker's base rate. frcr_evlu_tota (= _usd_deposit_krw) is the
    USD portion at the same rate. We re-value the USD portion at the live market
    rate to get an accurate total_equity.
    """

    def test_uses_tot_asst_krw_with_usd_breakdown(self):
        """Primary path: split KRW portion + USD re-valued at market rate."""
        # tot_asst_krw = 18_603_296 (full account at pb_rate=1450)
        # usd_deposit_krw = 13_050_000 (USD portion: ~$9,000 × 1450)
        # market_rate = 1498.7 (live rate)
        pb_rate = 1450.0
        usd_krw_at_broker = 13_050_000.0  # frcr_evlu_tota
        tot = 18_603_296.0
        market_rate = 1498.7
        app = _make_app(
            tot_asst_krw=tot,
            usd_deposit_krw=usd_krw_at_broker,
            last_exchange_rate=pb_rate,
            adapter_exchange_rate=market_rate,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()

        krw_portion = tot - usd_krw_at_broker
        usd_value = usd_krw_at_broker / pb_rate
        expected = krw_portion + usd_value * market_rate
        assert abs(data["total_equity"] - expected) < 1.0  # within 1 KRW

    def test_uses_tot_asst_krw_no_usd_deposit(self):
        """When usd_deposit_krw=0, total_equity = tot_asst_krw (pure KRW account)."""
        app = _make_app(
            tot_asst_krw=10_000_000,
            usd_deposit_krw=0,
            last_exchange_rate=1400.0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # krw_portion = 10_000_000 - 0 = 10_000_000; usd_value = 0
        assert data["total_equity"] == 10_000_000

    def test_fallback_when_tot_asst_krw_zero(self):
        """When tot_asst_krw=0, falls back to krw_total + usd_total * rate."""
        app = _make_app(tot_asst_krw=0, last_exchange_rate=1400.0)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # Default: krw_total=5_000_000, usd_total=5000, rate=1400
        expected = 5_000_000 + 5000 * 1400
        assert data["total_equity"] == expected

    def test_available_cash_includes_usd_with_tot_asst_krw(self):
        """available_cash = krw_available + usd_available * rate, regardless of tot_asst_krw."""
        kr_bal = Balance(currency="KRW", total=10_000_000, available=6_000_000, locked=4_000_000)
        us_bal = Balance(currency="USD", total=8000, available=5000, locked=3000)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            tot_asst_krw=50_000_000,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # available_cash = 6_000_000 + 5000 * 1400 = 13_000_000
        expected = 6_000_000 + 5000 * 1400
        assert data["available_cash"] == expected

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
        us_adapter._full_account_usd = 0
        us_adapter._full_available_usd = 0

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


class TestUnifiedMarginAvailableCash:
    """STOCK-42: available_cash must not double-count KRW in 통합증거금 accounts.

    In 통합증거금 mode, frcr_ord_psbl_amt1 already includes KRW auto-conversion.
    Adding krw_available on top would double-count the same cash pool.
    """

    def test_unified_margin_uses_full_available_usd(self):
        """통합증거금: available_cash = _full_available_usd * rate (no KR double-count)."""
        kr_bal = Balance(currency="KRW", total=10_000_000, available=8_000_000, locked=2_000_000)
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        # full_available_usd=9000 means frcr_ord_psbl_amt1 includes KRW conversion
        # full_account_usd=11000 triggers 통합증거금 path for total_equity
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            last_exchange_rate=1400.0,
            full_account_usd=11000,
            full_available_usd=9000,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # available_cash = 9000 * 1400 = 12_600_000
        # NOT: 8_000_000 + 3000 * 1400 = 12_200_000 (double-counting)
        expected = 9000 * 1400
        assert data["available_cash"] == expected

    def test_unified_margin_no_double_count(self):
        """Simulates the reported bug: available_cash should NOT exceed total_equity."""
        # Real-world scenario from bug report:
        # total_equity: ~18.8M, available_cash was ~20.8M (double-counted)
        kr_bal = Balance(currency="KRW", total=6_000_000, available=5_000_000, locked=1_000_000)
        us_bal = Balance(currency="USD", total=9000, available=7000, locked=2000)
        # US adapter's frcr_ord_psbl_amt1 already includes KRW auto-conversion
        # full_available_usd = 10 (uncapped buying power in USD)
        # full_account_usd = 12 (buying power + positions)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            last_exchange_rate=1450.0,
            full_account_usd=12000,
            full_available_usd=10000,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["available_cash"] <= data["total_equity"]

    def test_available_cash_capped_at_total_equity(self):
        """Safety cap: available_cash must never exceed total_equity."""
        kr_bal = Balance(currency="KRW", total=1_000_000, available=500_000, locked=500_000)
        us_bal = Balance(currency="USD", total=2000, available=1500, locked=500)
        # Extreme case: full_available_usd is very large
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            last_exchange_rate=1400.0,
            full_account_usd=5000,
            full_available_usd=15000,  # artificially high
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["available_cash"] <= data["total_equity"]

    def test_fallback_path_unchanged(self):
        """When full_available_usd=0, fallback to krw_available + usd_available * rate."""
        kr_bal = Balance(currency="KRW", total=5_000_000, available=3_000_000, locked=2_000_000)
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            last_exchange_rate=1400.0,
            full_account_usd=0,
            full_available_usd=0,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # Fallback: 3_000_000 + 3000 * 1400 = 7_200_000
        expected = 3_000_000 + 3000 * 1400
        assert data["available_cash"] == expected

    def test_fallback_when_no_full_us_usd(self):
        """When full_account_usd=0 but full_available_usd>0, still use fallback."""
        kr_bal = Balance(currency="KRW", total=5_000_000, available=3_000_000, locked=2_000_000)
        us_bal = Balance(currency="USD", total=5000, available=3000, locked=2000)
        # full_available_usd is set but full_account_usd (full_us_usd) is 0
        # => not in 통합증거금 mode, so fallback path should be used
        app = _make_app(
            kr_balance=kr_bal,
            us_balance=us_bal,
            last_exchange_rate=1400.0,
            full_account_usd=0,
            full_available_usd=9000,
        )
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/summary").json()
        # Fallback: 3_000_000 + 3000 * 1400 = 7_200_000
        expected = 3_000_000 + 3000 * 1400
        assert data["available_cash"] == expected
