"""Tests for Layer 2: Fundamental Enricher."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from scanner.fundamental_enricher import FundamentalEnricher, EnrichedCandidate
from data.external_data_service import (
    ExternalDataService, StockProfile, StockInfo,
    ConsensusData, FundamentalData, SmartMoneyData,
)


def _make_profile(
    consensus_bull_pct: float = 0.7,
    target_upside: float = 15.0,
    revenue_growth: float = 0.15,
    profit_margin: float = 0.22,
    forward_pe: float = 20.0,
    institutional_pct: float = 0.65,
) -> StockProfile:
    total = 30
    buy_count = int(total * consensus_bull_pct)
    return StockProfile(
        symbol="AAPL",
        info=StockInfo(symbol="AAPL", name="Apple Inc."),
        consensus=ConsensusData(
            analyst_count=total, strong_buy=buy_count // 2,
            buy=buy_count - buy_count // 2,
            hold=total - buy_count, sell=0, strong_sell=0,
            target_upside_pct=target_upside,
        ),
        fundamentals=FundamentalData(
            revenue_growth=revenue_growth,
            profit_margin=profit_margin,
            forward_pe=forward_pe,
            peg_ratio=1.2,
        ),
        smart_money=SmartMoneyData(
            institutional_pct=institutional_pct,
            insider_buy_count_90d=3,
            insider_sell_count_90d=1,
            short_ratio=1.5,
        ),
    )


@pytest.fixture
def mock_data_service():
    svc = AsyncMock(spec=ExternalDataService)
    svc.get_stock_profile = AsyncMock(return_value=_make_profile())
    return svc


class TestFundamentalEnricher:
    async def test_enrich_good_stock(self, mock_data_service):
        enricher = FundamentalEnricher(data_service=mock_data_service)
        result = await enricher.enrich("AAPL", indicator_score=75.0, current_price=175.0)

        assert result.symbol == "AAPL"
        assert result.indicator_score == 75.0
        assert result.combined_score > 50
        assert result.consensus_score > 50
        assert result.fundamental_score > 50

    async def test_enrich_weak_stock(self, mock_data_service):
        mock_data_service.get_stock_profile.return_value = _make_profile(
            consensus_bull_pct=0.2,
            target_upside=-15.0,
            revenue_growth=-0.10,
            profit_margin=-0.05,
            forward_pe=80.0,
            institutional_pct=0.20,
        )
        enricher = FundamentalEnricher(data_service=mock_data_service)
        result = await enricher.enrich("WEAK", indicator_score=30.0)
        assert result.combined_score < 40

    async def test_enrich_batch(self, mock_data_service):
        enricher = FundamentalEnricher(data_service=mock_data_service)
        candidates = [
            ("AAPL", 80.0, 175.0),
            ("TSLA", 70.0, 250.0),
        ]
        results = await enricher.enrich_batch(candidates)
        assert len(results) == 2
        # Should be sorted by combined_score descending
        assert results[0].combined_score >= results[1].combined_score

    async def test_enrich_batch_partial_failure(self, mock_data_service):
        mock_data_service.get_stock_profile.side_effect = [
            _make_profile(),
            Exception("API error"),
        ]
        enricher = FundamentalEnricher(data_service=mock_data_service)
        results = await enricher.enrich_batch([
            ("AAPL", 80.0, 175.0),
            ("FAIL", 70.0, 0.0),
        ])
        assert len(results) == 1

    async def test_grade_assignment(self, mock_data_service):
        enricher = FundamentalEnricher(data_service=mock_data_service)
        result = await enricher.enrich("AAPL", indicator_score=80.0)
        assert result.grade in ("A", "B", "C", "D", "F")

    async def test_custom_weights(self, mock_data_service):
        enricher = FundamentalEnricher(
            data_service=mock_data_service,
            weights={"consensus": 0.80, "fundamental": 0.10, "smart_money": 0.10},
        )
        result = await enricher.enrich("AAPL", indicator_score=75.0)
        assert result.combined_score > 0


class TestEnrichedCandidate:
    def test_defaults(self):
        c = EnrichedCandidate(
            symbol="TEST", indicator_score=50, consensus_score=50,
            fundamental_score=50, smart_money_score=50, combined_score=50,
        )
        assert c.profile is None
        assert c.grade == ""


class TestConsensusBranches:
    """All branches of _score_consensus (analyst bull_pct + target upside)."""

    def setup_method(self):
        self.enricher = FundamentalEnricher(data_service=AsyncMock())

    def _consensus(self, **kw):
        defaults = dict(
            analyst_count=0, strong_buy=0, buy=0, hold=0, sell=0, strong_sell=0,
            target_upside_pct=0.0,
        )
        defaults.update(kw)
        return StockProfile(
            symbol="X",
            info=StockInfo(symbol="X", name="X"),
            consensus=ConsensusData(**defaults),
            fundamentals=FundamentalData(),
            smart_money=SmartMoneyData(),
        )

    def test_bull_pct_above_70(self):
        # 25 of 30 = 83% bull → +25
        p = self._consensus(analyst_count=30, strong_buy=15, buy=10, hold=5)
        assert self.enricher._score_consensus(p) > 70

    def test_bull_pct_below_30(self):
        # 6 of 30 = 20% bull → -15
        p = self._consensus(analyst_count=30, strong_buy=3, buy=3, hold=24)
        assert self.enricher._score_consensus(p) < 50

    def test_target_upside_above_20(self):
        p = self._consensus(target_upside_pct=30.0)
        assert self.enricher._score_consensus(p) > 50

    def test_target_upside_below_neg_10(self):
        p = self._consensus(target_upside_pct=-20.0)
        assert self.enricher._score_consensus(p) < 50

    def test_no_analyst_no_upside(self):
        p = self._consensus()
        assert self.enricher._score_consensus(p) == 50.0


class TestFundamentalBranches:
    """All branches of _score_fundamentals."""

    def setup_method(self):
        self.enricher = FundamentalEnricher(data_service=AsyncMock())

    def _profile(self, **kw):
        return StockProfile(
            symbol="X",
            info=StockInfo(symbol="X", name="X"),
            consensus=ConsensusData(),
            fundamentals=FundamentalData(**kw),
            smart_money=SmartMoneyData(),
        )

    def test_revenue_growth_high(self):
        # > 0.25 → +18
        assert self.enricher._score_fundamentals(
            self._profile(revenue_growth=0.30)
        ) > 50

    def test_revenue_growth_mid(self):
        # 0.15 < x → +12 (boundary just above 0.15)
        assert self.enricher._score_fundamentals(
            self._profile(revenue_growth=0.20)
        ) > 50

    def test_revenue_growth_negative(self):
        assert self.enricher._score_fundamentals(
            self._profile(revenue_growth=-0.10)
        ) < 50

    def test_earnings_growth_all_branches(self):
        for eg, predicate in [
            (0.30, lambda s: s > 50),
            (0.15, lambda s: s > 50),  # 0.10 < x
            (-0.20, lambda s: s < 50),
        ]:
            assert predicate(self.enricher._score_fundamentals(
                self._profile(earnings_growth=eg)
            ))

    def test_profit_margin_branches(self):
        # > 0.25 / > 0.15 / > 0.05 / < 0
        for pm in [0.30, 0.20, 0.08, -0.05]:
            s = self.enricher._score_fundamentals(self._profile(profit_margin=pm))
            assert 0 <= s <= 100

    def test_roe_branches(self):
        for roe in [0.30, 0.20, 0.02]:
            s = self.enricher._score_fundamentals(self._profile(roe=roe))
            assert 0 <= s <= 100

    def test_garp_branches(self):
        # garp = rev_growth / forward_pe * 100
        # > 1.0: rg=0.20 / pe=15 = 1.33
        # > 0.5: rg=0.10 / pe=15 = 0.67
        # < -0.2: rg=-0.05 / pe=15 = -0.33
        for rg, pe in [(0.20, 15.0), (0.10, 15.0), (-0.05, 15.0)]:
            s = self.enricher._score_fundamentals(
                self._profile(revenue_growth=rg, forward_pe=pe)
            )
            assert 0 <= s <= 100

    def test_score_clamped_to_range(self):
        # All-positive profile shouldn't exceed 100
        p = self._profile(
            revenue_growth=0.50, earnings_growth=0.50,
            profit_margin=0.30, roe=0.30, forward_pe=10.0,
        )
        assert 0 <= self.enricher._score_fundamentals(p) <= 100


class TestSmartMoneyBranches:
    """All branches of _score_smart_money."""

    def setup_method(self):
        self.enricher = FundamentalEnricher(data_service=AsyncMock())

    def _profile(self, **kw):
        return StockProfile(
            symbol="X",
            info=StockInfo(symbol="X", name="X"),
            consensus=ConsensusData(),
            fundamentals=FundamentalData(),
            smart_money=SmartMoneyData(**kw),
        )

    def test_institutional_high(self):
        # > 0.70 → +10
        assert self.enricher._score_smart_money(
            self._profile(institutional_pct=0.80)
        ) > 50

    def test_institutional_low(self):
        assert self.enricher._score_smart_money(
            self._profile(institutional_pct=0.20)
        ) < 50

    def test_insider_dominant_sell(self):
        # sell > buy * 2 → -10
        assert self.enricher._score_smart_money(
            self._profile(insider_buy_count_90d=1, insider_sell_count_90d=5)
        ) < 50

    def test_short_ratio_high(self):
        assert self.enricher._score_smart_money(
            self._profile(short_ratio=8.0)
        ) < 50

    def test_short_ratio_low_boost(self):
        assert self.enricher._score_smart_money(
            self._profile(short_ratio=1.0)
        ) > 50


class TestGradeBoundaries:
    """_to_grade — every threshold."""

    def test_grade_a(self):
        assert FundamentalEnricher._to_grade(85) == "A"

    def test_grade_b(self):
        assert FundamentalEnricher._to_grade(70) == "B"

    def test_grade_c(self):
        assert FundamentalEnricher._to_grade(55) == "C"

    def test_grade_d(self):
        assert FundamentalEnricher._to_grade(40) == "D"

    def test_grade_f(self):
        assert FundamentalEnricher._to_grade(20) == "F"
