"""Tests for core enums — Market and Exchange."""

from core.enums import Market, Exchange


class TestMarket:
    def test_values(self):
        assert Market.US == "US"
        assert Market.KR == "KR"

    def test_string_serialization(self):
        assert Market.US.value == "US"
        assert Market.KR.value == "KR"


class TestExchange:
    def test_us_exchanges(self):
        assert Exchange.NASD == "NASD"
        assert Exchange.NYSE == "NYSE"
        assert Exchange.AMEX == "AMEX"

    def test_kr_exchanges(self):
        assert Exchange.KRX == "KRX"
        assert Exchange.KOSDAQ == "KOSDAQ"
