"""Tests for Korean stock symbol mapper."""

from data.kr_symbol_mapper import (
    to_yfinance,
    from_yfinance,
    is_kr_symbol,
    normalize_kr_symbol,
)


class TestToYfinance:
    def test_kospi(self):
        assert to_yfinance("005930", "KRX") == "005930.KS"

    def test_kosdaq(self):
        assert to_yfinance("035420", "KOSDAQ") == "035420.KQ"

    def test_default_kospi(self):
        assert to_yfinance("005930") == "005930.KS"


class TestFromYfinance:
    def test_kospi(self):
        sym, exch = from_yfinance("005930.KS")
        assert sym == "005930"
        assert exch == "KRX"

    def test_kosdaq(self):
        sym, exch = from_yfinance("035420.KQ")
        assert sym == "035420"
        assert exch == "KOSDAQ"

    def test_no_suffix(self):
        sym, exch = from_yfinance("005930")
        assert sym == "005930"
        assert exch == "KRX"


class TestIsKrSymbol:
    def test_valid(self):
        assert is_kr_symbol("005930") is True
        assert is_kr_symbol("035420") is True

    def test_us_symbol(self):
        assert is_kr_symbol("AAPL") is False
        assert is_kr_symbol("MSFT") is False

    def test_too_short(self):
        assert is_kr_symbol("12345") is False

    def test_too_long(self):
        assert is_kr_symbol("1234567") is False


class TestNormalizeKrSymbol:
    def test_already_padded(self):
        assert normalize_kr_symbol("005930") == "005930"

    def test_needs_padding(self):
        assert normalize_kr_symbol("5930") == "005930"

    def test_alpha_unchanged(self):
        assert normalize_kr_symbol("AAPL") == "AAPL"
