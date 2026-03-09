"""Tests for Exchange Resolver."""

from unittest.mock import MagicMock, patch

from services.exchange_resolver import ExchangeResolver, _YF_TO_KIS


class TestExchangeResolver:
    def test_cache_hit(self):
        resolver = ExchangeResolver()
        resolver.set("AAPL", "NASD")
        assert resolver.resolve("AAPL") == "NASD"

    def test_preload(self):
        resolver = ExchangeResolver()
        resolver.preload({"XLE": "AMEX", "XLU": "AMEX", "TQQQ": "NASD"})
        assert resolver.resolve("XLE") == "AMEX"
        assert resolver.resolve("XLU") == "AMEX"
        assert resolver.resolve("TQQQ") == "NASD"

    def test_set_overrides(self):
        resolver = ExchangeResolver()
        resolver.set("TEST", "NYSE")
        assert resolver.resolve("TEST") == "NYSE"
        resolver.set("TEST", "AMEX")
        assert resolver.resolve("TEST") == "AMEX"

    @patch("services.exchange_resolver.yf")
    def test_yfinance_nms_maps_to_nasd(self, mock_yf):
        mock_fast_info = MagicMock()
        mock_fast_info.exchange = "NMS"
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        mock_yf.Ticker.return_value = mock_ticker

        resolver = ExchangeResolver()
        assert resolver.resolve("AAPL") == "NASD"

    @patch("services.exchange_resolver.yf")
    def test_yfinance_nyq_maps_to_nyse(self, mock_yf):
        mock_fast_info = MagicMock()
        mock_fast_info.exchange = "NYQ"
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        mock_yf.Ticker.return_value = mock_ticker

        resolver = ExchangeResolver()
        assert resolver.resolve("JPM") == "NYSE"

    @patch("services.exchange_resolver.yf")
    def test_yfinance_pcx_maps_to_amex(self, mock_yf):
        mock_fast_info = MagicMock()
        mock_fast_info.exchange = "PCX"
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        mock_yf.Ticker.return_value = mock_ticker

        resolver = ExchangeResolver()
        assert resolver.resolve("XLE") == "AMEX"

    @patch("services.exchange_resolver.yf")
    def test_yfinance_error_defaults_to_nasd(self, mock_yf):
        mock_yf.Ticker.side_effect = Exception("Network error")

        resolver = ExchangeResolver()
        assert resolver.resolve("UNKNOWN") == "NASD"

    @patch("services.exchange_resolver.yf")
    def test_unknown_exchange_defaults_to_nasd(self, mock_yf):
        mock_fast_info = MagicMock()
        mock_fast_info.exchange = "UNKNOWN_EXCHANGE"
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        mock_yf.Ticker.return_value = mock_ticker

        resolver = ExchangeResolver()
        assert resolver.resolve("X") == "NASD"

    @patch("services.exchange_resolver.yf")
    def test_caches_yfinance_result(self, mock_yf):
        mock_fast_info = MagicMock()
        mock_fast_info.exchange = "NYQ"
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        mock_yf.Ticker.return_value = mock_ticker

        resolver = ExchangeResolver()
        assert resolver.resolve("JPM") == "NYSE"
        assert resolver.resolve("JPM") == "NYSE"  # second call uses cache
        mock_yf.Ticker.assert_called_once()  # only 1 yfinance call

    def test_all_yf_codes_mapped(self):
        """All expected yfinance exchange codes have KIS mappings."""
        expected = {"NMS", "NGM", "NCM", "NYQ", "PCX", "ASE", "BTS"}
        assert set(_YF_TO_KIS.keys()) == expected
