"""Tests for ETFUniverse manager."""

import pytest
from pathlib import Path

from scanner.etf_universe import ETFUniverse


@pytest.fixture
def universe():
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "etf_universe.yaml"
    return ETFUniverse(config_path)


def test_load_pairs(universe):
    pair = universe.get_pair("QQQ")
    assert pair is not None
    assert pair.bull == "TQQQ"
    assert pair.bear == "SQQQ"
    assert pair.leverage == 3


def test_load_sectors(universe):
    sectors = universe.get_all_sectors()
    assert len(sectors) >= 11
    tech = universe.get_sector("Technology")
    assert tech is not None
    assert tech.etf == "XLK"
    assert "AAPL" in tech.top_holdings


def test_get_bull_bear_etfs(universe):
    assert universe.get_bull_etf("SPY") == "UPRO"
    assert universe.get_bear_etf("SPY") == "SPXU"
    assert universe.get_bull_etf("UNKNOWN") is None


def test_get_regime_etfs(universe):
    bull_etfs = universe.get_regime_etfs("bull")
    assert "TQQQ" in bull_etfs
    assert "SOXL" in bull_etfs

    bear_etfs = universe.get_regime_etfs("bear")
    assert "SQQQ" in bear_etfs
    assert "SOXS" in bear_etfs


def test_sector_etf_symbols(universe):
    symbols = universe.get_sector_etf_symbols()
    assert "XLK" in symbols
    assert "XLF" in symbols
    assert len(symbols) == 11


def test_sector_for_symbol(universe):
    assert universe.get_sector_for_symbol("AAPL") == "Technology"
    assert universe.get_sector_for_symbol("JPM") == "Financials"
    assert universe.get_sector_for_symbol("UNKNOWN") is None


def test_is_leveraged(universe):
    assert universe.is_leveraged("TQQQ") is True
    assert universe.is_leveraged("SQQQ") is True
    assert universe.is_leveraged("AAPL") is False
    assert universe.is_leveraged("QQQ") is False  # base, not leveraged


def test_risk_rules(universe):
    rules = universe.risk_rules
    assert rules.max_hold_days == 20
    # 2026-05-07: 0.20 → 0.10 (conservative ETF re-enable)
    assert rules.max_portfolio_pct == 0.10
    assert rules.require_stop_loss is True


def test_safe_haven(universe):
    assert "SHY" in universe.safe_haven
    assert "GLD" in universe.safe_haven


def test_all_etf_symbols(universe):
    all_syms = universe.all_etf_symbols
    assert len(all_syms) > 20
    assert "TQQQ" in all_syms
    assert "XLK" in all_syms
    assert "GLD" in all_syms


def test_pair_siblings_us(universe):
    """Bull/bear/base siblings for US ETFs (base = index key)."""
    siblings = universe.get_pair_siblings("TQQQ")
    assert "SQQQ" in siblings
    assert "QQQ" in siblings
    assert "TQQQ" not in siblings

    siblings_base = universe.get_pair_siblings("QQQ")
    assert "TQQQ" in siblings_base
    assert "SQQQ" in siblings_base


def test_pair_siblings_kr():
    """KR ETFs: base symbol from YAML (069500), bull (122630), bear (114800)."""
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "kr_etf_universe.yaml"
    kr = ETFUniverse(config_path)
    # Base (KODEX 200) → bull + bear
    siblings = kr.get_pair_siblings("069500")
    assert "122630" in siblings
    assert "114800" in siblings
    # Bull (KODEX 레버리지) → base + bear
    siblings = kr.get_pair_siblings("122630")
    assert "069500" in siblings
    assert "114800" in siblings
    # Unknown → empty
    assert kr.get_pair_siblings("999999") == []


def test_missing_config():
    universe = ETFUniverse("/nonexistent/path.yaml")
    assert len(universe.all_etf_symbols) == 0
    assert universe.get_pair("QQQ") is None
