"""Tests for F2 target_exposure_pct loading from markets section."""

from pathlib import Path

import yaml

from strategies.config_loader import StrategyConfigLoader


def _write(tmp_path: Path, markets: dict) -> StrategyConfigLoader:
    config = {"global": {}, "strategies": {}, "markets": markets}
    f = tmp_path / "strategies.yaml"
    f.write_text(yaml.dump(config))
    return StrategyConfigLoader(f)


class TestGetMarketTargetExposurePct:
    def test_returns_fraction_when_set(self, tmp_path: Path):
        loader = _write(tmp_path, {"US": {"target_exposure_pct": 0.70}})
        assert loader.get_market_target_exposure_pct("US") == 0.70

    def test_returns_none_when_unset(self, tmp_path: Path):
        loader = _write(tmp_path, {"US": {}})
        assert loader.get_market_target_exposure_pct("US") is None

    def test_returns_none_when_market_missing(self, tmp_path: Path):
        loader = _write(tmp_path, {"US": {"target_exposure_pct": 0.70}})
        assert loader.get_market_target_exposure_pct("KR") is None

    def test_independent_per_market(self, tmp_path: Path):
        loader = _write(
            tmp_path,
            {"US": {"target_exposure_pct": 0.70}, "KR": {"target_exposure_pct": 0.50}},
        )
        assert loader.get_market_target_exposure_pct("US") == 0.70
        assert loader.get_market_target_exposure_pct("KR") == 0.50

    def test_coerces_int_to_float(self, tmp_path: Path):
        loader = _write(tmp_path, {"US": {"target_exposure_pct": 1}})
        v = loader.get_market_target_exposure_pct("US")
        assert isinstance(v, float)
        assert v == 1.0
