"""Tests for services.yaml_mutator (A4 — accept-time yaml apply)."""

from pathlib import Path

import pytest
import yaml

from services.yaml_mutator import (
    YamlMutationError,
    _is_path_allowed,
    apply_yaml_change,
)


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    p = tmp_path / "strategies.yaml"
    p.write_text(yaml.safe_dump({
        "markets": {
            "KR": {
                "disabled_strategies": ["trend_following"],
                "evaluation_loop": {
                    "sector_boost_weight": 0.3,
                    "opening_avoidance_minutes": 30,
                },
                "risk": {
                    "max_positions": 18,
                    "max_position_pct": 0.20,
                },
            },
            "US": {
                "disabled_strategies": ["dual_momentum"],
                "evaluation_loop": {
                    "daily_buy_limit": 10,
                },
            },
        },
        "tiered_trailing_stop": {
            "enabled": True,
            "tiers": [
                {"gain_pct": 0.05, "trail_pct": 0.03},
                {"gain_pct": 0.10, "trail_pct": 0.05},
            ],
        },
        "internal_secret": "do not touch",
    }), encoding="utf-8")
    return p


class TestPathWhitelist:
    def test_allowed_paths(self):
        assert _is_path_allowed("markets.KR.evaluation_loop.sector_boost_weight")
        assert _is_path_allowed("markets.US.disabled_strategies")
        assert _is_path_allowed("markets.KR.risk.max_positions")
        assert _is_path_allowed("tiered_trailing_stop.tiers")

    def test_disallowed_paths(self):
        assert not _is_path_allowed("internal_secret")
        assert not _is_path_allowed("markets.KR.unknown_section")
        assert not _is_path_allowed("markets.KR.risk.kelly_fraction")  # restart-required


class TestApplyYamlChange:
    def test_int_change(self, yaml_file):
        old, new = apply_yaml_change(
            yaml_file, "markets.KR.risk.max_positions", 22,
        )
        assert old == 18
        assert new == 22
        loaded = yaml.safe_load(yaml_file.read_text())
        assert loaded["markets"]["KR"]["risk"]["max_positions"] == 22

    def test_float_change(self, yaml_file):
        apply_yaml_change(
            yaml_file, "markets.KR.evaluation_loop.sector_boost_weight", 0.5,
        )
        loaded = yaml.safe_load(yaml_file.read_text())
        assert loaded["markets"]["KR"]["evaluation_loop"]["sector_boost_weight"] == 0.5

    def test_list_change(self, yaml_file):
        apply_yaml_change(
            yaml_file,
            "markets.KR.disabled_strategies",
            ["trend_following", "supertrend"],
        )
        loaded = yaml.safe_load(yaml_file.read_text())
        assert loaded["markets"]["KR"]["disabled_strategies"] == [
            "trend_following", "supertrend",
        ]

    def test_creates_backup(self, yaml_file):
        apply_yaml_change(yaml_file, "markets.KR.risk.max_positions", 25)
        backup = yaml_file.with_suffix(yaml_file.suffix + ".bak")
        assert backup.exists()
        # Backup contains the pre-change value
        bak_data = yaml.safe_load(backup.read_text())
        assert bak_data["markets"]["KR"]["risk"]["max_positions"] == 18

    def test_disallowed_path_raises(self, yaml_file):
        with pytest.raises(YamlMutationError, match="whitelist"):
            apply_yaml_change(yaml_file, "internal_secret", "changed")

    def test_type_mismatch_raises(self, yaml_file):
        with pytest.raises(YamlMutationError, match="type mismatch"):
            apply_yaml_change(
                yaml_file, "markets.KR.risk.max_positions", "twenty",
            )

    def test_missing_path_raises(self, yaml_file):
        with pytest.raises(YamlMutationError, match="not found"):
            apply_yaml_change(
                yaml_file,
                "markets.KR.evaluation_loop.does_not_exist",
                42,
            )

    def test_int_to_float_allowed(self, yaml_file):
        # int existing → float proposed (both numeric) — allowed
        apply_yaml_change(yaml_file, "markets.KR.risk.max_positions", 18.5)
        loaded = yaml.safe_load(yaml_file.read_text())
        assert loaded["markets"]["KR"]["risk"]["max_positions"] == 18.5

    def test_atomic_no_temp_left(self, yaml_file, tmp_path):
        apply_yaml_change(yaml_file, "markets.KR.risk.max_positions", 30)
        # tmp file should be removed (renamed to target)
        tmp = yaml_file.with_suffix(yaml_file.suffix + ".tmp")
        assert not tmp.exists()
