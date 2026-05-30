"""Tests for services.yaml_mutator (A4 — accept-time yaml apply)."""

from pathlib import Path

import pytest
import yaml

from services.yaml_mutator import (
    YamlMutationError,
    _is_path_allowed,
    apply_yaml_change,
    path_exists,
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

    def test_intermediate_path_missing_raises(self, tmp_path):
        """Whitelisted prefix but an interior key doesn't exist — walk
        raises before we even reach the leaf check."""
        p = tmp_path / "s.yaml"
        # markets.KR exists but markets.KR.risk is missing
        p.write_text(yaml.safe_dump({"markets": {"KR": {}}}), encoding="utf-8")
        with pytest.raises(YamlMutationError, match="yaml path not found"):
            apply_yaml_change(p, "markets.KR.risk.max_positions", 18)

    def test_intermediate_path_not_mapping_raises(self, tmp_path):
        """Walk hits a scalar where a dict was expected."""
        p = tmp_path / "s.yaml"
        # markets.KR.risk is a string, not a dict
        p.write_text(
            yaml.safe_dump({"markets": {"KR": {"risk": "broken"}}}),
            encoding="utf-8",
        )
        with pytest.raises(YamlMutationError, match="not a mapping"):
            apply_yaml_change(p, "markets.KR.risk.max_positions", 18)

    def test_yaml_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "nope.yaml"
        with pytest.raises(YamlMutationError, match="yaml file not found"):
            apply_yaml_change(missing, "markets.KR.risk.max_positions", 18)

    def test_yaml_parse_failure_raises(self, tmp_path):
        p = tmp_path / "broken.yaml"
        p.write_text("foo: [unclosed", encoding="utf-8")
        with pytest.raises(YamlMutationError, match="yaml parse failed"):
            apply_yaml_change(p, "markets.KR.risk.max_positions", 18)

    def test_yaml_root_not_mapping_raises(self, tmp_path):
        p = tmp_path / "list.yaml"
        # Root is a list, not a dict — apply must refuse.
        p.write_text(yaml.safe_dump(["a", "b"]), encoding="utf-8")
        with pytest.raises(YamlMutationError, match="root is not a mapping"):
            apply_yaml_change(p, "markets.KR.risk.max_positions", 18)


class TestTypeMatching:
    """Cover every branch of _types_match (bool / numeric / str / list /
    dict / None) via apply_yaml_change."""

    def _yaml_with(self, tmp_path, value):
        p = tmp_path / "s.yaml"
        p.write_text(
            yaml.safe_dump({"markets": {"KR": {"evaluation_loop": {"k": value}}}}),
            encoding="utf-8",
        )
        return p

    def test_bool_to_bool_ok(self, tmp_path):
        p = self._yaml_with(tmp_path, True)
        apply_yaml_change(p, "markets.KR.evaluation_loop.k", False)
        assert yaml.safe_load(p.read_text())["markets"]["KR"]["evaluation_loop"]["k"] is False

    def test_bool_to_int_rejected(self, tmp_path):
        p = self._yaml_with(tmp_path, True)
        with pytest.raises(YamlMutationError, match="type mismatch"):
            apply_yaml_change(p, "markets.KR.evaluation_loop.k", 1)

    def test_int_to_bool_rejected(self, tmp_path):
        p = self._yaml_with(tmp_path, 1)
        with pytest.raises(YamlMutationError, match="type mismatch"):
            apply_yaml_change(p, "markets.KR.evaluation_loop.k", True)

    def test_str_to_str_ok(self, tmp_path):
        p = self._yaml_with(tmp_path, "SPY")
        apply_yaml_change(p, "markets.KR.evaluation_loop.k", "KODEX")
        assert yaml.safe_load(p.read_text())["markets"]["KR"]["evaluation_loop"]["k"] == "KODEX"

    def test_str_to_int_rejected(self, tmp_path):
        p = self._yaml_with(tmp_path, "label")
        with pytest.raises(YamlMutationError, match="type mismatch"):
            apply_yaml_change(p, "markets.KR.evaluation_loop.k", 99)

    def test_list_to_list_ok(self, tmp_path):
        p = self._yaml_with(tmp_path, [1, 2])
        apply_yaml_change(p, "markets.KR.evaluation_loop.k", [3, 4, 5])
        assert yaml.safe_load(p.read_text())["markets"]["KR"]["evaluation_loop"]["k"] == [3, 4, 5]

    def test_dict_to_dict_ok(self, tmp_path):
        p = self._yaml_with(tmp_path, {"x": 1})
        apply_yaml_change(p, "markets.KR.evaluation_loop.k", {"y": 2})
        assert yaml.safe_load(p.read_text())["markets"]["KR"]["evaluation_loop"]["k"] == {"y": 2}

    def test_dict_to_list_rejected(self, tmp_path):
        p = self._yaml_with(tmp_path, {"x": 1})
        with pytest.raises(YamlMutationError, match="type mismatch"):
            apply_yaml_change(p, "markets.KR.evaluation_loop.k", [1, 2])

    def test_none_existing_accepts_any(self, tmp_path):
        """None on either side bypasses type-match (operator initializing)."""
        p = self._yaml_with(tmp_path, None)
        apply_yaml_change(p, "markets.KR.evaluation_loop.k", 0.5)
        assert yaml.safe_load(p.read_text())["markets"]["KR"]["evaluation_loop"]["k"] == 0.5

    def test_none_proposed_accepts(self, tmp_path):
        """Caller can clear a value by proposing None."""
        p = self._yaml_with(tmp_path, 42)
        apply_yaml_change(p, "markets.KR.evaluation_loop.k", None)
        assert yaml.safe_load(p.read_text())["markets"]["KR"]["evaluation_loop"]["k"] is None


class TestPathExists:
    """path_exists — phantom-path filter for LLM hallucinations."""

    def test_returns_true_for_real_path(self, yaml_file):
        assert path_exists(yaml_file, "markets.KR.evaluation_loop.sector_boost_weight") is True
        assert path_exists(yaml_file, "markets.KR.risk.max_positions") is True
        assert path_exists(yaml_file, "markets.US.disabled_strategies") is True

    def test_returns_false_for_phantom_leaf(self, yaml_file):
        # Real LLM hallucinations seen 2026-05-29:
        assert path_exists(
            yaml_file, "markets.KR.evaluation_loop.sell_cooldown_hours",
        ) is False
        assert path_exists(
            yaml_file, "markets.US.evaluation_loop.same_signal_dedup_hours",
        ) is False

    def test_returns_false_for_missing_intermediate(self, yaml_file):
        assert path_exists(yaml_file, "markets.JP.evaluation_loop.x") is False
        assert path_exists(yaml_file, "markets.KR.bogus_section.x") is False

    def test_returns_false_for_missing_file(self, tmp_path):
        assert path_exists(tmp_path / "nope.yaml", "markets.KR.x") is False

    def test_returns_false_for_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("not: valid: yaml: at: all:\n  - mixed types: [{", encoding="utf-8")
        assert path_exists(p, "markets.KR.x") is False

    def test_returns_true_for_path_to_dict_node(self, yaml_file):
        # Path resolving to a sub-dict (not a leaf) is still a real path
        assert path_exists(yaml_file, "markets.KR.evaluation_loop") is True

    def test_returns_true_for_path_to_list_node(self, yaml_file):
        assert path_exists(yaml_file, "markets.KR.disabled_strategies") is True
