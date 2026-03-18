"""Tests for Signal Combiner."""

import pytest

from strategies.combiner import SignalCombiner
from strategies.base import Signal
from core.enums import SignalType


def _signal(name: str, sig_type: SignalType, conf: float = 0.7) -> Signal:
    return Signal(
        signal_type=sig_type,
        confidence=conf,
        strategy_name=name,
        reason="test",
        indicators={"test_val": 1.0},
    )


# Default consensus config for tests
_CONSENSUS_CFG = {
    "enabled": True,
    "min_group_signals": 2,
    "consensus_boost": 0.30,
    "discord_penalty": 0.40,
    "groups": {
        "trend": [
            "trend_following", "dual_momentum", "donchian_breakout",
            "supertrend", "cis_momentum", "larry_williams",
        ],
        "mean_reversion": [
            "rsi_divergence", "bollinger_squeeze", "bnf_deviation",
            "macd_histogram", "volume_profile",
        ],
    },
}


class TestSignalCombiner:
    def test_unanimous_buy(self):
        combiner = SignalCombiner()
        signals = [
            _signal("trend_following", SignalType.BUY, 0.8),
            _signal("donchian_breakout", SignalType.BUY, 0.7),
            _signal("supertrend", SignalType.BUY, 0.9),
        ]
        weights = {
            "trend_following": 0.35,
            "donchian_breakout": 0.20,
            "supertrend": 0.20,
        }
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.BUY
        assert result.confidence > 0.5

    def test_unanimous_sell(self):
        combiner = SignalCombiner()
        signals = [
            _signal("trend_following", SignalType.SELL, 0.8),
            _signal("supertrend", SignalType.SELL, 0.7),
        ]
        weights = {"trend_following": 0.5, "supertrend": 0.5}
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.SELL

    def test_mixed_signals_buy_wins(self):
        combiner = SignalCombiner()
        signals = [
            _signal("trend_following", SignalType.BUY, 0.9),
            _signal("donchian_breakout", SignalType.BUY, 0.8),
            _signal("supertrend", SignalType.SELL, 0.6),
        ]
        weights = {
            "trend_following": 0.35,
            "donchian_breakout": 0.25,
            "supertrend": 0.20,
        }
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.BUY

    def test_below_min_confidence_hold(self):
        combiner = SignalCombiner()
        signals = [
            _signal("trend_following", SignalType.BUY, 0.3),
        ]
        weights = {"trend_following": 0.5}
        result = combiner.combine(signals, weights, min_confidence=0.5)
        assert result.signal_type == SignalType.HOLD

    def test_no_signals(self):
        combiner = SignalCombiner()
        result = combiner.combine([], {})
        assert result.signal_type == SignalType.HOLD
        assert result.confidence == 0.0

    def test_zero_weight_ignored(self):
        combiner = SignalCombiner()
        signals = [
            _signal("trend_following", SignalType.BUY, 0.9),
            _signal("ignored", SignalType.SELL, 0.9),
        ]
        weights = {"trend_following": 1.0, "ignored": 0.0}
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.BUY

    def test_unweighted_strategy_ignored(self):
        combiner = SignalCombiner()
        signals = [_signal("unknown_strat", SignalType.BUY, 0.9)]
        weights = {"trend_following": 0.5}
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.HOLD

    def test_indicators_aggregated(self):
        combiner = SignalCombiner()
        signals = [
            _signal("a", SignalType.BUY, 0.8),
            _signal("b", SignalType.BUY, 0.7),
        ]
        weights = {"a": 0.5, "b": 0.5}
        result = combiner.combine(signals, weights)
        assert "a.test_val" in result.indicators
        assert "b.test_val" in result.indicators

    def test_hold_signals_excluded_from_denominator(self):
        combiner = SignalCombiner()
        signals = [
            _signal("a", SignalType.HOLD, 0.5),
            _signal("b", SignalType.BUY, 0.8),
        ]
        weights = {"a": 0.5, "b": 0.5}
        result = combiner.combine(signals, weights)
        # HOLD excluded from denominator: buy_norm = 0.4/0.5 = 0.8 → BUY
        assert result.signal_type == SignalType.BUY
        assert result.confidence == pytest.approx(0.8)

    def test_active_ratio_too_low(self):
        combiner = SignalCombiner(min_active_ratio=0.50)
        signals = [
            _signal("a", SignalType.HOLD, 0.5),
            _signal("b", SignalType.HOLD, 0.5),
            _signal("c", SignalType.HOLD, 0.5),
            _signal("d", SignalType.BUY, 0.9),
        ]
        weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        result = combiner.combine(signals, weights)
        # Only 25% active (1/4), below 50% threshold → HOLD
        assert result.signal_type == SignalType.HOLD
        assert "Active ratio too low" in result.reason


class TestConsensusSignalCombiner:
    """Tests for group consensus mechanism."""

    def test_consensus_disabled_by_default(self):
        """Without consensus config, BUY side with higher weight wins."""
        combiner = SignalCombiner()
        # BUY weight=0.35 vs SELL weight=0.25 → BUY wins (min_confidence=0.35)
        signals = [
            _signal("rsi_divergence", SignalType.BUY, 0.7),
            _signal("bnf_deviation", SignalType.SELL, 0.7),
            _signal("bollinger_squeeze", SignalType.BUY, 0.7),
            _signal("macd_histogram", SignalType.SELL, 0.7),
        ]
        weights = {
            "rsi_divergence": 0.20,
            "bnf_deviation": 0.15,
            "bollinger_squeeze": 0.15,
            "macd_histogram": 0.10,
        }
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.BUY

    def test_true_deadlock_returns_hold(self):
        """Equal buy/sell weight with low confidence → HOLD."""
        combiner = SignalCombiner()
        signals = [
            _signal("rsi_divergence", SignalType.BUY, 0.4),
            _signal("bnf_deviation", SignalType.SELL, 0.4),
        ]
        weights = {
            "rsi_divergence": 0.15,
            "bnf_deviation": 0.15,
        }
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.HOLD

    def test_consensus_mr_3v1_buy_wins(self):
        """3 BUY vs 1 SELL in mean_reversion → BUY wins."""
        combiner = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        signals = [
            _signal("rsi_divergence", SignalType.BUY, 0.7),
            _signal("bnf_deviation", SignalType.BUY, 0.7),
            _signal("bollinger_squeeze", SignalType.BUY, 0.7),
            _signal("macd_histogram", SignalType.SELL, 0.7),
        ]
        weights = {
            "rsi_divergence": 0.20,
            "bnf_deviation": 0.15,
            "bollinger_squeeze": 0.15,
            "macd_histogram": 0.10,
        }
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.BUY

    def test_consensus_mr_3v2_buy_wins(self):
        """3 BUY vs 2 SELL in mean_reversion → deadlock broken."""
        combiner = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        signals = [
            _signal("rsi_divergence", SignalType.BUY, 0.7),
            _signal("bnf_deviation", SignalType.BUY, 0.7),
            _signal("volume_profile", SignalType.BUY, 0.7),
            _signal("bollinger_squeeze", SignalType.SELL, 0.7),
            _signal("macd_histogram", SignalType.SELL, 0.7),
        ]
        weights = {
            "rsi_divergence": 0.20,
            "bnf_deviation": 0.15,
            "volume_profile": 0.10,
            "bollinger_squeeze": 0.15,
            "macd_histogram": 0.10,
        }
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.BUY

    def test_consensus_tie_reduces_influence(self):
        """Perfect tie in mean_reversion: trend signals tip the scale."""
        combiner = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        signals = [
            # Mean reversion: 2 BUY vs 2 SELL → tie → penalized
            _signal("rsi_divergence", SignalType.BUY, 0.7),
            _signal("bnf_deviation", SignalType.BUY, 0.7),
            _signal("bollinger_squeeze", SignalType.SELL, 0.7),
            _signal("macd_histogram", SignalType.SELL, 0.7),
            # Trend: 2 BUY → not penalized, tips the scale
            _signal("trend_following", SignalType.BUY, 0.8),
            _signal("dual_momentum", SignalType.BUY, 0.8),
        ]
        weights = {
            "rsi_divergence": 0.15,
            "bnf_deviation": 0.15,
            "bollinger_squeeze": 0.15,
            "macd_histogram": 0.10,
            "trend_following": 0.25,
            "dual_momentum": 0.20,
        }
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.BUY

    def test_consensus_unanimous_boosted(self):
        """All mean_reversion BUY → boosted confidence."""
        combiner_no = SignalCombiner()
        combiner_yes = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        signals = [
            _signal("rsi_divergence", SignalType.BUY, 0.7),
            _signal("bnf_deviation", SignalType.BUY, 0.7),
            _signal("bollinger_squeeze", SignalType.BUY, 0.7),
            _signal("macd_histogram", SignalType.BUY, 0.7),
            _signal("volume_profile", SignalType.BUY, 0.7),
        ]
        weights = {
            "rsi_divergence": 0.20,
            "bnf_deviation": 0.15,
            "bollinger_squeeze": 0.15,
            "macd_histogram": 0.10,
            "volume_profile": 0.10,
        }
        r_no = combiner_no.combine(signals, weights)
        r_yes = combiner_yes.combine(signals, weights)
        assert r_no.signal_type == SignalType.BUY
        assert r_yes.signal_type == SignalType.BUY
        # Both BUY; with uniform confidence normalization cancels boost,
        # but consensus should not degrade the result
        assert abs(r_yes.confidence - r_no.confidence) < 0.01

    def test_consensus_ungrouped_unaffected(self):
        """Strategies not in any group keep original weights."""
        combiner = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        signals = [
            _signal("regime_switch", SignalType.BUY, 0.8),
            _signal("sector_rotation", SignalType.SELL, 0.8),
        ]
        weights = {"regime_switch": 0.55, "sector_rotation": 0.45}
        # These are not in any group → no consensus adjustment
        modified, _ = combiner._apply_consensus(signals, weights)
        assert modified["regime_switch"] == 0.55
        assert modified["sector_rotation"] == 0.45

    def test_consensus_single_active_no_change(self):
        """1 active signal in group (below min_group_signals) → no change."""
        combiner = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        signals = [
            _signal("rsi_divergence", SignalType.BUY, 0.7),
            # Others HOLD → not counted
            _signal("bnf_deviation", SignalType.HOLD, 0.5),
        ]
        weights = {"rsi_divergence": 0.20, "bnf_deviation": 0.15}
        modified, _ = combiner._apply_consensus(signals, weights)
        assert modified["rsi_divergence"] == 0.20

    def test_consensus_cross_group_independence(self):
        """Trend all BUY, mean_reversion all SELL → each group boosted independently."""
        combiner = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        signals = [
            _signal("trend_following", SignalType.BUY, 0.7),
            _signal("dual_momentum", SignalType.BUY, 0.7),
            _signal("rsi_divergence", SignalType.SELL, 0.7),
            _signal("bnf_deviation", SignalType.SELL, 0.7),
        ]
        weights = {
            "trend_following": 0.20,
            "dual_momentum": 0.15,
            "rsi_divergence": 0.20,
            "bnf_deviation": 0.15,
        }
        modified, scores = combiner._apply_consensus(signals, weights)
        # Trend group: unanimous BUY → boosted
        assert modified["trend_following"] > 0.20
        assert modified["dual_momentum"] > 0.15
        # Mean reversion group: unanimous SELL → boosted
        assert modified["rsi_divergence"] > 0.20
        assert modified["bnf_deviation"] > 0.15
        assert scores["trend"] == 1.0
        assert scores["mean_reversion"] == 1.0

    def test_consensus_metadata_in_indicators(self):
        """Consensus agreement scores appear in indicators."""
        combiner = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        signals = [
            _signal("rsi_divergence", SignalType.BUY, 0.7),
            _signal("bnf_deviation", SignalType.BUY, 0.7),
        ]
        weights = {"rsi_divergence": 0.20, "bnf_deviation": 0.15}
        result = combiner.combine(signals, weights)
        assert "combiner.mean_reversion_agreement" in result.indicators
        assert result.indicators["combiner.mean_reversion_agreement"] == 1.0

    def test_consensus_no_boost_zero_params(self):
        """boost=0, penalty=0 → no weight modification."""
        cfg = {**_CONSENSUS_CFG, "consensus_boost": 0.0, "discord_penalty": 0.0}
        combiner = SignalCombiner(consensus_config=cfg)
        signals = [
            _signal("rsi_divergence", SignalType.BUY, 0.7),
            _signal("bnf_deviation", SignalType.SELL, 0.7),
        ]
        weights = {"rsi_divergence": 0.20, "bnf_deviation": 0.15}
        modified, _ = combiner._apply_consensus(signals, weights)
        assert modified["rsi_divergence"] == 0.20
        assert modified["bnf_deviation"] == 0.15

    def test_existing_unanimous_buy_unbroken(self):
        """Consensus enabled doesn't break existing unanimous-buy case."""
        combiner = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        signals = [
            _signal("trend_following", SignalType.BUY, 0.8),
            _signal("donchian_breakout", SignalType.BUY, 0.7),
            _signal("supertrend", SignalType.BUY, 0.9),
        ]
        weights = {
            "trend_following": 0.35,
            "donchian_breakout": 0.20,
            "supertrend": 0.20,
        }
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.BUY
        assert result.confidence > 0.5


class TestMinActiveRatioOverride:
    """Tests for per-call min_active_ratio override (used for exit evaluation)."""

    def test_override_raises_active_ratio_threshold(self):
        """Override min_active_ratio blocks signal when too few strategies are active."""
        combiner = SignalCombiner(min_active_ratio=0.05)
        # 1 SELL out of 5 total → active_ratio = 0.20 / 1.0 = 20%
        signals = [
            _signal("a", SignalType.SELL, 0.7),
            _signal("b", SignalType.HOLD, 0.5),
            _signal("c", SignalType.HOLD, 0.5),
            _signal("d", SignalType.HOLD, 0.5),
            _signal("e", SignalType.HOLD, 0.5),
        ]
        weights = {"a": 0.20, "b": 0.20, "c": 0.20, "d": 0.20, "e": 0.20}

        # With default min_active_ratio=0.05: 20% > 5% → SELL
        result = combiner.combine(signals, weights)
        assert result.signal_type == SignalType.SELL

        # With override min_active_ratio=0.25: 20% < 25% → HOLD
        result = combiner.combine(signals, weights, min_active_ratio=0.25)
        assert result.signal_type == SignalType.HOLD
        assert "Active ratio too low" in result.reason

    def test_override_none_uses_instance_default(self):
        """Passing None for override uses the instance-level default."""
        combiner = SignalCombiner(min_active_ratio=0.50)
        signals = [
            _signal("a", SignalType.SELL, 0.7),
            _signal("b", SignalType.HOLD, 0.5),
            _signal("c", SignalType.HOLD, 0.5),
        ]
        weights = {"a": 0.33, "b": 0.33, "c": 0.34}

        # active_ratio = 0.33 / 1.0 = 33%, below instance default 50%
        result = combiner.combine(signals, weights, min_active_ratio=None)
        assert result.signal_type == SignalType.HOLD

    def test_override_allows_signal_with_lower_threshold(self):
        """Override with lower threshold allows signal through."""
        combiner = SignalCombiner(min_active_ratio=0.50)
        signals = [
            _signal("a", SignalType.SELL, 0.7),
            _signal("b", SignalType.HOLD, 0.5),
            _signal("c", SignalType.HOLD, 0.5),
        ]
        weights = {"a": 0.33, "b": 0.33, "c": 0.34}

        # Instance default blocks (33% < 50%), but override 0.10 allows it
        result = combiner.combine(signals, weights, min_active_ratio=0.10)
        assert result.signal_type == SignalType.SELL

    def test_exit_mode_sell_only_signals(self):
        """Simulates held-position exit: only SELL and HOLD signals (BUY remapped).

        This models the evaluation_loop behavior where BUY signals are
        remapped to HOLD for held positions before combining.
        """
        combiner = SignalCombiner(consensus_config=_CONSENSUS_CFG)
        # Scenario: 3 strategies vote SELL, rest are HOLD (originally BUY, remapped)
        signals = [
            # Original BUY signals, remapped to HOLD
            _signal("trend_following", SignalType.HOLD, 0.8),
            _signal("dual_momentum", SignalType.HOLD, 0.7),
            _signal("donchian_breakout", SignalType.HOLD, 0.7),
            _signal("cis_momentum", SignalType.HOLD, 0.6),
            _signal("larry_williams", SignalType.HOLD, 0.6),
            # Actual SELL signals
            _signal("supertrend", SignalType.SELL, 0.7),
            _signal("rsi_divergence", SignalType.SELL, 0.65),
            _signal("bnf_deviation", SignalType.SELL, 0.6),
            # More HOLD
            _signal("bollinger_squeeze", SignalType.HOLD, 0.5),
            _signal("macd_histogram", SignalType.HOLD, 0.5),
        ]
        weights = {
            "trend_following": 0.15,
            "dual_momentum": 0.10,
            "donchian_breakout": 0.10,
            "cis_momentum": 0.10,
            "larry_williams": 0.10,
            "supertrend": 0.10,
            "rsi_divergence": 0.15,
            "bnf_deviation": 0.10,
            "bollinger_squeeze": 0.05,
            "macd_histogram": 0.05,
        }

        # With min_active_ratio=0.15: active = SELL weight 0.35, total 1.0
        # active_ratio = 0.35 > 0.15 ✓
        # sell_norm = sell_score / 0.35, buy_norm = 0 → SELL wins
        result = combiner.combine(signals, weights, min_active_ratio=0.15)
        assert result.signal_type == SignalType.SELL
        assert result.confidence > 0.5

    def test_exit_mode_single_sell_blocked_by_active_ratio(self):
        """Single SELL signal should be blocked by exit min_active_ratio.

        Prevents noise-based exits when only 1 strategy detects exit.
        """
        combiner = SignalCombiner()
        signals = [
            _signal("supertrend", SignalType.SELL, 0.7),
            # All others are HOLD (remapped from BUY)
            _signal("trend_following", SignalType.HOLD, 0.8),
            _signal("dual_momentum", SignalType.HOLD, 0.7),
            _signal("donchian_breakout", SignalType.HOLD, 0.7),
            _signal("rsi_divergence", SignalType.HOLD, 0.6),
            _signal("bnf_deviation", SignalType.HOLD, 0.6),
            _signal("bollinger_squeeze", SignalType.HOLD, 0.5),
            _signal("macd_histogram", SignalType.HOLD, 0.5),
            _signal("cis_momentum", SignalType.HOLD, 0.6),
            _signal("larry_williams", SignalType.HOLD, 0.6),
        ]
        weights = {
            "supertrend": 0.10,
            "trend_following": 0.15,
            "dual_momentum": 0.10,
            "donchian_breakout": 0.10,
            "rsi_divergence": 0.10,
            "bnf_deviation": 0.10,
            "bollinger_squeeze": 0.10,
            "macd_histogram": 0.05,
            "cis_momentum": 0.10,
            "larry_williams": 0.10,
        }

        # active_ratio = 0.10 / 1.0 = 10% < 15% → HOLD
        result = combiner.combine(signals, weights, min_active_ratio=0.15)
        assert result.signal_type == SignalType.HOLD
        assert "Active ratio too low" in result.reason

    def test_exit_mode_two_sells_pass_active_ratio(self):
        """Two SELL signals should pass 15% exit threshold with typical weights."""
        combiner = SignalCombiner()
        signals = [
            _signal("supertrend", SignalType.SELL, 0.7),
            _signal("trend_following", SignalType.SELL, 0.65),
            # Rest are HOLD (remapped from BUY)
            _signal("dual_momentum", SignalType.HOLD, 0.7),
            _signal("donchian_breakout", SignalType.HOLD, 0.7),
            _signal("rsi_divergence", SignalType.HOLD, 0.6),
            _signal("bnf_deviation", SignalType.HOLD, 0.6),
        ]
        weights = {
            "supertrend": 0.10,
            "trend_following": 0.15,
            "dual_momentum": 0.15,
            "donchian_breakout": 0.10,
            "rsi_divergence": 0.15,
            "bnf_deviation": 0.10,
        }

        # active_ratio = 0.25 / 0.75 = 33% > 15% → SELL
        result = combiner.combine(signals, weights, min_active_ratio=0.15)
        assert result.signal_type == SignalType.SELL

    def test_without_remap_buy_drowns_sell(self):
        """Demonstrates the original bug: BUY signals drown out SELL for held positions.

        Before the fix, trend strategies emitting BUY for a held position
        would overpower SELL signals from exit-detecting strategies.
        """
        combiner = SignalCombiner()
        # Realistic scenario: 5 BUY (trend up) vs 3 SELL (exit detected)
        signals = [
            _signal("trend_following", SignalType.BUY, 0.8),
            _signal("dual_momentum", SignalType.BUY, 0.7),
            _signal("donchian_breakout", SignalType.BUY, 0.7),
            _signal("cis_momentum", SignalType.BUY, 0.6),
            _signal("larry_williams", SignalType.BUY, 0.6),
            _signal("supertrend", SignalType.SELL, 0.7),
            _signal("rsi_divergence", SignalType.SELL, 0.65),
            _signal("bnf_deviation", SignalType.SELL, 0.6),
        ]
        weights = {
            "trend_following": 0.15,
            "dual_momentum": 0.10,
            "donchian_breakout": 0.10,
            "cis_momentum": 0.10,
            "larry_williams": 0.10,
            "supertrend": 0.10,
            "rsi_divergence": 0.15,
            "bnf_deviation": 0.10,
        }

        # Without remap: BUY wins because more weight on BUY side
        result_no_remap = combiner.combine(signals, weights)
        assert result_no_remap.signal_type == SignalType.BUY

        # With remap (BUY→HOLD): only SELL signals are active → SELL wins
        remapped = [
            Signal(
                signal_type=SignalType.HOLD,
                confidence=s.confidence,
                strategy_name=s.strategy_name,
                reason=s.reason,
                indicators=s.indicators,
            )
            if s.signal_type == SignalType.BUY
            else s
            for s in signals
        ]
        result_remapped = combiner.combine(remapped, weights, min_active_ratio=0.15)
        assert result_remapped.signal_type == SignalType.SELL
