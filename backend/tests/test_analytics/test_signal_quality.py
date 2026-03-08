"""Tests for Signal Quality & Confidence Calibration."""

import time

import pytest

from analytics.signal_quality import SignalQualityTracker, StrategyMetrics, TradeRecord


@pytest.fixture
def tracker():
    return SignalQualityTracker(min_trades_for_gating=5)


def _seed_trades(tracker, strategy, wins, losses):
    """Helper to seed trade records."""
    for _ in range(wins):
        tracker.record_trade(strategy, "AAPL", 0.08)
    for _ in range(losses):
        tracker.record_trade(strategy, "AAPL", -0.04)


class TestTradeRecording:
    def test_record_single_trade(self, tracker):
        tracker.record_trade("dual_momentum", "AAPL", 0.05)
        metrics = tracker.get_metrics("dual_momentum")
        assert metrics.total_trades == 1

    def test_max_trades_limit(self):
        tracker = SignalQualityTracker(max_trades_per_strategy=10)
        for i in range(20):
            tracker.record_trade("test", "AAPL", 0.01)
        metrics = tracker.get_metrics("test")
        assert metrics.total_trades == 10

    def test_empty_strategy(self, tracker):
        metrics = tracker.get_metrics("nonexistent")
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0
        assert metrics.quality_score == 0


class TestMetricsCalculation:
    def test_win_rate(self, tracker):
        _seed_trades(tracker, "strat", wins=6, losses=4)
        metrics = tracker.get_metrics("strat")
        assert metrics.win_rate == 0.6

    def test_avg_win_loss(self, tracker):
        _seed_trades(tracker, "strat", wins=5, losses=5)
        metrics = tracker.get_metrics("strat")
        assert metrics.avg_win == 0.08
        assert metrics.avg_loss == 0.04

    def test_profit_factor(self, tracker):
        _seed_trades(tracker, "strat", wins=6, losses=4)
        metrics = tracker.get_metrics("strat")
        # 6 * 0.08 = 0.48 profit, 4 * 0.04 = 0.16 loss
        assert metrics.profit_factor == pytest.approx(0.48 / 0.16, rel=0.01)

    def test_all_wins(self, tracker):
        _seed_trades(tracker, "winner", wins=10, losses=0)
        metrics = tracker.get_metrics("winner")
        assert metrics.win_rate == 1.0
        assert metrics.profit_factor == 10.0  # capped at 10.0

    def test_all_losses(self, tracker):
        _seed_trades(tracker, "loser", wins=0, losses=10)
        metrics = tracker.get_metrics("loser")
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0


class TestEdgeDetection:
    def test_has_edge(self, tracker):
        _seed_trades(tracker, "good", wins=14, losses=6)  # 20 trades, pf > 1.0
        assert tracker.get_metrics("good").has_edge is True

    def test_no_edge_low_win_rate(self, tracker):
        _seed_trades(tracker, "bad", wins=4, losses=16)  # 20 trades, pf < 1.0
        assert tracker.get_metrics("bad").has_edge is False

    def test_no_edge_insufficient_trades(self, tracker):
        _seed_trades(tracker, "new", wins=10, losses=5)
        assert tracker.get_metrics("new").has_edge is False  # 15 < 20 trades


class TestStrategyGating:
    def test_active_strategies(self, tracker):
        _seed_trades(tracker, "good", wins=14, losses=6)  # 20 trades, pf > 1.0
        _seed_trades(tracker, "bad", wins=4, losses=16)   # 20 trades, pf < 1.0
        active = tracker.get_active_strategies()
        assert "good" in active
        assert "bad" not in active

    def test_gated_strategies(self, tracker):
        _seed_trades(tracker, "good", wins=14, losses=6)  # 20 trades, pf > 1.0
        _seed_trades(tracker, "bad", wins=4, losses=16)   # 20 trades, pf < 1.0
        gated = tracker.get_gated_strategies()
        assert "bad" in gated
        assert "good" not in gated

    def test_new_strategy_not_gated(self, tracker):
        _seed_trades(tracker, "new", wins=1, losses=3)
        gated = tracker.get_gated_strategies()
        assert "new" not in gated  # < min_trades, not enough data to gate


class TestStrategyWeights:
    def test_weight_proportional_to_quality(self, tracker):
        _seed_trades(tracker, "great", wins=16, losses=4)  # 20 trades
        _seed_trades(tracker, "ok", wins=12, losses=8)      # 20 trades
        weights = tracker.get_strategy_weights()
        assert weights["great"] > weights["ok"]

    def test_no_edge_gets_zero_weight(self, tracker):
        _seed_trades(tracker, "good", wins=14, losses=6)   # 20 trades, pf > 1.0
        _seed_trades(tracker, "bad", wins=4, losses=16)    # 20 trades, pf < 1.0
        weights = tracker.get_strategy_weights()
        assert weights["bad"] == 0.0
        assert weights["good"] > 0

    def test_weights_sum_to_one(self, tracker):
        _seed_trades(tracker, "a", wins=14, losses=6)      # 20 trades
        _seed_trades(tracker, "b", wins=12, losses=8)      # 20 trades
        weights = tracker.get_strategy_weights()
        non_zero = {k: v for k, v in weights.items() if v > 0}
        assert abs(sum(non_zero.values()) - 1.0) < 1e-9


class TestKellyInputs:
    def test_kelly_inputs(self, tracker):
        _seed_trades(tracker, "strat", wins=6, losses=4)
        metrics = tracker.get_metrics("strat")
        wr, aw, al = metrics.kelly_inputs
        assert wr == 0.6
        assert aw == 0.08
        assert al == 0.04


class TestQualityScore:
    def test_high_quality_strategy(self, tracker):
        _seed_trades(tracker, "good", wins=40, losses=10)
        metrics = tracker.get_metrics("good")
        assert metrics.quality_score > 50

    def test_low_quality_strategy(self, tracker):
        _seed_trades(tracker, "bad", wins=5, losses=15)
        metrics = tracker.get_metrics("bad")
        assert metrics.quality_score < 30

    def test_few_trades_low_quality(self, tracker):
        _seed_trades(tracker, "new", wins=1, losses=1)
        metrics = tracker.get_metrics("new")
        assert metrics.quality_score == 0  # < 3 trades


class TestLookback:
    def test_old_trades_excluded(self):
        tracker = SignalQualityTracker(lookback_days=30)
        # Add old trade
        record = TradeRecord(
            strategy="old_strat", symbol="AAPL",
            return_pct=0.10,
            timestamp=time.time() - 60 * 86400,  # 60 days ago
        )
        tracker._trades["old_strat"].append(record)

        # Should not count
        metrics = tracker.get_metrics("old_strat")
        assert metrics.total_trades == 0
