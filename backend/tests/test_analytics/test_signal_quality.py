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


class TestMaxDrawdown:
    def test_no_drawdown_all_wins(self, tracker):
        _seed_trades(tracker, "winner", wins=10, losses=0)
        metrics = tracker.get_metrics("winner")
        assert metrics.max_drawdown == 0.0

    def test_drawdown_after_losses(self, tracker):
        # Win then lose: cumulative goes up then down
        tracker.record_trade("strat", "AAPL", 0.10)  # +10%
        tracker.record_trade("strat", "AAPL", -0.08)  # -8%
        tracker.record_trade("strat", "AAPL", -0.05)  # -5%
        metrics = tracker.get_metrics("strat")
        assert metrics.max_drawdown > 0.10  # significant drawdown

    def test_drawdown_single_trade(self, tracker):
        tracker.record_trade("strat", "AAPL", -0.05)
        metrics = tracker.get_metrics("strat")
        assert metrics.max_drawdown == 0.0  # < 2 trades → 0

    def test_drawdown_two_losses(self, tracker):
        tracker.record_trade("strat", "AAPL", -0.05)
        tracker.record_trade("strat", "AAPL", -0.05)
        metrics = tracker.get_metrics("strat")
        # cumulative: 0.95 → 0.9025, peak=1.0, dd = (1.0-0.9025)/1.0 ≈ 0.0975
        assert metrics.max_drawdown == pytest.approx(0.0975, abs=0.001)

    def test_recovery_resets_peak(self, tracker):
        tracker.record_trade("strat", "AAPL", 0.20)   # up 20%
        tracker.record_trade("strat", "AAPL", -0.10)  # down 10%
        tracker.record_trade("strat", "AAPL", 0.20)   # back up
        metrics = tracker.get_metrics("strat")
        # Drawdown should be from peak after first win
        assert metrics.max_drawdown > 0
        assert metrics.max_drawdown < 0.15


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


class TestSerialization:
    """Tests for to_dict / load_dict / seed_from_trades.

    These power live → backtest state injection so the backtest's gating
    + Kelly inputs match live behavior instead of starting cold.
    """

    def test_to_dict_round_trip(self, tracker):
        _seed_trades(tracker, "dual_momentum", wins=8, losses=4)
        _seed_trades(tracker, "trend_following", wins=5, losses=5)

        snapshot = tracker.to_dict()
        assert snapshot["version"] == 1
        assert "dual_momentum" in snapshot["trades"]
        assert "trend_following" in snapshot["trades"]
        assert len(snapshot["trades"]["dual_momentum"]) == 12

        # Load into a fresh tracker
        fresh = SignalQualityTracker()
        n = fresh.load_dict(snapshot)
        assert n == 22

        # Metrics match the original
        m_orig = tracker.get_metrics("dual_momentum")
        m_fresh = fresh.get_metrics("dual_momentum")
        assert m_fresh.total_trades == m_orig.total_trades
        assert m_fresh.win_rate == m_orig.win_rate
        assert m_fresh.profit_factor == m_orig.profit_factor

    def test_load_dict_replaces_existing(self, tracker):
        _seed_trades(tracker, "old", wins=3, losses=3)
        snapshot = {
            "version": 1,
            "trades": {
                "new_strat": [
                    {"symbol": "AAPL", "return_pct": 0.05, "timestamp": time.time()},
                    {"symbol": "MSFT", "return_pct": -0.02, "timestamp": time.time()},
                ],
            },
        }
        tracker.load_dict(snapshot)
        # "old" gone, "new_strat" present
        assert tracker.get_metrics("old").total_trades == 0
        assert tracker.get_metrics("new_strat").total_trades == 2

    def test_load_dict_rejects_non_dict(self, tracker):
        with pytest.raises(ValueError):
            tracker.load_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_load_dict_skips_malformed_records(self, tracker):
        now = time.time()
        snapshot = {
            "trades": {
                "s1": [
                    {"symbol": "AAPL", "return_pct": 0.05, "timestamp": now},
                    {"missing": "fields"},  # skipped (no return_pct)
                    "not a dict",            # skipped (wrong type)
                    {"symbol": "MSFT", "return_pct": "not a number"},  # skipped (bad cast)
                ],
            },
        }
        n = tracker.load_dict(snapshot)
        assert n == 1
        assert tracker.get_metrics("s1").total_trades == 1

    def test_load_dict_respects_max_trades(self):
        tracker = SignalQualityTracker(max_trades_per_strategy=5)
        snapshot = {
            "trades": {
                "s1": [
                    {"symbol": "X", "return_pct": float(i % 2 * 2 - 1) * 0.01,
                     "timestamp": time.time() - i}
                    for i in range(20)
                ],
            },
        }
        tracker.load_dict(snapshot)
        # Only the most recent 5 are kept
        assert len(tracker._trades["s1"]) == 5

    def test_seed_from_trades_dicts(self):
        tracker = SignalQualityTracker()
        records = [
            {"strategy": "supertrend", "symbol": "AAPL",
             "return_pct": 0.06, "timestamp": time.time()},
            {"strategy": "supertrend", "symbol": "MSFT",
             "return_pct": -0.03, "timestamp": time.time()},
            {"strategy": "dual_momentum", "symbol": "NVDA",
             "return_pct": 0.10, "timestamp": time.time()},
        ]
        n = tracker.seed_from_trades(records)
        assert n == 3
        assert tracker.get_metrics("supertrend").total_trades == 2
        assert tracker.get_metrics("dual_momentum").total_trades == 1

    def test_seed_from_trades_skips_bad_records(self):
        tracker = SignalQualityTracker()
        records = [
            {"strategy": "s1", "return_pct": 0.05},
            {"strategy": "s1", "return_pct": "bad"},  # skipped
            {"return_pct": 0.05},  # missing strategy → skipped
            "not a dict",  # skipped
        ]
        n = tracker.seed_from_trades(records)  # type: ignore[arg-type]
        assert n == 1
