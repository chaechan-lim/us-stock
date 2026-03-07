"""Tests for BacktestResultStore."""

import json
import pytest
from pathlib import Path

from backtest.result_store import BacktestResultStore, _make_safe


@pytest.fixture
def store(tmp_path):
    return BacktestResultStore(store_dir=tmp_path / "results")


class TestMakeSafe:
    def test_inf_becomes_none(self):
        assert _make_safe(float("inf")) is None

    def test_nan_becomes_none(self):
        assert _make_safe(float("nan")) is None

    def test_normal_float(self):
        assert _make_safe(3.14) == 3.14

    def test_nested_dict(self):
        data = {"a": float("inf"), "b": {"c": float("nan"), "d": 1.0}}
        result = _make_safe(data)
        assert result == {"a": None, "b": {"c": None, "d": 1.0}}

    def test_list(self):
        data = [1.0, float("inf"), float("nan")]
        result = _make_safe(data)
        assert result == [1.0, None, None]


class TestBacktestResultStore:
    def test_save_and_get(self, store):
        data = {"metrics": {"cagr": 0.15}, "trades": []}
        key = store.save("trend_following", "AAPL", "3y", data)
        assert key
        assert store.exists("trend_following", "AAPL", "3y")

        retrieved = store.get("trend_following", "AAPL", "3y")
        assert retrieved is not None
        assert retrieved["result"]["metrics"]["cagr"] == 0.15

    def test_dedup_prevents_rerun(self, store):
        data = {"metrics": {"cagr": 0.15}}
        store.save("trend_following", "AAPL", "3y", data)
        assert store.exists("trend_following", "AAPL", "3y")

    def test_different_params_different_key(self, store):
        data = {"metrics": {"cagr": 0.15}}
        k1 = store.save("trend_following", "AAPL", "3y", data, params={"a": 1})
        k2 = store.save("trend_following", "AAPL", "3y", data, params={"a": 2})
        assert k1 != k2
        assert store.count == 2

    def test_different_symbol_different_key(self, store):
        data = {"metrics": {"cagr": 0.1}}
        store.save("trend_following", "AAPL", "3y", data)
        store.save("trend_following", "MSFT", "3y", data)
        assert store.count == 2

    def test_list_results(self, store):
        store.save("s1", "AAPL", "3y", {"v": 1})
        store.save("s2", "AAPL", "3y", {"v": 2})
        store.save("s1", "MSFT", "3y", {"v": 3})

        all_results = store.list_results()
        assert len(all_results) == 3

        s1_results = store.list_results(strategy_name="s1")
        assert len(s1_results) == 2

        aapl_results = store.list_results(symbol="AAPL")
        assert len(aapl_results) == 2

    def test_delete(self, store):
        store.save("s1", "AAPL", "3y", {"v": 1})
        assert store.count == 1
        results = store.list_results()
        assert store.delete(results[0]["key"])
        assert store.count == 0

    def test_clear_all(self, store):
        store.save("s1", "AAPL", "3y", {"v": 1})
        store.save("s2", "MSFT", "3y", {"v": 2})
        count = store.clear_all()
        assert count == 2
        assert store.count == 0

    def test_get_by_key(self, store):
        key = store.save("s1", "AAPL", "3y", {"v": 1})
        result = store.get_by_key(key)
        assert result is not None
        assert result["strategy"] == "s1"

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent", "AAPL", "3y") is None
        assert store.get_by_key("nonexistent") is None

    def test_mode_separation(self, store):
        store.save("s1", "AAPL", "3y", {"v": 1}, mode="single")
        store.save("s1", "AAPL", "3y", {"v": 2}, mode="adaptive_comparison")
        assert store.count == 2

        single = store.list_results(mode="single")
        assert len(single) == 1
        adaptive = store.list_results(mode="adaptive_comparison")
        assert len(adaptive) == 1

    def test_persistence_across_instances(self, tmp_path):
        store_dir = tmp_path / "persist"
        store1 = BacktestResultStore(store_dir=store_dir)
        store1.save("s1", "AAPL", "3y", {"v": 1})
        assert store1.count == 1

        store2 = BacktestResultStore(store_dir=store_dir)
        assert store2.count == 1
        assert store2.exists("s1", "AAPL", "3y")

    def test_inf_in_result_data(self, store):
        data = {"profit_factor": float("inf"), "nested": {"val": float("nan")}}
        key = store.save("s1", "AAPL", "3y", data)
        retrieved = store.get_by_key(key)
        assert retrieved["result"]["profit_factor"] is None
        assert retrieved["result"]["nested"]["val"] is None

    def test_make_key_deterministic(self):
        k1 = BacktestResultStore.make_key("s1", "AAPL", "3y", {"a": 1})
        k2 = BacktestResultStore.make_key("s1", "AAPL", "3y", {"a": 1})
        assert k1 == k2
