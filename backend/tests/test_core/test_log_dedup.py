"""M13 (2026-06-07): tests for core.log_dedup TTL warning suppression."""

import logging
import time

import pytest

from core import log_dedup


@pytest.fixture(autouse=True)
def _reset_cache():
    log_dedup.reset()
    yield
    log_dedup.reset()


def test_first_call_emits(caplog):
    caplog.set_level(logging.WARNING)
    log = logging.getLogger("test")
    emitted = log_dedup.warn_once_per(log, "k", 60.0, "first %s", "msg")
    assert emitted is True
    assert any("first msg" in r.message for r in caplog.records)


def test_repeat_within_ttl_suppressed(caplog):
    caplog.set_level(logging.WARNING)
    log = logging.getLogger("test")
    log_dedup.warn_once_per(log, "k", 60.0, "first")
    caplog.clear()
    emitted = log_dedup.warn_once_per(log, "k", 60.0, "second")
    assert emitted is False
    assert not any("second" in r.message for r in caplog.records)


def test_different_keys_independent(caplog):
    caplog.set_level(logging.WARNING)
    log = logging.getLogger("test")
    log_dedup.warn_once_per(log, "a", 60.0, "alpha")
    log_dedup.warn_once_per(log, "b", 60.0, "beta")
    msgs = [r.message for r in caplog.records]
    assert any("alpha" in m for m in msgs)
    assert any("beta" in m for m in msgs)


def test_expired_ttl_re_emits(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    log = logging.getLogger("test")
    log_dedup.warn_once_per(log, "k", 0.01, "first")
    time.sleep(0.02)
    caplog.clear()
    emitted = log_dedup.warn_once_per(log, "k", 0.01, "second")
    assert emitted is True
    assert any("second" in r.message for r in caplog.records)


def test_cache_size_grows():
    log = logging.getLogger("test")
    assert log_dedup.cache_size() == 0
    log_dedup.warn_once_per(log, "x", 60.0, "x")
    log_dedup.warn_once_per(log, "y", 60.0, "y")
    assert log_dedup.cache_size() == 2
