"""Persistent backtest result storage with deduplication.

Stores results as JSON files indexed by a hash of
(strategy, symbol, period, params). Prevents re-running identical backtests.

Results are tagged with a `stale` flag if they were produced before
``RELIABILITY_CUTOFF`` — the commit ff6279f (2026-04-07 09:37 KST) that fixed
look-ahead bias, Kelly param mismatch, asymmetric slippage, and the KR
currency bug. Stale results are still readable for historical comparison
but should not be used as evidence for live decisions.
"""

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path(__file__).parent.parent.parent / "data" / "backtest_results"

# Backtest reliability cutoff: any result file with a timestamp earlier than
# this was produced before fix commit ff6279f and is no longer trustworthy
# for live decisions. The cutoff is the commit's author time in KST.
RELIABILITY_CUTOFF_ISO = "2026-04-07T09:37:20"
RELIABILITY_CUTOFF = datetime.fromisoformat(RELIABILITY_CUTOFF_ISO)


def _is_stale_timestamp(ts: str | None) -> bool:
    """Return True if a backtest timestamp predates the reliability cutoff.

    Naive (timezone-less) ISO timestamps are interpreted as the local KST
    timezone in which the result store has historically run. Malformed or
    missing timestamps are treated as stale (safer default).
    """
    if not ts:
        return True
    try:
        # Strip timezone if present so naive comparison works
        if "+" in ts or ts.endswith("Z"):
            ts_clean = ts.split("+")[0].rstrip("Z")
        else:
            ts_clean = ts
        return datetime.fromisoformat(ts_clean) < RELIABILITY_CUTOFF
    except (ValueError, TypeError):
        return True


def _make_safe(obj):
    """Recursively replace inf/nan with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _make_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_safe(v) for v in obj]
    return obj


class BacktestResultStore:
    """Persistent store for backtest results with dedup."""

    def __init__(self, store_dir: str | Path | None = None):
        self._store_dir = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._store_dir / "index.json"
        self._index = self._load_index()
        self._migrate_stale_flags()

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupted index, rebuilding")
        return {}

    def _migrate_stale_flags(self) -> None:
        """One-time migration: tag any pre-cutoff entries as stale.

        Idempotent — re-running on an already-migrated index is a no-op.
        Persists the index back to disk if any change is made.
        """
        changed = False
        n_stale = 0
        for key, meta in self._index.items():
            if "stale" in meta:
                if meta["stale"]:
                    n_stale += 1
                continue
            if _is_stale_timestamp(meta.get("timestamp")):
                meta["stale"] = True
                meta["stale_reason"] = (
                    f"timestamp predates reliability cutoff "
                    f"{RELIABILITY_CUTOFF_ISO} (commit ff6279f)"
                )
                n_stale += 1
                changed = True
            else:
                meta["stale"] = False
        if changed:
            self._save_index()
            logger.info(
                "Tagged %d backtest result(s) as stale (pre-%s)",
                n_stale, RELIABILITY_CUTOFF_ISO,
            )

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, indent=2, default=str)
        )

    @staticmethod
    def make_key(
        strategy_name: str,
        symbol: str,
        period: str,
        params: dict | None = None,
        mode: str = "single",
    ) -> str:
        """Create a deterministic hash key for a backtest run."""
        key_data = {
            "strategy": strategy_name,
            "symbol": symbol,
            "period": period,
            "params": params or {},
            "mode": mode,
        }
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def exists(
        self,
        strategy_name: str,
        symbol: str,
        period: str,
        params: dict | None = None,
        mode: str = "single",
    ) -> bool:
        """Check if a backtest result already exists."""
        key = self.make_key(strategy_name, symbol, period, params, mode)
        return key in self._index

    def get(
        self,
        strategy_name: str,
        symbol: str,
        period: str,
        params: dict | None = None,
        mode: str = "single",
    ) -> dict | None:
        """Get a stored backtest result."""
        key = self.make_key(strategy_name, symbol, period, params, mode)
        if key not in self._index:
            return None
        result_path = self._store_dir / f"{key}.json"
        if not result_path.exists():
            del self._index[key]
            self._save_index()
            return None
        try:
            return json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def save(
        self,
        strategy_name: str,
        symbol: str,
        period: str,
        result_data: dict,
        params: dict | None = None,
        mode: str = "single",
    ) -> str:
        """Save a backtest result. Returns the key."""
        key = self.make_key(strategy_name, symbol, period, params, mode)

        record = {
            "key": key,
            "strategy": strategy_name,
            "symbol": symbol,
            "period": period,
            "mode": mode,
            "params": params or {},
            "timestamp": datetime.now().isoformat(),
            "result": _make_safe(result_data),
        }

        result_path = self._store_dir / f"{key}.json"
        result_path.write_text(json.dumps(record, indent=2, default=str))

        self._index[key] = {
            "strategy": strategy_name,
            "symbol": symbol,
            "period": period,
            "mode": mode,
            "timestamp": record["timestamp"],
            "stale": False,  # New saves are always post-cutoff by definition
        }
        self._save_index()

        logger.info("Saved backtest result: %s/%s [%s]", strategy_name, symbol, key)
        return key

    def list_results(
        self,
        strategy_name: str | None = None,
        symbol: str | None = None,
        mode: str | None = None,
        include_stale: bool = True,
    ) -> list[dict]:
        """List stored results with optional filters.

        Args:
            include_stale: When False, results produced before the
                reliability cutoff (commit ff6279f) are filtered out.
                Default True for backwards compatibility — callers that
                make live decisions should pass False.
        """
        results = []
        for key, meta in self._index.items():
            if strategy_name and meta["strategy"] != strategy_name:
                continue
            if symbol and meta["symbol"] != symbol:
                continue
            if mode and meta.get("mode") != mode:
                continue
            if not include_stale and meta.get("stale", False):
                continue
            results.append({"key": key, **meta})
        return sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_by_key(self, key: str) -> dict | None:
        """Get result by its hash key."""
        if key not in self._index:
            return None
        result_path = self._store_dir / f"{key}.json"
        if not result_path.exists():
            return None
        try:
            return json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def delete(self, key: str) -> bool:
        """Delete a stored result."""
        if key not in self._index:
            return False
        result_path = self._store_dir / f"{key}.json"
        if result_path.exists():
            result_path.unlink()
        del self._index[key]
        self._save_index()
        return True

    def clear_all(self) -> int:
        """Delete all stored results. Returns count deleted."""
        count = len(self._index)
        for key in list(self._index.keys()):
            result_path = self._store_dir / f"{key}.json"
            if result_path.exists():
                result_path.unlink()
        self._index.clear()
        self._save_index()
        return count

    @property
    def count(self) -> int:
        return len(self._index)
