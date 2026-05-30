"""Apply structured agent recommendations to strategies.yaml at runtime.

Track A4: when the operator clicks Accept on a recommendation, this
mutator updates the yaml file in place (atomic write + .bak backup) and
the caller triggers the existing /strategies/reload path to pick up the
change without a restart.

Safety guards:
1. Whitelist — only paths that are actually consumed by hot-reload
   (markets.{KR,US}.evaluation_loop.* + risk.* + disabled_strategies +
   tiered_trailing_stop.tiers). main.py-bound params (e.g. regime_position
   _pct in RiskParams) are excluded; they need a backend restart.
2. Type validation — proposed_value type must match the existing value's
   type (or both are list/dict).
3. Atomic write — write to a temp file then rename, so a crash mid-write
   doesn't leave a half-written yaml.
4. Backup — previous yaml saved as strategies.yaml.bak before each apply.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Hot-reload-able paths. Anything not in this list raises and forces the
# operator to apply manually (with a restart if needed).
ALLOWED_PARAM_PREFIXES: tuple[str, ...] = (
    "markets.KR.disabled_strategies",
    "markets.KR.evaluation_loop.",
    "markets.KR.risk.max_positions",
    "markets.KR.risk.max_position_pct",
    "markets.KR.risk.min_position_pct",
    "markets.KR.risk.default_stop_loss_pct",
    "markets.KR.risk.default_take_profit_pct",
    "markets.KR.risk.default_trailing_activation_pct",
    "markets.KR.risk.default_trailing_stop_pct",
    "markets.KR.cash_parking.",
    "markets.US.disabled_strategies",
    "markets.US.evaluation_loop.",
    "markets.US.cash_parking.",
    "tiered_trailing_stop.tiers",
    "tiered_trailing_stop.enabled",
    "breakeven_stop.enabled",
    "breakeven_stop.activation_ratio",
    "breakeven_stop.lock_ratio",
    "breakeven_stop.lock_pct",
)


class YamlMutationError(Exception):
    """Raised when an apply fails — caller surfaces the message to the user."""


def _is_path_allowed(param_path: str) -> bool:
    return any(
        param_path == p.rstrip(".") or param_path.startswith(p)
        for p in ALLOWED_PARAM_PREFIXES
    )


def path_exists(yaml_path: str | os.PathLike, param_path: str) -> bool:
    """Return True iff every segment of `param_path` resolves to a key
    in the yaml file. Used to drop LLM-hallucinated param paths before
    they are inserted as recommendations.

    Unlike apply_yaml_change which RAISES on missing intermediates,
    this returns a clean boolean — safe for filter loops.
    """
    try:
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return False
    parts = param_path.split(".")
    cur: Any = data
    for key in parts:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _walk(d: dict, parts: list[str]) -> tuple[dict, str]:
    """Walk nested dict to the parent of the leaf, return (parent, leaf_key).
    Raises if any intermediate key is missing or not a dict."""
    cur: Any = d
    for i, key in enumerate(parts[:-1]):
        if not isinstance(cur, dict) or key not in cur:
            trail = ".".join(parts[: i + 1])
            raise YamlMutationError(f"yaml path not found: {trail}")
        cur = cur[key]
    if not isinstance(cur, dict):
        raise YamlMutationError(f"yaml parent of {parts[-1]} is not a mapping")
    return cur, parts[-1]


def _types_match(old: Any, new: Any) -> bool:
    """Allow same primitive type, both list, or both dict. None matches
    anything (operator may be initializing a new value)."""
    if old is None or new is None:
        return True
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool)
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return True
    if isinstance(old, str) and isinstance(new, str):
        return True
    if isinstance(old, list) and isinstance(new, list):
        return True
    if isinstance(old, dict) and isinstance(new, dict):
        return True
    return False


def apply_yaml_change(
    yaml_path: str | os.PathLike,
    param_path: str,
    new_value: Any,
) -> tuple[Any, Any]:
    """Apply param_path = new_value to the yaml file.

    Returns (old_value, new_value) tuple on success. Raises
    YamlMutationError on any safety failure.
    """
    if not _is_path_allowed(param_path):
        raise YamlMutationError(
            f"param_path {param_path!r} is not in the hot-reload whitelist; "
            "apply via PR + restart instead",
        )

    p = Path(yaml_path)
    if not p.exists():
        raise YamlMutationError(f"yaml file not found: {p}")

    raw = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise YamlMutationError(f"yaml parse failed: {e}") from e
    if not isinstance(data, dict):
        raise YamlMutationError("yaml root is not a mapping")

    parts = param_path.split(".")
    parent, leaf = _walk(data, parts)
    if leaf not in parent:
        raise YamlMutationError(f"yaml leaf not found: {param_path}")
    old_value = parent[leaf]

    if not _types_match(old_value, new_value):
        raise YamlMutationError(
            f"type mismatch for {param_path}: existing {type(old_value).__name__}, "
            f"proposed {type(new_value).__name__}",
        )

    parent[leaf] = new_value

    # Backup + atomic write
    backup = p.with_suffix(p.suffix + ".bak")
    shutil.copy2(p, backup)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    os.replace(tmp, p)
    logger.info(
        "yaml mutated: %s : %r → %r (backup at %s)",
        param_path, old_value, new_value, backup.name,
    )
    return old_value, new_value
