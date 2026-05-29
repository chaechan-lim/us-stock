"""Hermes Phase 0+1 — Exposure tracker + sentinel.

Captures a daily snapshot of deployment health (per market):
  - deployed_pct = stock_value / equity
  - slot_fill_ratio = avg(min(1, pos_value / target_slot))
  - placeholder_count = positions below 30% of target slot
  - funnel breakdown = top 3 rejection reasons + share of total
  - cash_idle_days = consecutive days deployed_pct < 50%

Then evaluates sentinel flags (chronic under-deployment, funnel
concentration) and returns them for Discord push / AgentRecommendation
queueing by the daily-post-market chain.

This is the observation layer for Hermes self-evolution. It does not
mutate yaml or place orders. Sentinel flags surface investigation
triggers to the LLM rec chain (Phase 2) which proposes parameter or
gate changes — those still go through backtest + counterfactual gates
before live application.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("exposure_tracker")

KST = timezone(timedelta(hours=9))

# Tuneable thresholds. Kept module-level for easy override in tests.
PLACEHOLDER_RATIO = 0.30           # pos < 30% of target slot = placeholder
CHRONIC_DEPLOYED_PCT = 0.50        # deployed < 50% → counts toward idle streak
CHRONIC_DAYS = 5                   # 5+ consecutive idle days → sentinel
FUNNEL_CONCENTRATION_PCT = 0.40    # any single reject reason > 40% → sentinel
MIN_FUNNEL_SIGNALS = 5             # ignore funnel concentration if too few signals


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FunnelBreakdown:
    market: str
    total_signals: int
    buys_placed: int
    fill_rate: float | None
    top_reasons: list[tuple[str, int, float]] = field(default_factory=list)
    top_reason_pct: float = 0.0


@dataclass
class MarketExposure:
    market: str
    equity: float
    cash: float
    stock_value: float
    deployed_pct: float
    position_count: int
    placeholder_count: int
    target_slot: float
    slot_fill_ratio: float
    cash_idle_days: int
    funnel: FunnelBreakdown | None = None


@dataclass
class SentinelFlag:
    market: str
    flag: str
    severity: str  # "warning" | "critical"
    detail: str


@dataclass
class ExposureSnapshot:
    date: str
    generated_at: str
    markets: dict[str, MarketExposure]
    flags: list[SentinelFlag]


# ---------------------------------------------------------------------------
# Pure compute (unit-testable)
# ---------------------------------------------------------------------------


def compute_funnel_breakdown(market: str, raw: dict[str, Any] | None) -> FunnelBreakdown | None:
    """Convert /api/v1/engine/rejection-funnel payload for one market.

    The endpoint reports per-market subdicts with `buy_signals_total`,
    `buys_placed`, `rejected_total`, `fill_rate`, and a `rejections`
    map of reason → count.
    """
    if not raw:
        return None
    total = int(raw.get("buy_signals_total", 0) or 0)
    placed = int(raw.get("buys_placed", 0) or 0)
    rejections = raw.get("rejections") or {}
    items = sorted(
        ((str(k), int(v)) for k, v in rejections.items() if int(v) > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top = []
    top_pct = 0.0
    if total > 0:
        for reason, cnt in items[:3]:
            pct = cnt / total
            top.append((reason, cnt, round(pct, 3)))
        if items:
            top_pct = round(items[0][1] / total, 3)
    fill_rate = raw.get("fill_rate")
    return FunnelBreakdown(
        market=market,
        total_signals=total,
        buys_placed=placed,
        fill_rate=float(fill_rate) if fill_rate is not None else None,
        top_reasons=top,
        top_reason_pct=top_pct,
    )


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def compute_market_exposure(
    market: str,
    equity: float,
    cash: float,
    positions: list[dict[str, Any]],
    funnel: FunnelBreakdown | None,
    min_position_pct: float,
    prior_idle_streak: int = 0,
    placeholder_ratio: float = PLACEHOLDER_RATIO,
) -> MarketExposure:
    """Pure computation — no I/O. Easy to unit-test."""
    market_positions = [p for p in positions if (p.get("market") or market) == market]
    pos_values = []
    for p in market_positions:
        qty = _safe_float(p.get("quantity"))
        price = _safe_float(p.get("current_price")) or _safe_float(p.get("avg_price"))
        if qty > 0 and price > 0:
            pos_values.append(qty * price)
    stock_value = sum(pos_values)
    deployed_pct = (stock_value / equity) if equity > 0 else 0.0

    target_slot = max(0.0, equity * min_position_pct)
    if target_slot > 0 and pos_values:
        slot_fill = sum(min(1.0, v / target_slot) for v in pos_values) / len(pos_values)
        placeholders = sum(1 for v in pos_values if v < target_slot * placeholder_ratio)
    else:
        slot_fill = 0.0
        placeholders = 0

    # Idle streak: yesterday's streak + 1 if today still under threshold, else 0
    if deployed_pct < CHRONIC_DEPLOYED_PCT:
        idle_streak = prior_idle_streak + 1
    else:
        idle_streak = 0

    return MarketExposure(
        market=market,
        equity=round(equity, 2),
        cash=round(cash, 2),
        stock_value=round(stock_value, 2),
        deployed_pct=round(deployed_pct, 4),
        position_count=len(market_positions),
        placeholder_count=placeholders,
        target_slot=round(target_slot, 2),
        slot_fill_ratio=round(slot_fill, 4),
        cash_idle_days=idle_streak,
        funnel=funnel,
    )


def evaluate_sentinel(
    market_exp: MarketExposure,
    chronic_days: int = CHRONIC_DAYS,
    funnel_concentration_pct: float = FUNNEL_CONCENTRATION_PCT,
    min_funnel_signals: int = MIN_FUNNEL_SIGNALS,
) -> list[SentinelFlag]:
    """Evaluate sentinel rules against one market's exposure snapshot."""
    flags: list[SentinelFlag] = []

    if (
        market_exp.cash_idle_days >= chronic_days
        and market_exp.deployed_pct < CHRONIC_DEPLOYED_PCT
    ):
        flags.append(SentinelFlag(
            market=market_exp.market,
            flag="chronic_under_deployment",
            severity="warning",
            detail=(
                f"deployed {market_exp.deployed_pct*100:.1f}% for "
                f"{market_exp.cash_idle_days}d (target ≥{CHRONIC_DEPLOYED_PCT*100:.0f}%)"
            ),
        ))

    funnel = market_exp.funnel
    if (
        funnel is not None
        and funnel.total_signals >= min_funnel_signals
        and funnel.top_reason_pct >= funnel_concentration_pct
        and funnel.top_reasons
    ):
        reason, cnt, pct = funnel.top_reasons[0]
        flags.append(SentinelFlag(
            market=market_exp.market,
            flag="funnel_concentration",
            severity="warning",
            detail=(
                f"{reason} = {cnt}/{funnel.total_signals} "
                f"({pct*100:.1f}%) of rejects"
            ),
        ))

    return flags


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _serialize_snapshot(snap: ExposureSnapshot) -> dict[str, Any]:
    payload = {
        "version": 1,
        "date": snap.date,
        "generated_at": snap.generated_at,
        "markets": {},
        "flags": [asdict(f) for f in snap.flags],
    }
    for market, exp in snap.markets.items():
        d = asdict(exp)
        if exp.funnel is not None:
            d["funnel"] = asdict(exp.funnel)
        payload["markets"][market] = d
    return payload


def load_prior_idle_streak(history_dir: Path, market: str, today: date) -> int:
    """Read yesterday's cash_idle_days for this market. Returns 0 if missing."""
    if not history_dir.exists():
        return 0
    yesterday = today - timedelta(days=1)
    candidate = history_dir / f"{yesterday}.json"
    if not candidate.exists():
        # Walk back up to 7d in case the timer was offline
        for back in range(2, 8):
            d = today - timedelta(days=back)
            candidate = history_dir / f"{d}.json"
            if candidate.exists():
                break
        else:
            return 0
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return int(data.get("markets", {}).get(market, {}).get("cash_idle_days", 0))
    except (OSError, ValueError, KeyError) as e:
        logger.warning("Failed to read prior streak from %s: %s", candidate, e)
        return 0


def write_snapshot(snap: ExposureSnapshot, history_dir: Path) -> Path:
    """Atomic write to history_dir/{date}.json."""
    history_dir.mkdir(parents=True, exist_ok=True)
    out = history_dir / f"{snap.date}.json"
    tmp = out.with_suffix(".json.tmp")
    payload = _serialize_snapshot(snap)
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# Live snapshot collection (I/O)
# ---------------------------------------------------------------------------


async def _http_get(session: Any, url: str, timeout: int = 5) -> Any:
    import aiohttp
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except Exception as e:
        logger.warning("API fetch %s failed: %s", url, e)
        return None


async def collect_snapshot(
    api_base: str = "http://localhost:8001",
    history_dir: Path | None = None,
    min_position_pcts: dict[str, float] | None = None,
    today: date | None = None,
) -> ExposureSnapshot:
    """End-to-end: hit the live API, compute exposure per market, evaluate
    sentinel, return snapshot. Caller writes to disk if desired."""
    import aiohttp

    if history_dir is None:
        history_dir = Path(__file__).resolve().parents[2] / "data" / "exposure_history"
    if min_position_pcts is None:
        min_position_pcts = _load_min_position_pcts()
    if today is None:
        today = datetime.now(KST).date()

    async with aiohttp.ClientSession() as session:
        funnel = await _http_get(session, f"{api_base}/api/v1/engine/rejection-funnel")
        markets_payload: dict[str, MarketExposure] = {}
        all_flags: list[SentinelFlag] = []

        for market in ("US", "KR"):
            summary = await _http_get(
                session, f"{api_base}/api/v1/portfolio/summary?market={market}",
            )
            positions = await _http_get(
                session, f"{api_base}/api/v1/portfolio/positions?market={market}",
            )
            if not summary:
                logger.warning("Missing summary for %s — skipping", market)
                continue
            equity = _extract_equity(market, summary)
            cash = _extract_cash(market, summary)
            funnel_market = compute_funnel_breakdown(market, (funnel or {}).get(market))
            prior_streak = load_prior_idle_streak(history_dir, market, today)
            exp = compute_market_exposure(
                market=market,
                equity=equity,
                cash=cash,
                positions=positions or [],
                funnel=funnel_market,
                min_position_pct=min_position_pcts.get(market, 0.05),
                prior_idle_streak=prior_streak,
            )
            markets_payload[market] = exp
            all_flags.extend(evaluate_sentinel(exp))

    return ExposureSnapshot(
        date=today.isoformat(),
        generated_at=datetime.now(KST).isoformat(),
        markets=markets_payload,
        flags=all_flags,
    )


def _extract_equity(market: str, summary: dict[str, Any]) -> float:
    """portfolio/summary returns different shapes for ?market=ALL vs
    ?market=KR/US. With market=KR or US the response has the simpler
    balance.total / total_equity in market-native currency, which is
    what we want. With market=ALL the breakdown nests it in
    equity_breakdown / usd_balance."""
    # Simple shape (per-market query): balance.total is the equity.
    bal = summary.get("balance") or {}
    if bal.get("total") is not None:
        return _safe_float(bal.get("total"))
    bd = summary.get("equity_breakdown") or {}
    if market == "KR":
        return _safe_float(bd.get("kr_total_krw") or summary.get("total_equity"))
    if market == "US":
        usd_total = (summary.get("usd_balance") or {}).get("total")
        if usd_total is not None:
            return _safe_float(usd_total)
        fx = _safe_float(summary.get("exchange_rate"), 1.0) or 1.0
        return _safe_float(summary.get("total_equity")) / fx
    return _safe_float(summary.get("total_equity"))


def _extract_cash(market: str, summary: dict[str, Any]) -> float:
    """Cash = orderable balance in market-native currency. Per-market
    response exposes it as balance.available; ALL response has the
    cash_breakdown sub-dict."""
    bal = summary.get("balance") or {}
    if bal.get("available") is not None:
        return _safe_float(bal.get("available"))
    cb = summary.get("cash_breakdown") or {}
    if market == "KR":
        return _safe_float(cb.get("kr_orderable_cash_krw"))
    if market == "US":
        return _safe_float(cb.get("us_orderable_cash_usd"))
    return _safe_float(summary.get("available_cash"))


def _load_min_position_pcts() -> dict[str, float]:
    """Read min_position_pct per market from strategies.yaml."""
    try:
        from strategies.config_loader import StrategyConfigLoader
        loader = StrategyConfigLoader()
        out = {}
        for m in ("US", "KR"):
            risk = loader.get_market_risk_config(m) or {}
            out[m] = float(risk.get("min_position_pct", 0.05))
        return out
    except Exception as e:
        logger.warning("Failed to load min_position_pct from yaml: %s — using 0.05", e)
        return {"US": 0.05, "KR": 0.05}
