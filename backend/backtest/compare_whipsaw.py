"""A/B comparison: whipsaw prevention parameters (STOCK-47).

Replays actual trade history from the orders table and simulates how
the new parameters would have affected trading behavior.

Compares:
- BEFORE: cleanup=-3%, held_min_conf=0.25, held_sell_bias=0.10, no min hold, no whipsaw counter
- AFTER:  cleanup=-5%, held_min_conf=0.40, held_sell_bias=0.05, min hold 4h, whipsaw counter (2 loss sells → block)

Usage:
    cd backend
    python3 -m backtest.compare_whipsaw
"""

import asyncio
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import asyncpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DB_URL", "postgresql://localhost:5432/us_stock_trading"
)


async def load_trades(days: int = 30) -> list[dict]:
    """Load filled orders from the database."""
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT id, symbol, side, strategy_name, filled_price, filled_quantity,
                   pnl, pnl_pct, market, created_at
            FROM orders
            WHERE status = 'filled' AND created_at > NOW() - $1::interval
            ORDER BY created_at
            """,
            timedelta(days=days),
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


def analyze_whipsaw(trades: list[dict], params: dict) -> dict:
    """Simulate whipsaw filtering with given parameters.

    Returns metrics about how many trades would be blocked.
    """
    cleanup_threshold = params["cleanup_threshold"]  # e.g. -0.03 or -0.05
    min_hold_secs = params["min_hold_secs"]  # e.g. 0 or 14400
    max_loss_sells = params["max_loss_sells"]  # e.g. 999 or 2
    held_min_conf = params["held_min_conf"]  # e.g. 0.25 or 0.40
    held_sell_bias = params["held_sell_bias"]  # e.g. 0.10 or 0.05

    # Track state per symbol
    positions: dict[str, dict] = {}  # symbol -> {buy_time, buy_price, qty}
    loss_sell_history: dict[str, list[datetime]] = defaultdict(list)
    sell_cooldown: dict[str, datetime] = {}  # symbol -> last sell time
    cooldown_secs = 24 * 3600

    executed_buys = 0
    executed_sells = 0
    blocked_buys_cooldown = 0
    blocked_buys_whipsaw = 0
    blocked_sells_min_hold = 0
    blocked_sells_threshold = 0
    total_pnl = 0.0
    loss_sell_count = 0
    whipsaw_cycles = 0  # buy->loss_sell->buy within 7d on same symbol

    # Track symbols with multiple loss sells for whipsaw detection
    recent_loss_sells: dict[str, list[datetime]] = defaultdict(list)

    for trade in trades:
        symbol = trade["symbol"]
        side = trade["side"]
        price = trade["filled_price"] or 0
        qty = trade["filled_quantity"] or 0
        pnl = trade["pnl"] or 0
        strategy = trade["strategy_name"] or ""
        ts = trade["created_at"]

        if side == "BUY":
            # Check cooldown
            if symbol in sell_cooldown:
                elapsed = (ts - sell_cooldown[symbol]).total_seconds()
                if elapsed < cooldown_secs:
                    blocked_buys_cooldown += 1
                    continue

            # Check whipsaw counter
            cutoff = ts - timedelta(days=7)
            recent = [t for t in recent_loss_sells.get(symbol, []) if t > cutoff]
            if len(recent) >= max_loss_sells:
                blocked_buys_whipsaw += 1
                continue

            # Execute buy
            positions[symbol] = {"buy_time": ts, "buy_price": price, "qty": qty}
            executed_buys += 1

        elif side == "SELL":
            pos = positions.get(symbol)

            # Check if this is a position_cleanup sell
            if strategy == "position_cleanup" and pos:
                pnl_pct = (price - pos["buy_price"]) / pos["buy_price"] if pos["buy_price"] > 0 else 0
                # Check threshold
                if pnl_pct >= cleanup_threshold:
                    blocked_sells_threshold += 1
                    continue
                # Check min hold (except hard SL at -7%)
                if pnl_pct >= -0.07 and min_hold_secs > 0:
                    hold_secs = (ts - pos["buy_time"]).total_seconds()
                    if hold_secs < min_hold_secs:
                        blocked_sells_min_hold += 1
                        continue

            # Check min hold for other strategy sells
            if pos and strategy not in ("position_cleanup", "profit_protection"):
                if min_hold_secs > 0:
                    hold_secs = (ts - pos["buy_time"]).total_seconds()
                    pnl_pct = (price - pos["buy_price"]) / pos["buy_price"] if pos["buy_price"] > 0 else 0
                    # Allow hard SL
                    if pnl_pct >= -0.07 and hold_secs < min_hold_secs:
                        blocked_sells_min_hold += 1
                        continue

            # Execute sell
            executed_sells += 1
            total_pnl += pnl

            # Track loss sells
            is_loss = pos and price < pos.get("buy_price", price)
            if is_loss:
                loss_sell_count += 1
                recent_loss_sells[symbol].append(ts)

                # Check if this is a whipsaw (loss sell after recent loss sell)
                cutoff = ts - timedelta(days=7)
                recent = [t for t in recent_loss_sells[symbol] if t > cutoff]
                if len(recent) >= 2:
                    whipsaw_cycles += 1

            sell_cooldown[symbol] = ts
            positions.pop(symbol, None)

    return {
        "executed_buys": executed_buys,
        "executed_sells": executed_sells,
        "blocked_buys_cooldown": blocked_buys_cooldown,
        "blocked_buys_whipsaw": blocked_buys_whipsaw,
        "blocked_sells_min_hold": blocked_sells_min_hold,
        "blocked_sells_threshold": blocked_sells_threshold,
        "total_pnl": total_pnl,
        "loss_sell_count": loss_sell_count,
        "whipsaw_cycles": whipsaw_cycles,
    }


async def main():
    logger.info("Loading trade history (30 days)...")
    trades = await load_trades(days=30)
    logger.info("Loaded %d filled trades", len(trades))

    if not trades:
        print("No trades found.")
        return

    # Separate by market
    kr_trades = [t for t in trades if t["market"] == "KR"]
    us_trades = [t for t in trades if t["market"] == "US"]

    before_params = {
        "cleanup_threshold": -0.03,
        "min_hold_secs": 0,
        "max_loss_sells": 999,  # effectively disabled
        "held_min_conf": 0.25,
        "held_sell_bias": 0.10,
    }

    after_params = {
        "cleanup_threshold": -0.05,
        "min_hold_secs": 4 * 3600,
        "max_loss_sells": 2,
        "held_min_conf": 0.40,
        "held_sell_bias": 0.05,
    }

    print("\n" + "=" * 75)
    print("STOCK-47 Whipsaw Prevention — A/B Comparison (30-day trade replay)")
    print("=" * 75)

    for market_name, market_trades in [("KR", kr_trades), ("US", us_trades), ("ALL", trades)]:
        if not market_trades:
            continue

        before = analyze_whipsaw(market_trades, before_params)
        after = analyze_whipsaw(market_trades, after_params)

        print(f"\n── {market_name} Market ({len(market_trades)} trades) ──")
        print(f"{'Metric':<30s} {'BEFORE':>10s} {'AFTER':>10s} {'Diff':>10s}")
        print("-" * 65)

        metrics = [
            ("Executed buys", "executed_buys", "d"),
            ("Executed sells", "executed_sells", "d"),
            ("Blocked buys (cooldown)", "blocked_buys_cooldown", "d"),
            ("Blocked buys (whipsaw)", "blocked_buys_whipsaw", "d"),
            ("Blocked sells (min hold)", "blocked_sells_min_hold", "d"),
            ("Blocked sells (threshold)", "blocked_sells_threshold", "d"),
            ("Total realized PnL", "total_pnl", ",.0f"),
            ("Loss sell count", "loss_sell_count", "d"),
            ("Whipsaw cycles", "whipsaw_cycles", "d"),
        ]

        for label, key, fmt in metrics:
            v1 = before[key]
            v2 = after[key]
            diff = v2 - v1
            if fmt == "d":
                print(f"  {label:<28s} {v1:>10d} {v2:>10d} {diff:>+10d}")
            else:
                print(f"  {label:<28s} {v1:>10{fmt}} {v2:>10{fmt}} {diff:>+10{fmt}}")

    # Summary
    before_all = analyze_whipsaw(trades, before_params)
    after_all = analyze_whipsaw(trades, after_params)

    print("\n" + "=" * 75)
    print("VERDICT")
    print("=" * 75)

    pnl_diff = after_all["total_pnl"] - before_all["total_pnl"]
    ws_diff = after_all["whipsaw_cycles"] - before_all["whipsaw_cycles"]
    loss_diff = after_all["loss_sell_count"] - before_all["loss_sell_count"]
    blocked_total = (
        after_all["blocked_buys_whipsaw"]
        + after_all["blocked_sells_min_hold"]
        + after_all["blocked_sells_threshold"]
    )

    print(f"  Whipsaw cycles: {before_all['whipsaw_cycles']} → {after_all['whipsaw_cycles']} ({ws_diff:+d})")
    print(f"  Loss sells: {before_all['loss_sell_count']} → {after_all['loss_sell_count']} ({loss_diff:+d})")
    print(f"  PnL impact: {pnl_diff:+,.0f}")
    print(f"  Total trades blocked: {blocked_total}")

    if ws_diff < 0 and pnl_diff >= 0:
        print("\n  ✓ SAFE TO APPLY: Fewer whipsaws, better or equal PnL")
    elif ws_diff < 0:
        print(f"\n  ⚠ APPLY WITH CAUTION: Fewer whipsaws but PnL changed by {pnl_diff:+,.0f}")
    else:
        print("\n  ✗ NO IMPROVEMENT: Parameters did not reduce whipsaw cycles")


if __name__ == "__main__":
    asyncio.run(main())
