"""Analyze recent live trade patterns to diagnose burst-catcher fitness.

Reads MCP-saved trade history JSON and:
  1. Pairs BUYs with SELLs by symbol/market (FIFO matching).
  2. Computes per-trade realized PnL%, hold time.
  3. Buckets exits by reason (profit_taking, trailing, supertrend, hard_sl,
     position_cleanup, etc.).
  4. Computes Maximum Favorable Excursion (MFE) and Maximum Adverse
     Excursion (MAE) using yfinance daily data over the holding window
     (so we can spot "exited too early" — where MFE was much higher than
     realized — and "let it run too long" — where MFE was high but exit
     happened on the way down).
  5. Looks for obvious "burst missed" cases by checking the buy_strategy
     vs the actual price action between buy and sell.
  6. Prints aggregate stats by strategy + per-market.

Pattern labels:
  EARLY_EXIT  : Realized < MFE * 0.5 AND realized > 0  (left meat on table)
  CAUGHT_PEAK : Realized > MFE * 0.8                    (rode it well)
  GAVE_BACK   : MFE > +5% but realized < +1%            (caught burst but lost it)
  SMALL_WIN_BIG_LOSS : Realized < -5%                   (the pattern user complains about)
  CLEAN_LOSS  : MFE < +2% AND realized < -2%            (never worked)
  FLAT        : -2% <= realized <= +2% AND |MFE| < 3%   (noise)
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

import functools
print = functools.partial(print, flush=True)

sys.path.insert(0, ".")

import yfinance as yf
import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.ERROR)


# ── Input file ──
FILE = "/home/chans/.claude/projects/-home-chans-us-stock/e04b372c-45a4-4a61-bc9d-c387ba3973a4/tool-results/mcp-us-stock-get_trade_history-1775661835498.txt"


def load_trades(path: str) -> list[dict]:
    blob = json.loads(Path(path).read_text())
    raw = json.loads(blob["result"])
    # Filter to filled non-paper orders only
    out = []
    for t in raw:
        if t.get("status") != "filled":
            continue
        if t.get("is_paper"):
            continue
        if t.get("filled_quantity", 0) <= 0 and t.get("quantity", 0) <= 0:
            # Some position_cleanup trades have filled_qty=0 but pnl set;
            # treat them as filled if they have pnl
            if t.get("pnl") is None:
                continue
        out.append(t)
    return out


def pair_buys_sells(trades: list[dict]) -> list[dict]:
    """FIFO-pair BUYs and SELLs per (market, symbol).

    Each pair represents one round-trip with realized PnL%.
    """
    # Sort chronologically (oldest first)
    trades = sorted(trades, key=lambda t: t["created_at"])
    queues: dict[tuple, deque] = defaultdict(deque)
    pairs: list[dict] = []

    for t in trades:
        key = (t["market"], t["symbol"])
        side = t["side"].upper()
        if side == "BUY":
            queues[key].append(t)
        elif side == "SELL":
            # Match against the oldest BUY in the queue
            if not queues[key]:
                # Orphan SELL — ignore for round-trip analysis
                continue
            buy = queues[key].popleft()
            buy_price = buy.get("filled_price") or buy.get("price")
            sell_price = t.get("filled_price") or t.get("price")
            if not buy_price or not sell_price:
                continue
            pnl_pct = (float(sell_price) - float(buy_price)) / float(buy_price) * 100
            try:
                buy_dt = datetime.fromisoformat(buy["created_at"])
                sell_dt = datetime.fromisoformat(t["created_at"])
                hold_days = (sell_dt - buy_dt).total_seconds() / 86400
            except Exception:
                hold_days = -1
            pairs.append({
                "market": t["market"],
                "symbol": t["symbol"],
                "name": t.get("name", "")[:20],
                "buy_at": buy["created_at"][:16],
                "sell_at": t["created_at"][:16],
                "buy_strategy": buy.get("strategy", ""),
                "sell_strategy": t.get("strategy", ""),
                "buy_price": float(buy_price),
                "sell_price": float(sell_price),
                "pnl_pct": pnl_pct,
                "hold_days": hold_days,
            })
    return pairs


def fetch_mfe_mae(pair: dict) -> tuple[float, float]:
    """Fetch yfinance daily highs/lows during the hold window and compute
    MFE / MAE relative to buy price.

    Returns (mfe_pct, mae_pct) or (None, None) on failure.
    """
    sym = pair["symbol"]
    # KR symbol → yfinance format
    if pair["market"] == "KR":
        if sym.isdigit():
            sym = f"{sym}.KS"  # try KS first; KOSDAQ is .KQ but we'd need a mapper
    try:
        buy_dt = datetime.fromisoformat(pair["buy_at"])
        sell_dt = datetime.fromisoformat(pair["sell_at"])
        # Pad by 1 day to ensure we get the full window
        from datetime import timedelta
        start = (buy_dt - timedelta(days=1)).date()
        end = (sell_dt + timedelta(days=1)).date()
        df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            return None, None
        # Use the period strictly between buy_dt and sell_dt
        in_window = df[(df.index.date >= buy_dt.date()) & (df.index.date <= sell_dt.date())]
        if in_window.empty:
            return None, None
        max_high = float(in_window["High"].max())
        min_low = float(in_window["Low"].min())
        bp = pair["buy_price"]
        mfe = (max_high - bp) / bp * 100
        mae = (min_low - bp) / bp * 100
        return mfe, mae
    except Exception:
        return None, None


def label_pattern(pair: dict, mfe: float | None, mae: float | None) -> str:
    realized = pair["pnl_pct"]
    if mfe is None:
        # Fall back without MFE
        if realized < -5:
            return "SMALL_WIN_BIG_LOSS"
        if -2 <= realized <= 2:
            return "FLAT_NOISE"
        if realized > 2:
            return "WIN_NO_MFE_DATA"
        return "LOSS_NO_MFE_DATA"

    # With MFE/MAE
    if realized < -5:
        return "SMALL_WIN_BIG_LOSS"
    if mfe >= 5 and realized < 1:
        return "GAVE_BACK"  # caught burst but lost it
    if mfe >= 5 and realized > 0 and realized < mfe * 0.5:
        return "EARLY_EXIT"  # took some, left more
    if mfe >= 3 and realized >= mfe * 0.7:
        return "CAUGHT_BURST_OK"
    if mfe < 2 and -2 <= realized <= 2:
        return "FLAT_NEVER_MOVED"
    if mfe < 2 and realized < -2:
        return "CLEAN_LOSS_NEVER_WORKED"
    return "OTHER"


def main():
    trades = load_trades(FILE)
    print(f"Loaded {len(trades)} filled trades")
    print(f"  BUYs : {sum(1 for t in trades if t['side'] == 'BUY')}")
    print(f"  SELLs: {sum(1 for t in trades if t['side'] == 'SELL')}")

    pairs = pair_buys_sells(trades)
    print(f"\nPaired into {len(pairs)} round-trips")

    print(f"\n{'─' * 100}")
    print("Per-trade detail (fetching MFE/MAE from yfinance...)")
    print(f"{'─' * 100}")

    pattern_counts: dict[str, int] = defaultdict(int)
    pattern_pnl: dict[str, float] = defaultdict(float)
    by_strategy: dict[str, list] = defaultdict(list)

    print(f"{'mkt':<3} {'symbol':<10} {'name':<22} {'buy@':<6} {'sell@':<6} {'hold':>4} "
          f"{'realized%':>10} {'MFE%':>7} {'MAE%':>7} {'pattern':<22} {'buy_strat':<15} {'sell_strat':<25}")
    print("-" * 145)
    for p in pairs:
        mfe, mae = fetch_mfe_mae(p)
        label = label_pattern(p, mfe, mae)
        pattern_counts[label] += 1
        pattern_pnl[label] += p["pnl_pct"]
        by_strategy[p["buy_strategy"]].append((label, p["pnl_pct"]))

        mfe_str = f"{mfe:+6.1f}" if mfe is not None else "  -  "
        mae_str = f"{mae:+6.1f}" if mae is not None else "  -  "
        print(f"{p['market']:<3} {p['symbol']:<10} {p['name']:<22} "
              f"{p['buy_price']:>6.2f} {p['sell_price']:>6.2f} {p['hold_days']:>4.1f} "
              f"{p['pnl_pct']:>+9.2f}% {mfe_str:>7} {mae_str:>7} "
              f"{label:<22} {p['buy_strategy'][:13]:<15} {p['sell_strategy'][:23]:<25}")

    print(f"\n{'─' * 100}")
    print("Pattern aggregate:")
    print(f"{'─' * 100}")
    print(f"{'pattern':<26} {'count':>6} {'%':>6} {'total_pnl%':>12} {'avg_pnl%':>10}")
    total = sum(pattern_counts.values()) or 1
    for label in sorted(pattern_counts, key=lambda x: -pattern_counts[x]):
        n = pattern_counts[label]
        pnl = pattern_pnl[label]
        print(f"{label:<26} {n:>6} {n/total*100:>5.1f}% {pnl:>+11.2f}% {pnl/n:>+9.2f}%")

    print(f"\n{'─' * 100}")
    print("Per-buy-strategy summary:")
    print(f"{'─' * 100}")
    print(f"{'buy_strategy':<22} {'trades':>7} {'wins':>5} {'losses':>7} {'avg_pnl%':>10} "
          f"{'best%':>7} {'worst%':>7} {'patterns'}")
    for strat in sorted(by_strategy.keys()):
        trades_list = by_strategy[strat]
        n = len(trades_list)
        wins = sum(1 for _, pnl in trades_list if pnl > 0)
        losses = sum(1 for _, pnl in trades_list if pnl < 0)
        avg = sum(pnl for _, pnl in trades_list) / n if n else 0
        best = max(pnl for _, pnl in trades_list) if trades_list else 0
        worst = min(pnl for _, pnl in trades_list) if trades_list else 0
        # pattern distribution
        plabels = defaultdict(int)
        for label, _ in trades_list:
            plabels[label] += 1
        ptxt = " ".join(f"{k.split('_')[0]}:{v}" for k, v in sorted(plabels.items(), key=lambda x: -x[1]))
        print(f"{strat[:20]:<22} {n:>7} {wins:>5} {losses:>7} {avg:>+9.2f}% "
              f"{best:>+6.2f}% {worst:>+6.2f}% {ptxt}")


if __name__ == "__main__":
    main()
