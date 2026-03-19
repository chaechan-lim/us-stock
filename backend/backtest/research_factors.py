"""Factor Predictive Power Research.

Empirically tests which stock selection factors best predict
future returns. Uses rolling 6-month forward returns across
a wide stock universe over 3 years.

For each factor, calculates:
- IC (Information Coefficient): rank correlation with forward returns
- Top quintile return vs bottom quintile return (spread)
- Hit rate: % of top-quintile stocks that outperform median

Usage:
    cd backend
    python3 -m backtest.research_factors
    python3 -m backtest.research_factors --quick
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Wide universe for factor research (60 stocks)
RESEARCH_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "CRM",
    "ADBE", "ORCL", "CSCO", "AMD", "INTC", "QCOM", "NFLX", "TSLA",
    # Financials
    "JPM", "BAC", "GS", "V", "MA", "BRK-B", "AXP", "MS",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABT", "LLY", "MRK", "TMO", "ABBV",
    # Consumer
    "WMT", "PG", "KO", "MCD", "COST", "HD", "NKE", "SBUX",
    # Energy + Industrial
    "XOM", "CVX", "CAT", "HON", "GE", "LMT", "UPS", "DE",
    # Other
    "DIS", "PYPL", "SQ", "SNOW", "UBER", "ABNB", "COIN", "PLTR",
    "NOW", "PANW", "CRWD", "DDOG",
]

QUICK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "AMD",
    "JPM", "GS", "V", "UNH", "LLY", "MRK", "WMT", "COST",
    "HD", "XOM", "CAT", "GE", "NFLX", "TSLA", "UBER", "CRWD",
]

PERIOD = "5y"
FORWARD_DAYS = 126  # ~6 months forward return
EVAL_INTERVAL = 63  # Evaluate every ~3 months


@dataclass
class FactorResult:
    """Result for a single factor's predictive power."""
    name: str
    avg_ic: float          # Average Information Coefficient (rank correlation)
    ic_ir: float           # IC Information Ratio (avg_ic / std_ic)
    top_q_return: float    # Average return of top quintile
    bot_q_return: float    # Average return of bottom quintile
    spread: float          # Top quintile - bottom quintile
    hit_rate: float        # % of top quintile that beats median
    n_periods: int         # Number of evaluation periods
    description: str = ""


def load_data(symbols: list[str], period: str = PERIOD):
    """Load price + fundamental data."""
    logger.info("Loading price data for %d symbols (%s)...", len(symbols), period)
    price_data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 300:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            price_data[sym] = df
        except Exception as e:
            logger.warning("Failed to load %s: %s", sym, e)

    logger.info("Loaded %d/%d symbols", len(price_data), len(symbols))

    logger.info("Loading fundamentals...")
    fundamentals = {}
    for sym in price_data:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}
            fundamentals[sym] = info
        except Exception as e:
            logger.debug("Fundamental data fetch failed for %s: %s", sym, e)
            fundamentals[sym] = {}

    return price_data, fundamentals


def compute_factors(
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    eval_idx: int,
    common_dates,
    closes: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute all factors at a given evaluation point.

    Returns: {factor_name: {symbol: score}}
    """
    factors = {}
    symbols = list(price_data.keys())

    # === PRICE-BASED FACTORS ===

    # 1. Momentum 12-1 (12-month return minus last month)
    mom_12_1 = {}
    for sym in symbols:
        if eval_idx < 252:
            continue
        p_now = float(closes[sym].iloc[eval_idx])
        p_12m = float(closes[sym].iloc[eval_idx - 252])
        p_1m = float(closes[sym].iloc[eval_idx - 21])
        if p_12m > 0 and p_1m > 0 and not np.isnan(p_now):
            mom_12_1[sym] = (p_now / p_12m - 1) - (p_now / p_1m - 1)
    factors["mom_12_1"] = mom_12_1

    # 2. Momentum 6-month
    mom_6m = {}
    for sym in symbols:
        if eval_idx < 126:
            continue
        p_now = float(closes[sym].iloc[eval_idx])
        p_6m = float(closes[sym].iloc[eval_idx - 126])
        if p_6m > 0 and not np.isnan(p_now):
            mom_6m[sym] = p_now / p_6m - 1
    factors["mom_6m"] = mom_6m

    # 3. Momentum 3-month
    mom_3m = {}
    for sym in symbols:
        if eval_idx < 63:
            continue
        p_now = float(closes[sym].iloc[eval_idx])
        p_3m = float(closes[sym].iloc[eval_idx - 63])
        if p_3m > 0 and not np.isnan(p_now):
            mom_3m[sym] = p_now / p_3m - 1
    factors["mom_3m"] = mom_3m

    # 4. Volatility (annualized, 60-day, INVERSE — lower = better)
    vol_inv = {}
    for sym in symbols:
        if eval_idx < 60:
            continue
        returns = closes[sym].iloc[eval_idx - 60:eval_idx + 1].pct_change().dropna()
        if len(returns) > 10:
            vol = float(returns.std() * np.sqrt(252))
            if vol > 0:
                vol_inv[sym] = -vol  # Inverse: lower vol = higher score
    factors["low_vol"] = vol_inv

    # 5. 52-week high proximity (higher = closer to high = better momentum)
    high_52w = {}
    lookback = min(252, eval_idx)
    for sym in symbols:
        if lookback < 50:
            continue
        recent = closes[sym].iloc[eval_idx - lookback:eval_idx + 1]
        p_now = float(closes[sym].iloc[eval_idx])
        p_high = float(recent.max())
        if p_high > 0 and not np.isnan(p_now):
            high_52w[sym] = p_now / p_high  # 1.0 = at 52w high
    factors["near_52w_high"] = high_52w

    # 6. Price vs SMA200 distance
    sma200_dist = {}
    for sym in symbols:
        if eval_idx < 200:
            continue
        sma = float(closes[sym].iloc[eval_idx - 199:eval_idx + 1].mean())
        p_now = float(closes[sym].iloc[eval_idx])
        if sma > 0 and not np.isnan(p_now):
            sma200_dist[sym] = (p_now - sma) / sma
    factors["above_sma200"] = sma200_dist

    # 7. Volume trend (20d avg volume vs 60d avg volume)
    vol_trend = {}
    for sym in symbols:
        if eval_idx < 60 or "volume" not in price_data[sym].columns:
            continue
        df = price_data[sym]
        # Find the index in df that corresponds to eval_idx in common_dates
        date = common_dates[eval_idx]
        if date in df.index:
            loc = df.index.get_loc(date)
            if loc >= 60:
                vol_20 = float(df["volume"].iloc[loc - 19:loc + 1].mean())
                vol_60 = float(df["volume"].iloc[loc - 59:loc + 1].mean())
                if vol_60 > 0:
                    vol_trend[sym] = vol_20 / vol_60
    factors["vol_trend"] = vol_trend

    # === FUNDAMENTAL FACTORS ===

    # 8. Revenue Growth
    rev_growth = {}
    for sym in symbols:
        rg = fundamentals.get(sym, {}).get("revenueGrowth")
        if rg is not None and not np.isnan(rg):
            rev_growth[sym] = rg
    factors["rev_growth"] = rev_growth

    # 9. Earnings Growth
    earn_growth = {}
    for sym in symbols:
        eg = fundamentals.get(sym, {}).get("earningsGrowth")
        if eg is not None and not np.isnan(eg):
            earn_growth[sym] = eg
    factors["earn_growth"] = earn_growth

    # 10. Profit Margin
    profit_margin = {}
    for sym in symbols:
        pm = fundamentals.get(sym, {}).get("profitMargins")
        if pm is not None and not np.isnan(pm):
            profit_margin[sym] = pm
    factors["profit_margin"] = profit_margin

    # 11. ROE
    roe = {}
    for sym in symbols:
        r = fundamentals.get(sym, {}).get("returnOnEquity")
        if r is not None and not np.isnan(r):
            roe[sym] = r
    factors["roe"] = roe

    # 12. Forward PE (INVERSE — lower PE = cheaper = potentially better)
    fwd_pe_inv = {}
    for sym in symbols:
        pe = fundamentals.get(sym, {}).get("forwardPE")
        if pe and pe > 0 and not np.isnan(pe):
            fwd_pe_inv[sym] = -pe  # Inverse: lower PE = higher score
    factors["cheap_fwd_pe"] = fwd_pe_inv

    # 13. PEG Ratio (INVERSE — lower PEG = better)
    peg_inv = {}
    for sym in symbols:
        peg = fundamentals.get(sym, {}).get("pegRatio")
        if peg and peg > 0 and not np.isnan(peg) and peg < 10:
            peg_inv[sym] = -peg
    factors["cheap_peg"] = peg_inv

    # 14. Debt-to-Equity (INVERSE — lower = better)
    low_debt = {}
    for sym in symbols:
        de = fundamentals.get(sym, {}).get("debtToEquity")
        if de is not None and not np.isnan(de) and de >= 0:
            low_debt[sym] = -de
    factors["low_debt"] = low_debt

    # 15. Institutional Ownership
    inst_own = {}
    for sym in symbols:
        io = fundamentals.get(sym, {}).get("heldPercentInstitutions")
        if io is not None and not np.isnan(io):
            inst_own[sym] = io
    factors["inst_ownership"] = inst_own

    # 16. Analyst Target Upside (target price / current price - 1)
    target_upside = {}
    for sym in symbols:
        target = fundamentals.get(sym, {}).get("targetMeanPrice")
        current = fundamentals.get(sym, {}).get("currentPrice")
        if target and current and current > 0:
            target_upside[sym] = target / current - 1
    factors["target_upside"] = target_upside

    # === COMPOSITE FACTORS ===

    # 17. Quality composite (ROE + margin + low debt)
    quality = {}
    for sym in symbols:
        components = []
        r = roe.get(sym)
        if r is not None:
            components.append(min(r, 0.5))
        m = profit_margin.get(sym)
        if m is not None:
            components.append(min(m, 0.4))
        d = low_debt.get(sym)
        if d is not None:
            components.append(max(d, -2) / 2)  # Normalize
        if len(components) >= 2:
            quality[sym] = float(np.mean(components))
    factors["quality_composite"] = quality

    # 18. Momentum + Quality (combined)
    mom_quality = {}
    for sym in symbols:
        m = mom_12_1.get(sym)
        q = quality.get(sym)
        if m is not None and q is not None:
            # Z-score each and combine
            mom_quality[sym] = m * 0.6 + q * 0.4
    factors["mom_quality"] = mom_quality

    # 19. Growth + Momentum
    growth_mom = {}
    for sym in symbols:
        m = mom_6m.get(sym)
        rg = rev_growth.get(sym)
        if m is not None and rg is not None:
            growth_mom[sym] = m * 0.5 + rg * 0.5
    factors["growth_mom"] = growth_mom

    # 20. GARP (Growth at Reasonable Price) — growth / PE
    garp = {}
    for sym in symbols:
        rg = rev_growth.get(sym)
        pe = fundamentals.get(sym, {}).get("forwardPE")
        if rg is not None and pe and pe > 0:
            garp[sym] = rg / pe  # Higher growth relative to PE = better
    factors["garp"] = garp

    return factors


def evaluate_factor(
    factor_name: str,
    factor_values: dict[str, float],
    forward_returns: dict[str, float],
) -> dict | None:
    """Evaluate a single factor's predictive power for one period."""
    # Need both factor and forward return for each stock
    common = set(factor_values.keys()) & set(forward_returns.keys())
    if len(common) < 10:
        return None

    syms = sorted(common)
    f_vals = [factor_values[s] for s in syms]
    r_vals = [forward_returns[s] for s in syms]

    # Information Coefficient (Spearman rank correlation)
    ic, _ = stats.spearmanr(f_vals, r_vals)
    if np.isnan(ic):
        return None

    # Quintile analysis
    n = len(syms)
    q_size = n // 5
    if q_size < 2:
        return None

    ranked = sorted(zip(syms, f_vals, r_vals), key=lambda x: x[1], reverse=True)
    top_q = ranked[:q_size]
    bot_q = ranked[-q_size:]

    top_ret = np.mean([x[2] for x in top_q])
    bot_ret = np.mean([x[2] for x in bot_q])
    spread = top_ret - bot_ret

    # Hit rate: % of top quintile that beats median return
    median_ret = np.median(r_vals)
    hits = sum(1 for x in top_q if x[2] > median_ret)
    hit_rate = hits / len(top_q)

    return {
        "ic": ic,
        "top_ret": top_ret,
        "bot_ret": bot_ret,
        "spread": spread,
        "hit_rate": hit_rate,
    }


def run_research(
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
) -> list[FactorResult]:
    """Run full factor research across all evaluation periods."""

    # Align dates
    symbols = list(price_data.keys())
    common_dates = None
    for sym in symbols:
        idx = price_data[sym].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()

    closes = pd.DataFrame({
        sym: price_data[sym].loc[common_dates, "close"] for sym in symbols
    })

    n_bars = len(common_dates)
    logger.info("Aligned data: %d stocks, %d bars", len(symbols), n_bars)

    # Collect IC values for each factor across periods
    factor_ics: dict[str, list[dict]] = {}

    eval_points = list(range(252, n_bars - FORWARD_DAYS, EVAL_INTERVAL))
    logger.info("Evaluating %d periods...", len(eval_points))

    for eval_idx in eval_points:
        date = common_dates[eval_idx]

        # Compute factors at this point
        factors = compute_factors(price_data, fundamentals, eval_idx, common_dates, closes)

        # Compute forward returns
        fwd_idx = eval_idx + FORWARD_DAYS
        forward_returns = {}
        for sym in symbols:
            p_now = float(closes[sym].iloc[eval_idx])
            p_fwd = float(closes[sym].iloc[fwd_idx])
            if p_now > 0 and not np.isnan(p_fwd):
                forward_returns[sym] = p_fwd / p_now - 1

        # Evaluate each factor
        for fname, fvalues in factors.items():
            result = evaluate_factor(fname, fvalues, forward_returns)
            if result:
                if fname not in factor_ics:
                    factor_ics[fname] = []
                factor_ics[fname].append(result)

    # Aggregate results
    results = []
    for fname, periods in factor_ics.items():
        if len(periods) < 3:
            continue

        ics = [p["ic"] for p in periods]
        avg_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        ic_ir = avg_ic / std_ic if std_ic > 0 else 0

        avg_top = float(np.mean([p["top_ret"] for p in periods]))
        avg_bot = float(np.mean([p["bot_ret"] for p in periods]))
        avg_spread = float(np.mean([p["spread"] for p in periods]))
        avg_hit = float(np.mean([p["hit_rate"] for p in periods]))

        results.append(FactorResult(
            name=fname,
            avg_ic=round(avg_ic, 4),
            ic_ir=round(ic_ir, 4),
            top_q_return=round(avg_top * 100, 1),
            bot_q_return=round(avg_bot * 100, 1),
            spread=round(avg_spread * 100, 1),
            hit_rate=round(avg_hit * 100, 1),
            n_periods=len(periods),
        ))

    # Sort by IC (absolute value)
    results.sort(key=lambda r: abs(r.avg_ic), reverse=True)
    return results


def print_results(results: list[FactorResult]) -> None:
    """Print factor research results."""
    print("\n" + "=" * 100)
    print("FACTOR PREDICTIVE POWER RESEARCH")
    print("=" * 100)
    print(f"{'Factor':<20s} {'Avg IC':>7s} {'IC IR':>6s} {'Top Q':>7s} "
          f"{'Bot Q':>7s} {'Spread':>7s} {'Hit%':>6s} {'N':>4s} {'Grade':>6s}")
    print("-" * 100)

    for r in results:
        # Grade based on IC
        if abs(r.avg_ic) >= 0.10:
            grade = "A"
        elif abs(r.avg_ic) >= 0.05:
            grade = "B"
        elif abs(r.avg_ic) >= 0.03:
            grade = "C"
        else:
            grade = "D"

        # Direction indicator
        direction = "+" if r.avg_ic > 0 else "-"

        print(
            f"  {r.name:<18s} {r.avg_ic:>+.4f} {r.ic_ir:>6.2f} "
            f"{r.top_q_return:>+6.1f}% {r.bot_q_return:>+6.1f}% "
            f"{r.spread:>+6.1f}% {r.hit_rate:>5.1f}% {r.n_periods:>4d} "
            f"   [{grade}{direction}]"
        )

    print("=" * 100)

    print("\n--- Legend ---")
    print("IC: Information Coefficient (rank correlation with future 6m returns)")
    print("    |IC| > 0.10 = strong, > 0.05 = moderate, > 0.03 = weak")
    print("IC IR: IC / std(IC) — consistency of signal (higher = more reliable)")
    print("Top/Bot Q: Average 6-month return of top/bottom quintile by factor")
    print("Spread: Top quintile return - Bottom quintile return")
    print("Hit%: % of top quintile stocks that beat median return")

    # Top recommendations
    strong = [r for r in results if abs(r.avg_ic) >= 0.05 and r.spread > 0]
    if strong:
        print("\n--- Recommended Factors for Stock Selection ---")
        for r in strong[:7]:
            print(f"  * {r.name}: IC={r.avg_ic:+.3f}, spread={r.spread:+.1f}%/6m, hit={r.hit_rate:.0f}%")


async def main():
    t0 = time.time()

    quick = "--quick" in sys.argv
    universe = QUICK_UNIVERSE if quick else RESEARCH_UNIVERSE
    logger.info("Mode: %s (%d stocks)", "quick" if quick else "full", len(universe))

    price_data, fundamentals = load_data(universe, PERIOD)

    if len(price_data) < 15:
        logger.error("Not enough data. Exiting.")
        sys.exit(1)

    results = run_research(price_data, fundamentals)
    print_results(results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
