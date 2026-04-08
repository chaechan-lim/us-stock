"""Debug why KR full pipeline backtest produced 0 trades.

Hypothesis 1: yfinance can't pull .KS/.KQ data
Hypothesis 2: data shape is fine but screener filters everything out
Hypothesis 3: regime symbol (069500.KS) loads but stock symbols don't

Steps:
1. Try loading the default KR universe via BacktestDataLoader
2. Print rows/columns/date range for each
3. Try a single screening pass on a date
"""

import sys
import functools

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner", "data"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.data_loader import BacktestDataLoader
from backtest.full_pipeline import DEFAULT_KR_UNIVERSE


def main():
    loader = BacktestDataLoader()
    print(f"DEFAULT_KR_UNIVERSE has {len(DEFAULT_KR_UNIVERSE)} symbols")
    print(f"Regime symbol: 069500.KS")
    print()

    test_symbols = ["069500.KS"] + DEFAULT_KR_UNIVERSE[:10]
    results = {}
    for sym in test_symbols:
        try:
            data = loader.load(sym, period="2y")
            df = data.df
            results[sym] = (len(df), str(df.index[0])[:10] if len(df) else "-",
                           str(df.index[-1])[:10] if len(df) else "-")
        except Exception as e:
            results[sym] = (0, "FAIL", str(e)[:60])

    print(f"{'symbol':<14} {'bars':>6} {'first':<12} {'last':<12}")
    print("-" * 50)
    for sym, (n, first, last) in results.items():
        flag = "✓" if n > 200 else "✗"
        print(f"{sym:<14} {n:>6} {first:<12} {last:<12} {flag}")

    print()
    print("Loading FULL KR universe via load_multiple...")
    loaded = loader.load_multiple(DEFAULT_KR_UNIVERSE, period="2y")
    print(f"Loaded {len(loaded)} / {len(DEFAULT_KR_UNIVERSE)} symbols")

    if loaded:
        bars = [len(d.df) for d in loaded.values()]
        print(f"Bars range: {min(bars)} ~ {max(bars)}")
        empty = [s for s, d in loaded.items() if len(d.df) < 100]
        if empty:
            print(f"Symbols with <100 bars: {empty}")
    print()

    # Try one screening pass
    from scanner.indicator_screener import IndicatorScreener
    screener = IndicatorScreener()

    print("Running screener on full KR universe at last bar...")
    grades = {}
    scores_list = []
    for sym, data in loaded.items():
        try:
            score = screener.score(data.df, sym)
            grades[score.grade] = grades.get(score.grade, 0) + 1
            scores_list.append((sym, score.total_score, score.grade))
        except Exception as e:
            print(f"  scoring {sym} failed: {e}")
    print(f"Grade distribution: {grades}")
    scores_list.sort(key=lambda x: x[1], reverse=True)
    print(f"Top 10 by score:")
    for sym, sc, gr in scores_list[:10]:
        print(f"  {sym:<14} {sc:>5.1f}  {gr}")
    print(f"Bottom 5:")
    for sym, sc, gr in scores_list[-5:]:
        print(f"  {sym:<14} {sc:>5.1f}  {gr}")

    # Filter B+ candidates
    grade_filtered = screener.filter_candidates(
        [screener.score(d.df, s) for s, d in loaded.items()],
        max_candidates=30,
    )
    print(f"\nfilter_candidates(min_grade=B) returned {len(grade_filtered)} symbols")

    # Now try strategy execution on a B+ symbol
    if grade_filtered:
        from strategies.registry import StrategyRegistry
        from strategies.combiner import SignalCombiner
        from strategies.config_loader import StrategyConfigLoader

        loader_cfg = StrategyConfigLoader()
        try:
            loader_cfg.reload()
        except Exception:
            pass
        registry = StrategyRegistry(config_loader=loader_cfg)

        sample_sym = grade_filtered[0].symbol
        sample_data = loaded[sample_sym]
        df_window = sample_data.df.iloc[:-1]  # exclude current bar

        print(f"\nRunning all strategies on {sample_sym} (last bar):")
        # Get KR disabled list
        try:
            kr_disabled = set(loader_cfg.get_market_disabled_strategies("KR"))
        except AttributeError:
            mkts = loader_cfg.get_markets() if hasattr(loader_cfg, 'get_markets') else {}
            kr_disabled = set(mkts.get("KR", {}).get("disabled_strategies", []))
        print(f"KR disabled_strategies: {kr_disabled}")

        import asyncio
        async def run_strats():
            sigs = []
            for strat in registry.get_enabled():
                if strat.name in kr_disabled:
                    continue
                try:
                    s = await strat.analyze(df_window, sample_sym)
                    sigs.append(s)
                    t = s.signal_type.name if hasattr(s.signal_type, 'name') else str(s.signal_type)
                    print(f"  {strat.name:<28} {t:<5} conf={s.confidence:.2f}  {s.reason[:60]}")
                except Exception as e:
                    print(f"  {strat.name:<28} EXCEPTION: {type(e).__name__}: {e}")
            return sigs

        sigs = asyncio.run(run_strats())

        # Try combiner
        consensus_cfg = loader_cfg.get_consensus_config()
        combiner = SignalCombiner(consensus_config=consensus_cfg, min_active_ratio=0.15)
        weights = registry.get_profile_weights("uptrend")
        # Filter out KR-disabled
        weights = {k: v for k, v in weights.items() if k not in kr_disabled}
        print(f"\nProfile 'uptrend' weights (post KR-disable): {weights}")
        combined = combiner.combine(sigs, weights, min_confidence=0.40)
        print(f"\nCombined signal: {combined.signal_type} conf={combined.confidence:.2f} via {combined.strategy_name}")
        print(f"Reason: {combined.reason}")


if __name__ == "__main__":
    main()
