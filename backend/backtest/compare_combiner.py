"""Compare combiner modes: HOLD-in-denominator vs HOLD-excluded.

Runs all 14 strategies on 8 representative stocks, comparing:
  Mode A (current): HOLD signals counted in total_weight denominator
  Mode B (proposed): Only active (BUY/SELL) signals counted in denominator,
                     with min_active_ratio threshold

This reveals whether HOLD dilution is hurting signal quality.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

sys.path.insert(0, ".")

from data.indicator_service import IndicatorService
from strategies.registry import StrategyRegistry
from strategies.combiner import SignalCombiner
from strategies.config_loader import StrategyConfigLoader
from core.enums import SignalType
from backtest.simulator import BacktestSimulator, SimConfig
from backtest.metrics import MetricsCalculator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SYMBOLS = ["AAPL", "NVDA", "MSFT", "JPM", "XOM", "JNJ", "AMZN", "TSLA"]
PERIOD = "3y"
INITIAL_EQUITY = 10_000
MARKET_STATE = "uptrend"
MIN_CONFIDENCE = 0.50


class CombinerModeB(SignalCombiner):
    """Modified combiner: exclude HOLD from denominator."""

    def __init__(self, consensus_config=None, min_active_ratio: float = 0.20):
        super().__init__(consensus_config)
        self._min_active_ratio = min_active_ratio

    def combine(self, signals, weights, min_confidence=0.50):
        if not signals:
            from strategies.base import Signal
            return Signal(
                signal_type=SignalType.HOLD, confidence=0.0,
                strategy_name="combiner", reason="No signals",
            )

        effective_weights, agreement_scores = self._apply_consensus(signals, weights)

        buy_score = 0.0
        sell_score = 0.0
        active_weight = 0.0
        total_weight = 0.0
        reasons = []
        all_indicators = {}

        for signal in signals:
            w = effective_weights.get(signal.strategy_name, 0.0)
            if w <= 0:
                continue
            total_weight += w

            if signal.signal_type == SignalType.BUY:
                weighted_conf = signal.confidence * w
                buy_score += weighted_conf
                active_weight += w
                reasons.append(f"+{signal.strategy_name}({signal.confidence:.0%})")
            elif signal.signal_type == SignalType.SELL:
                weighted_conf = signal.confidence * w
                sell_score += weighted_conf
                active_weight += w
                reasons.append(f"-{signal.strategy_name}({signal.confidence:.0%})")

            for k, v in signal.indicators.items():
                all_indicators[f"{signal.strategy_name}.{k}"] = v

        for group_name, score in agreement_scores.items():
            all_indicators[f"combiner.{group_name}_agreement"] = score

        if active_weight == 0 or total_weight == 0:
            from strategies.base import Signal
            return Signal(
                signal_type=SignalType.HOLD, confidence=0.0,
                strategy_name="combiner", reason="No active signals",
            )

        # Min active ratio check: at least 20% of strategies must be active
        active_ratio = active_weight / total_weight
        if active_ratio < self._min_active_ratio:
            from strategies.base import Signal
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=max(buy_score, sell_score) / active_weight if active_weight > 0 else 0,
                strategy_name="combiner",
                reason=f"Active ratio too low ({active_ratio:.0%} < {self._min_active_ratio:.0%})",
            )

        # Normalize by active weight only (not total weight)
        buy_norm = buy_score / active_weight
        sell_norm = sell_score / active_weight

        from strategies.base import Signal
        if buy_norm > sell_norm and buy_norm >= min_confidence:
            return Signal(
                signal_type=SignalType.BUY, confidence=buy_norm,
                strategy_name="combiner",
                reason=f"BUY consensus: {', '.join(reasons)}",
                indicators=all_indicators,
            )
        if sell_norm > buy_norm and sell_norm >= min_confidence:
            return Signal(
                signal_type=SignalType.SELL, confidence=sell_norm,
                strategy_name="combiner",
                reason=f"SELL consensus: {', '.join(reasons)}",
                indicators=all_indicators,
            )

        return Signal(
            signal_type=SignalType.HOLD,
            confidence=max(buy_norm, sell_norm),
            strategy_name="combiner",
            reason=f"Below threshold ({max(buy_norm, sell_norm):.0%} < {min_confidence:.0%})",
            indicators=all_indicators,
        )


@dataclass
class ModeResult:
    symbol: str
    mode: str
    total_trades: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    mdd_pct: float = 0.0
    profit_factor: float = 0.0
    avg_signals_per_bar: float = 0.0


async def run_combined_backtest(
    symbol: str, df: pd.DataFrame, combiner: SignalCombiner,
    registry: StrategyRegistry, weights: dict[str, float],
    mode_name: str,
) -> ModeResult:
    """Run backtest with combined signals from all strategies."""
    indicator_svc = IndicatorService()
    df = indicator_svc.add_all_indicators(df.copy())

    strategies = registry.get_enabled()
    min_bars = max(s.min_candles_required for s in strategies)

    signals = {}
    active_counts = []

    for i in range(min_bars, len(df)):
        window = df.iloc[:i + 1]
        strategy_signals = []
        for strategy in strategies:
            try:
                signal = await strategy.analyze(window, symbol)
                strategy_signals.append(signal)
            except Exception as e:
                logger.debug("Strategy %s failed for %s at bar %d: %s", strategy.name, symbol, i, e)

        combined = combiner.combine(strategy_signals, weights, MIN_CONFIDENCE)

        # Track active signal count
        active = sum(
            1 for s in strategy_signals
            if s.signal_type != SignalType.HOLD
        )
        active_counts.append(active)

        if combined.signal_type != SignalType.HOLD:
            signals[i] = combined

    # Simulate
    sim = BacktestSimulator(config=SimConfig(initial_equity=INITIAL_EQUITY))
    sim.run(df, signals, symbol)

    metrics = MetricsCalculator.calculate(
        equity_curve=sim.equity_curve,
        trades=sim.trades,
        initial_equity=INITIAL_EQUITY,
    )

    return ModeResult(
        symbol=symbol,
        mode=mode_name,
        total_trades=metrics.total_trades,
        win_rate=metrics.win_rate,
        total_return_pct=metrics.total_return_pct,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        mdd_pct=metrics.max_drawdown_pct,
        profit_factor=metrics.profit_factor,
        avg_signals_per_bar=sum(active_counts) / len(active_counts) if active_counts else 0,
    )


async def main():
    print("=" * 80)
    print("COMBINER MODE COMPARISON: HOLD-in-denominator vs HOLD-excluded")
    print("=" * 80)
    print(f"Symbols: {SYMBOLS}")
    print(f"Period: {PERIOD}, Market state: {MARKET_STATE}")
    print(f"Strategies: 14, Min confidence: {MIN_CONFIDENCE}")
    print()

    # Setup
    registry = StrategyRegistry()
    config_loader = StrategyConfigLoader()
    consensus_cfg = config_loader.get_consensus_config()

    weights = config_loader.get_profile_weights(MARKET_STATE)
    print(f"Weights ({MARKET_STATE}):")
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.2f}")
    print()

    combiner_a = SignalCombiner(consensus_config=consensus_cfg)
    combiner_b = CombinerModeB(consensus_config=consensus_cfg, min_active_ratio=0.15)

    results_a: list[ModeResult] = []
    results_b: list[ModeResult] = []

    for symbol in SYMBOLS:
        print(f"--- {symbol} ---")
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period=PERIOD, interval="1d")
            if raw.empty or len(raw) < 100:
                print(f"  Skipped (insufficient data)")
                continue
            raw.columns = [c.lower() for c in raw.columns]
            df = raw[["open", "high", "low", "close", "volume"]].copy()

            # Mode A: current
            ra = await run_combined_backtest(
                symbol, df, combiner_a, registry, weights, "A_current",
            )
            results_a.append(ra)

            # Mode B: HOLD excluded
            rb = await run_combined_backtest(
                symbol, df, combiner_b, registry, weights, "B_hold_excluded",
            )
            results_b.append(rb)

            print(
                f"  A(current):       trades={ra.total_trades:3d}  "
                f"CAGR={ra.cagr:+7.1%}  Sharpe={ra.sharpe:5.2f}  "
                f"MDD={ra.mdd_pct:5.1f}%  WR={ra.win_rate:4.1f}%  "
                f"PF={ra.profit_factor:4.2f}  avg_active={ra.avg_signals_per_bar:.1f}"
            )
            print(
                f"  B(hold_excluded): trades={rb.total_trades:3d}  "
                f"CAGR={rb.cagr:+7.1%}  Sharpe={rb.sharpe:5.2f}  "
                f"MDD={rb.mdd_pct:5.1f}%  WR={rb.win_rate:4.1f}%  "
                f"PF={rb.profit_factor:4.2f}  avg_active={rb.avg_signals_per_bar:.1f}"
            )

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results_a and results_b:
        avg_a = {
            "trades": sum(r.total_trades for r in results_a) / len(results_a),
            "cagr": sum(r.cagr for r in results_a) / len(results_a),
            "sharpe": sum(r.sharpe for r in results_a) / len(results_a),
            "mdd": sum(r.mdd_pct for r in results_a) / len(results_a),
            "wr": sum(r.win_rate for r in results_a) / len(results_a),
            "pf": sum(r.profit_factor for r in results_a) / len(results_a),
        }
        avg_b = {
            "trades": sum(r.total_trades for r in results_b) / len(results_b),
            "cagr": sum(r.cagr for r in results_b) / len(results_b),
            "sharpe": sum(r.sharpe for r in results_b) / len(results_b),
            "mdd": sum(r.mdd_pct for r in results_b) / len(results_b),
            "wr": sum(r.win_rate for r in results_b) / len(results_b),
            "pf": sum(r.profit_factor for r in results_b) / len(results_b),
        }

        print(f"{'Metric':<20} {'A(current)':>12} {'B(hold_excl)':>12} {'Winner':>10}")
        print("-" * 56)
        for key, label in [
            ("trades", "Avg Trades"),
            ("cagr", "Avg CAGR"),
            ("sharpe", "Avg Sharpe"),
            ("mdd", "Avg MDD"),
            ("wr", "Avg Win Rate"),
            ("pf", "Avg Profit Factor"),
        ]:
            va, vb = avg_a[key], avg_b[key]
            if key == "mdd":
                winner = "A" if abs(va) < abs(vb) else "B"
            elif key == "trades":
                winner = "-"
            else:
                winner = "A" if va > vb else "B" if vb > va else "TIE"

            if key in ("cagr",):
                print(f"  {label:<18} {va:>11.1%} {vb:>11.1%} {winner:>10}")
            elif key in ("mdd", "wr"):
                print(f"  {label:<18} {va:>10.1f}% {vb:>10.1f}% {winner:>10}")
            else:
                print(f"  {label:<18} {va:>11.2f} {vb:>11.2f} {winner:>10}")

    print()
    print("Note: Mode B uses min_active_ratio=0.15 (at least 15% of strategies must be active)")


if __name__ == "__main__":
    asyncio.run(main())
