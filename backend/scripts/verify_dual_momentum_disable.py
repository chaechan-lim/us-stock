"""Verify dual_momentum US disable improves alpha. Quick 2y backtest."""

import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner", "data", "backtest", "strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

US_BASE = dict(
    market="US",
    use_wide_universe=True,
    default_stop_loss_pct=0.12,
    default_take_profit_pct=0.20,
    max_positions=20,
    max_position_pct=0.08,
    sell_cooldown_days=1,
    whipsaw_max_losses=2,
    min_hold_days=1,
    slippage_pct=0.05,
    volume_adjusted_slippage=True,
    min_confidence=0.50,
    held_sell_bias=0.05,
    held_min_confidence=0.40,
    enable_cash_parking=True,
    cash_parking_threshold=0.30,
)

VARIANTS = [
    ("V0_baseline", ["bollinger_squeeze", "volume_profile", "cis_momentum",
                     "larry_williams", "bnf_deviation", "donchian_breakout"]),
    ("V1_no_dm",    ["bollinger_squeeze", "volume_profile", "cis_momentum",
                     "larry_williams", "bnf_deviation", "donchian_breakout",
                     "dual_momentum"]),
]


async def main():
    print("=" * 100)
    print("  Verify dual_momentum US disable — 2Y wide universe + cash parking")
    print("=" * 100)
    out = []
    for name, disabled in VARIANTS:
        cfg = PipelineConfig(**{**US_BASE, "disabled_strategies": disabled})
        eng = FullPipelineBacktest(cfg)
        t0 = time.time()
        res = await eng.run(period="2y")
        m = res.metrics
        print(f"\n{name}: Ret={m.total_return_pct:+.1f}% Sharpe={m.sharpe_ratio:.2f} "
              f"MDD={m.max_drawdown_pct:.1f}% PF={m.profit_factor:.2f} "
              f"Trades={m.total_trades} Alpha={m.alpha:+.1f}% ({time.time()-t0:.0f}s)")
        out.append((name, m, res.strategy_stats))

    if len(out) == 2:
        b, n = out[0][1], out[1][1]
        print(f"\nDelta: ΔRet={n.total_return_pct-b.total_return_pct:+.1f}pp "
              f"ΔSharpe={n.sharpe_ratio-b.sharpe_ratio:+.2f} "
              f"ΔAlpha={n.alpha-b.alpha:+.1f}pp "
              f"ΔTrades={n.total_trades-b.total_trades:+d}")

    print("\nStrategy contribution (V1_no_dm):")
    for sname, st in sorted(out[-1][2].items(), key=lambda x: x[1]["pnl"], reverse=True):
        if st["trades"] > 0:
            print(f"  {sname:<28} trades={st['trades']:>3d} WR={st['win_rate']:>5.1f}% "
                  f"PnL=${st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
