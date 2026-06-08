"""KR individual-stock whipsaw diagnosis + stop-loss variant sweep.

User observation (2026-06-09): in choppy markets the bot "sells on the
dip, chases on the rally" — classic whipsaw. The EW-hedge work fixed
the ETF side; this studies the STOCK side (supertrend), which today's
SL tweaks did NOT touch.

Faithful reproduction of the live supertrend stock path:
  - Entry: supertrend(7, 2.0) bullish flip + price > line (confirm=1)
  - Exit, per bar, in priority order:
      1. hard SL  (entry × (1 - 0.12))
      2. dynamic ATR SL  (KR: clamp(atr/price × 2.5, 5%, 20%))
      3. trailing stop  (activate +10%, trail 4% from peak)
      4. take-profit  (KR: clamp(atr/price × 4.0, 8%, 25%))
      5. supertrend bear flip
  (matches engine/risk_manager.py calculate_dynamic_sl_tp +
   check_trailing_stop, config/strategies.yaml supertrend block)

WHIPSAW METRIC: for every SL/trailing exit, look ahead N=10 trading
days. If the stock's close recovers ABOVE the exit price, the stop
was premature → counted as a whipsaw. whipsaw_rate = whipsaws /
(SL + trailing exits).

Variants:
  A  baseline (live: dynamic ATR SL + trailing 10/4 + hard 12%)
  B  wider SL (ATR × 3.5, clamp 8%-25%)
  C  entry-grace (no SL in first 3 bars after entry — entry-noise)
  D  drop_day guard (skip SL on KS200 down >2% that day)
  E  trailing-only (no fixed SL; rely on trailing + signal flip)
  F  B+C combined (wider SL + entry grace)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandas_ta as ta

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_cache"

# Supertrend params (config/strategies.yaml supertrend block)
ST_LENGTH = 7
ST_MULT = 2.0
CONFIRM_BARS = 1

# KR dynamic SL/TP (risk_manager.calculate_dynamic_sl_tp, market=KR)
SL_ATR_MULT = 2.5
SL_CLAMP = (0.05, 0.20)
TP_ATR_MULT = 4.0
TP_CLAMP = (0.08, 0.25)
HARD_SL = 0.12
TRAIL_ACTIVATION = 0.10
TRAIL_PCT = 0.04

ATR_LEN = 14  # for dynamic SL sizing
WHIPSAW_LOOKAHEAD = 10  # trading days
COMMISSION = 0.0005  # 5 bps round-trip-ish per side
SLIPPAGE = 0.0005

REGIME_PROXY = "069500"  # KODEX 200 for drop_day (returns only, abs price dirty)
DROP_DAY_PCT = -0.02


def _load(sym: str) -> pd.DataFrame | None:
    # Try .KS then .KQ
    for suffix in ("KS", "KQ"):
        path = DATA_DIR / f"{sym}.{suffix}__2y__1d.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"])
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None).dt.normalize()
            df = df.set_index("Date").sort_index()
            df.columns = [c.lower() for c in df.columns]
            return df[["open", "high", "low", "close"]].dropna()
    return None


def _kr_stock_symbols() -> list[str]:
    syms = set()
    etf_skip = {
        "069500", "091160", "091170", "091180", "114800", "122630",
        "132030", "148070", "244580", "261240", "305720", "315930",
        "117680", "233740", "229200", "251340", "950160",
    }
    for f in DATA_DIR.glob("*__2y__1d.csv"):
        base = f.name.split("__")[0]
        sym = base.split(".")[0]
        if sym in etf_skip:
            continue
        syms.add(sym)
    return sorted(syms)


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    st = ta.supertrend(df["high"], df["low"], df["close"],
                       length=ST_LENGTH, multiplier=ST_MULT)
    if st is None or st.empty:
        return pd.DataFrame()
    dir_col = next((c for c in st.columns if c.startswith("SUPERTd")), None)
    line_col = next((c for c in st.columns if c.startswith("SUPERT_")), None)
    if dir_col is None or line_col is None:
        return pd.DataFrame()
    df["st_dir"] = st[dir_col]
    df["st_line"] = st[line_col]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN)
    return df.dropna()


@dataclass
class _Variant:
    label: str
    sl_atr_mult: float = SL_ATR_MULT
    sl_clamp: tuple = SL_CLAMP
    hard_sl: float = HARD_SL
    entry_grace_bars: int = 0       # C: ignore SL in first N bars
    drop_day_skip_sl: bool = False  # D
    no_fixed_sl: bool = False       # E: trailing + signal only


@dataclass
class _Result:
    label: str
    trades: int = 0
    sl_exits: int = 0
    trail_exits: int = 0
    tp_exits: int = 0
    signal_exits: int = 0
    whipsaws: int = 0           # SL/trail exit then recovered above ENTRY px
    wins: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    ret_sum_pct: float = 0.0    # sum of per-trade returns (EW proxy)
    pnls: list = field(default_factory=list)
    # Counterfactual: for SL exits, return if we'd held to signal-flip.
    sl_realized_sum: float = 0.0   # sum of actual SL-exit returns (pct)
    sl_counterfac_sum: float = 0.0  # sum of hold-to-signal returns (pct)

    @property
    def whipsaw_rate(self) -> float:
        denom = self.sl_exits + self.trail_exits
        return self.whipsaws / denom if denom else 0.0

    @property
    def sl_realized_avg(self) -> float:
        return self.sl_realized_sum / self.sl_exits if self.sl_exits else 0.0

    @property
    def sl_counterfac_avg(self) -> float:
        return self.sl_counterfac_sum / self.sl_exits if self.sl_exits else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def pf(self) -> float:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf")

    @property
    def avg_trade_pct(self) -> float:
        return self.ret_sum_pct / self.trades if self.trades else 0.0


def _dyn_sl(price: float, atr: float, v: _Variant) -> float:
    if price <= 0 or atr <= 0:
        return HARD_SL
    atr_pct = atr / price
    return max(v.sl_clamp[0], min(v.sl_clamp[1], atr_pct * v.sl_atr_mult))


def _dyn_tp(price: float, atr: float) -> float:
    if price <= 0 or atr <= 0:
        return TP_CLAMP[1]
    atr_pct = atr / price
    return max(TP_CLAMP[0], min(TP_CLAMP[1], atr_pct * TP_ATR_MULT))


def _run_symbol(
    df: pd.DataFrame, v: _Variant, res: _Result, dropday_idx: set,
) -> None:
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    st_dir = df["st_dir"].values
    st_line = df["st_line"].values
    atrs = df["atr"].values
    dates = list(df.index)
    n = len(df)

    in_pos = False
    entry_px = 0.0
    entry_i = 0
    peak = 0.0
    sl_pct = 0.0
    tp_pct = 0.0

    i = 1
    while i < n:
        if not in_pos:
            # Entry: bullish flip (dir -1→+1) + price above line
            bull_flip = st_dir[i] == 1 and st_dir[i - 1] == -1
            if bull_flip and closes[i] > st_line[i]:
                in_pos = True
                entry_px = closes[i] * (1 + SLIPPAGE)
                entry_i = i
                peak = closes[i]
                sl_pct = _dyn_sl(closes[i], atrs[i], v)
                tp_pct = _dyn_tp(closes[i], atrs[i])
            i += 1
            continue

        # In position — evaluate exits at bar i
        bars_held = i - entry_i
        hi, lo, cl = highs[i], lows[i], closes[i]
        peak = max(peak, hi)
        exit_px = None
        exit_reason = None

        grace = bars_held <= v.entry_grace_bars
        skip_sl_today = v.drop_day_skip_sl and (i in dropday_idx)

        # 1. hard SL (intrabar low) — unless grace/dropday/no_fixed
        if not v.no_fixed_sl and not grace and not skip_sl_today:
            hard_stop_px = entry_px * (1 - v.hard_sl)
            dyn_stop_px = entry_px * (1 - sl_pct)
            stop_px = max(hard_stop_px, dyn_stop_px)  # tighter of the two
            if lo <= stop_px:
                exit_px = stop_px
                exit_reason = "sl"

        # 2. trailing stop
        if exit_px is None:
            gain = (peak - entry_px) / entry_px
            if gain >= TRAIL_ACTIVATION:
                trail_px = peak * (1 - TRAIL_PCT)
                if lo <= trail_px:
                    exit_px = trail_px
                    exit_reason = "trail"

        # 3. take-profit (intrabar high)
        if exit_px is None:
            tp_px = entry_px * (1 + tp_pct)
            if hi >= tp_px:
                exit_px = tp_px
                exit_reason = "tp"

        # 4. supertrend bear flip (exit at close)
        if exit_px is None and st_dir[i] == -1:
            exit_px = cl
            exit_reason = "signal"

        if exit_px is not None:
            exit_px *= (1 - SLIPPAGE)
            ret = (exit_px / entry_px) - 1 - 2 * COMMISSION
            res.trades += 1
            res.ret_sum_pct += ret * 100
            res.pnls.append(ret)
            if ret > 0:
                res.wins += 1
                res.gross_profit += ret
            else:
                res.gross_loss += abs(ret)
            if exit_reason == "sl":
                res.sl_exits += 1
            elif exit_reason == "trail":
                res.trail_exits += 1
            elif exit_reason == "tp":
                res.tp_exits += 1
            else:
                res.signal_exits += 1

            # Whipsaw check: SL/trail exit, did price recover above the
            # ENTRY price within N days (i.e. the trade would have turned
            # profitable if held)? This is the meaningful definition —
            # recovering above the exit (stop) price is trivially common.
            if exit_reason in ("sl", "trail"):
                end = min(n, i + 1 + WHIPSAW_LOOKAHEAD)
                fwd_high = highs[i + 1 : end].max() if end > i + 1 else 0.0
                if fwd_high > entry_px:
                    res.whipsaws += 1

            # Counterfactual for hard/dynamic SL exits: what return would
            # holding to the next supertrend bear-flip have produced?
            if exit_reason == "sl":
                res.sl_realized_sum += ret * 100
                # Walk forward to the next bear flip (or series end)
                j = i + 1
                cf_exit = cl  # fallback: current close
                while j < n:
                    if st_dir[j] == -1:
                        cf_exit = closes[j]
                        break
                    j += 1
                else:
                    cf_exit = closes[-1]
                cf_ret = (cf_exit * (1 - SLIPPAGE) / entry_px) - 1 - 2 * COMMISSION
                res.sl_counterfac_sum += cf_ret * 100

            in_pos = False
        i += 1


def _dropday_indices(df: pd.DataFrame, proxy: pd.DataFrame | None) -> set:
    """Set of positional indices in df where KS200 fell >2% that day."""
    if proxy is None:
        return set()
    idx = set()
    proxy_ret = proxy["close"].pct_change()
    date_to_pos = {d: k for k, d in enumerate(df.index)}
    for d, r in proxy_ret.items():
        if d in date_to_pos and r is not None and r <= DROP_DAY_PCT:
            idx.add(date_to_pos[d])
    return idx


def main() -> None:
    syms = _kr_stock_symbols()
    proxy = _load(REGIME_PROXY)
    print(f"KR stock universe: {len(syms)} symbols")

    variants = [
        _Variant(label="A baseline (live)"),
        _Variant(label="B wider SL (ATR×3.5, 8-25%)",
                 sl_atr_mult=3.5, sl_clamp=(0.08, 0.25), hard_sl=0.18),
        _Variant(label="C entry-grace 3 bars", entry_grace_bars=3),
        _Variant(label="D drop_day skip SL", drop_day_skip_sl=True),
        _Variant(label="E trailing-only (no SL)", no_fixed_sl=True),
        _Variant(label="F B+C (wide SL + grace)",
                 sl_atr_mult=3.5, sl_clamp=(0.08, 0.25), hard_sl=0.18,
                 entry_grace_bars=3),
    ]
    results = [_Result(label=v.label) for v in variants]

    loaded = 0
    for sym in syms:
        raw = _load(sym)
        if raw is None or len(raw) < 100:
            continue
        df = _compute_indicators(raw)
        if df.empty or len(df) < 80:
            continue
        loaded += 1
        # Align proxy drop-days to this df's index
        for v, res in zip(variants, results):
            dropday_idx = (
                _dropday_indices(df, proxy) if v.drop_day_skip_sl else set()
            )
            _run_symbol(df, v, res, dropday_idx)

    print(f"Backtested {loaded} symbols with sufficient history\n")

    hdr = (
        f"{'variant':30} {'Trades':>7} {'SL':>5} {'Trail':>6} {'TP':>5} "
        f"{'Sig':>5} {'Whip':>5} {'Whip%':>7} {'Win%':>6} {'PF':>5} {'AvgTr%':>7}"
    )
    print(hdr)
    print("-" * 110)
    for r in results:
        print(
            f"{r.label:30} {r.trades:>7} {r.sl_exits:>5} {r.trail_exits:>6} "
            f"{r.tp_exits:>5} {r.signal_exits:>5} {r.whipsaws:>5} "
            f"{r.whipsaw_rate*100:>6.1f}% {r.win_rate*100:>5.1f}% "
            f"{r.pf:>5.2f} {r.avg_trade_pct:>6.2f}%"
        )

    print()
    print(f"Whip% = of all SL+trailing exits, how many recovered above "
          f"ENTRY px within {WHIPSAW_LOOKAHEAD} days (trade would've won if held)")
    print("AvgTr% = avg per-trade return (net of fees+slippage)")
    print()
    print("=" * 72)
    print("Counterfactual: hard/dynamic SL exits — realized vs hold-to-signal")
    print("=" * 72)
    print(f"{'variant':30} {'SL exits':>9} {'realized%':>10} "
          f"{'if-held%':>10} {'Δ (cost of SL)':>16}")
    print("-" * 72)
    for r in results:
        cost = r.sl_counterfac_avg - r.sl_realized_avg
        print(
            f"{r.label:30} {r.sl_exits:>9} {r.sl_realized_avg:>9.2f}% "
            f"{r.sl_counterfac_avg:>9.2f}% {cost:>+15.2f}pp"
        )
    print()
    base = results[0]
    print("=" * 60)
    print("Δ vs baseline (lower whipsaw + higher avg-trade = win)")
    print("=" * 60)
    for r in results[1:]:
        d_whip = (r.whipsaw_rate - base.whipsaw_rate) * 100
        d_avg = r.avg_trade_pct - base.avg_trade_pct
        d_pf = r.pf - base.pf
        verdict = "✓" if (d_avg > 0.05 and d_whip <= 0.5) else "?"
        print(
            f"  {verdict} {r.label:30} ΔWhip={d_whip:+5.1f}pp  "
            f"ΔAvgTr={d_avg:+.2f}pp  ΔPF={d_pf:+.2f}"
        )


if __name__ == "__main__":
    main()
