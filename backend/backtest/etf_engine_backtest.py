"""ETF Engine backtest harness.

Standalone simulator for the live ETFEngine logic. Reuses:
  - scanner/etf_universe.py (ETFUniverse + risk_rules from yaml)
  - data/market_state.py (regime detection from SPY/VIX)

NOT integrated with FullPipelineBacktest. ETF Engine is its own
engine that holds a separate cash pool with regime-based bull/bear/
sector rotation. The full pipeline (stocks) doesn't simulate ETF
trading; this harness fills that gap so per-cap / per-rule changes
can be backtest-validated before live deployment.

Limitations:
  - Daily bars only (no intraday). Live engine evaluates every cycle
    but this simulator picks one decision per day at close.
  - VIX is fetched as ^VIX. Skips if unavailable.
  - Sector rotation uses sector ETF prices directly (no separate
    sector_analyzer ranking). Top-N by trailing 1-month return.
  - Mutual exclusivity (sibling 1x/2x sell-before-buy) is honored.

Usage:
    cfg = ETFBacktestConfig(initial_capital=10_000, period="2y")
    res = await ETFBacktestEngine(cfg).run()
    print(res.summary())
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf

# Local imports — backend/ on sys.path is expected (set by caller scripts).
from data.market_state import MarketRegime, MarketStateDetector
from scanner.etf_universe import ETFUniverse

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────


@dataclass
class ETFBacktestConfig:
    """Knobs that change ETF engine behavior (mirror live yaml)."""
    initial_capital: float = 10_000.0
    period: str = "2y"
    market: str = "US"
    commission_per_order: float = 1.0
    slippage_pct: float = 0.05  # 5 bps per side
    # Risk caps (override the universe yaml when explicitly set)
    max_portfolio_pct: float | None = None
    max_single_etf_pct: float | None = None
    max_hold_days: int | None = None
    # Regime allocation (per-ETF allocation as fraction of equity).
    # Defaults match ETFEngine._regime_alloc_pct.
    regime_alloc_pct: dict[str, float] = field(default_factory=lambda: {
        "strong_uptrend": 0.10,
        "uptrend":        0.07,
        "sideways":       0.00,
        "weak_downtrend": 0.03,
        "downtrend":      0.05,
    })
    # Sector rotation
    max_regime_etfs: int = 2
    max_sector_etfs: int = 3
    # Bear ETF guards (match ETFEngine.__init__ defaults)
    bear_min_distance_pct: float = -5.0  # SPY must be this much below SMA200
    bear_min_confidence: float = 0.7
    bear_size_ratio: float = 0.4
    # Universe yaml override (None = use default scanner/etf_universe paths)
    universe_config_path: str | None = None


# ── Position / Result types ────────────────────────────────────────────


@dataclass
class _ETFPosition:
    symbol: str
    quantity: int
    entry_price: float
    entry_date: date
    reason: str  # "regime_bull" / "regime_bear" / "sector"


@dataclass
class ETFBacktestResult:
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    trades: int
    final_value: float
    daily_equity: list[float]
    initial_capital: float
    regime_changes: int

    def summary(self) -> str:
        return (
            f"Ret={self.total_return_pct:+.2f}%  "
            f"Sharpe={self.sharpe:+.2f}  "
            f"MDD={self.max_drawdown_pct:.2f}%  "
            f"Trades={self.trades}  "
            f"Final=${self.final_value:.0f}  "
            f"RegimeFlips={self.regime_changes}"
        )


# ── Engine ─────────────────────────────────────────────────────────────


class ETFBacktestEngine:
    def __init__(self, cfg: ETFBacktestConfig):
        self._cfg = cfg
        self._etf = ETFUniverse(cfg.universe_config_path)
        self._detector = MarketStateDetector()
        # Effective risk caps (cfg overrides universe yaml)
        rules = self._etf.risk_rules
        self._max_portfolio_pct = (
            cfg.max_portfolio_pct if cfg.max_portfolio_pct is not None
            else rules.max_portfolio_pct
        )
        self._max_single_pct = (
            cfg.max_single_etf_pct if cfg.max_single_etf_pct is not None
            else rules.max_single_etf_pct
        )
        self._max_hold_days = (
            cfg.max_hold_days if cfg.max_hold_days is not None
            else rules.max_hold_days
        )

    # ----- Data fetching ----------------------------------------------

    def _fetch_close(self, symbol: str, start: date, end: date) -> pd.Series | None:
        try:
            df = yf.download(
                symbol, start=start, end=end + timedelta(days=2),
                progress=False, auto_adjust=False, threads=False,
            )
            if df is None or df.empty:
                return None
            if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)
            return df["Close"].dropna()
        except Exception as e:
            logger.debug("fetch %s failed: %s", symbol, e)
            return None

    def _load_all_prices(self) -> dict[str, pd.Series]:
        """Load close-price series for every ETF in universe + SPY + ^VIX."""
        symbols = set(self._etf.all_etf_symbols)
        symbols.update(["SPY", "^VIX"])
        # Compute window from cfg.period (e.g. "2y")
        end = date.today()
        years = int(self._cfg.period.rstrip("y")) if self._cfg.period.endswith("y") else 2
        start = end - timedelta(days=years * 365 + 30)  # padding for SMA200 warmup
        prices: dict[str, pd.Series] = {}
        for sym in sorted(symbols):
            s = self._fetch_close(sym, start, end)
            if s is not None and len(s) > 0:
                prices[sym] = s
        return prices

    # ----- Regime → ETF decisions -------------------------------------

    def _pick_regime_etfs(self, regime: MarketRegime) -> list[str]:
        """Return the bull/bear leveraged ETFs to hold for the regime."""
        regime_str = regime.value if hasattr(regime, "value") else str(regime)
        return self._etf.get_regime_etfs(regime_str)[: self._cfg.max_regime_etfs]

    def _pick_top_sectors(
        self,
        prices: dict[str, pd.Series],
        date_idx: int,
        sector_symbols: list[str],
    ) -> list[str]:
        """Top-N sector ETFs by multi-horizon weighted return (matches live
        SectorAnalyzer):
            score = 0.20 × r1w + 0.40 × r1m + 0.40 × r3m
        Then min-max normalize to 0-100 and filter by min_score=60
        (live default in scanner/sector_analyzer.py).
        """
        w_1w, w_1m, w_3m = 0.20, 0.40, 0.40
        min_score = 60.0
        # Trading-day lookbacks: 5, 21, 63
        lb_1w, lb_1m, lb_3m = 5, 21, 63
        raw: list[tuple[str, float]] = []
        for sym in sector_symbols:
            s = prices.get(sym)
            if s is None or date_idx >= len(s) or date_idx < lb_3m:
                continue
            try:
                cur = float(s.iloc[date_idx])
                p1w = float(s.iloc[date_idx - lb_1w])
                p1m = float(s.iloc[date_idx - lb_1m])
                p3m = float(s.iloc[date_idx - lb_3m])
                if min(p1w, p1m, p3m) <= 0:
                    continue
                r1w = (cur / p1w - 1) * 100
                r1m = (cur / p1m - 1) * 100
                r3m = (cur / p3m - 1) * 100
                raw.append((sym, r1w * w_1w + r1m * w_1m + r3m * w_3m))
            except Exception:
                continue
        if not raw:
            return []
        # Min-max normalize to 0-100
        vals = [v for _, v in raw]
        min_v, max_v = min(vals), max(vals)
        spread = max_v - min_v
        scored: list[tuple[str, float]] = []
        if spread == 0:
            # All same: assign 100 if positive else 0
            for sym, v in raw:
                scored.append((sym, 100.0 if v > 0 else 0.0))
        else:
            for sym, v in raw:
                scored.append((sym, (v - min_v) / spread * 100))
        scored.sort(key=lambda x: -x[1])
        # Take top-N filtered by min_score (matches get_top_sectors)
        return [
            sym for sym, sc in scored[: self._cfg.max_sector_etfs]
            if sc >= min_score
        ]

    # ----- Trade execution --------------------------------------------

    def _execute_buy(
        self,
        symbol: str,
        price: float,
        equity: float,
        reason: str,
        d: date,
        positions: dict[str, _ETFPosition],
        cash_box: list[float],  # mutable single-element box
        trades_box: list[int],
    ) -> bool:
        """Buy `symbol` subject to single-ETF cap. Returns True if filled."""
        # Per-ETF cap
        single_cap = equity * self._max_single_pct
        # Portfolio cap (total ETF value)
        total_etf_value = sum(
            p.quantity * self._latest_price(p.symbol, d, p.entry_price)
            for p in positions.values()
        )
        portfolio_cap = equity * self._max_portfolio_pct
        portfolio_room = portfolio_cap - total_etf_value
        if portfolio_room <= 0:
            return False
        target_value = min(single_cap, portfolio_room)
        if target_value <= 0:
            return False
        exec_price = price * (1 + self._cfg.slippage_pct / 100)
        qty = int(target_value / exec_price)
        if qty <= 0:
            return False
        cost = qty * exec_price + self._cfg.commission_per_order
        if cost > cash_box[0]:
            return False
        cash_box[0] -= cost
        positions[symbol] = _ETFPosition(
            symbol=symbol, quantity=qty, entry_price=exec_price,
            entry_date=d, reason=reason,
        )
        trades_box[0] += 1
        return True

    def _execute_sell(
        self,
        symbol: str,
        price: float,
        positions: dict[str, _ETFPosition],
        cash_box: list[float],
        trades_box: list[int],
        reason: str = "",
    ) -> None:
        if symbol not in positions:
            return
        pos = positions[symbol]
        exec_price = price * (1 - self._cfg.slippage_pct / 100)
        proceeds = pos.quantity * exec_price - self._cfg.commission_per_order
        cash_box[0] += proceeds
        del positions[symbol]
        trades_box[0] += 1

    @staticmethod
    def _latest_price(symbol: str, d: date, fallback: float) -> float:
        # Stub for unit-test paths; real call sites pass current price directly.
        return fallback

    # ----- Main loop --------------------------------------------------

    async def run(self) -> ETFBacktestResult:
        prices = self._load_all_prices()
        spy = prices.get("SPY")
        if spy is None or len(spy) < 210:
            raise RuntimeError("SPY history insufficient for backtest")
        vix = prices.get("^VIX")
        sector_symbols = self._etf.get_sector_etf_symbols()

        # Align all series to SPY index for daily iteration
        dates_idx = list(spy.index)

        cash = [self._cfg.initial_capital]
        trades = [0]
        positions: dict[str, _ETFPosition] = {}
        daily_equity: list[float] = []

        last_regime: MarketRegime | None = None
        regime_changes = 0

        # Start after SMA200 warmup
        warmup = 200
        for i in range(warmup, len(dates_idx)):
            d = dates_idx[i].date()

            # ---- 1) Detect regime
            spy_window = spy.iloc[: i + 1].rename("close").to_frame()
            vix_val = float(vix.iloc[i]) if (vix is not None and i < len(vix)) else None
            state = self._detector.detect(spy_window, vix_level=vix_val)
            if last_regime is not None and state.regime != last_regime:
                regime_changes += 1
            last_regime = state.regime

            # ---- 2) Max-hold-days enforcement
            held_to_close = []
            for sym, pos in positions.items():
                days_held = (d - pos.entry_date).days
                if days_held >= self._max_hold_days:
                    held_to_close.append(sym)
            for sym in held_to_close:
                s = prices.get(sym)
                if s is not None and i < len(s):
                    self._execute_sell(sym, float(s.iloc[i]), positions, cash, trades)

            # ---- 3) Regime-based bull/bear rotation
            #    Bear ETF guards (match live ETFEngine):
            #      - SPY must be ≥ bear_min_distance_pct below SMA200
            #      - regime confidence ≥ bear_min_confidence
            is_bear_regime = state.regime in (
                MarketRegime.WEAK_DOWNTREND, MarketRegime.DOWNTREND
            )
            bear_ok = (
                not is_bear_regime
                or (
                    state.spy_distance_pct <= self._cfg.bear_min_distance_pct
                    and state.confidence >= self._cfg.bear_min_confidence
                )
            )
            desired_raw = self._pick_regime_etfs(state.regime)
            desired = set(desired_raw) if bear_ok else set()

            # Sell positions that no longer match the regime (bull/bear leveraged only).
            # Sibling sell-first: when target is bull (eg TQQQ) and we hold bear sibling
            # (eg SQQQ), sell the bear first.
            for sym in list(positions):
                pos = positions[sym]
                if not pos.reason.startswith("regime_"):
                    continue
                if sym in desired:
                    continue
                # Check sibling sell — if any desired symbol is sibling of held, sell held
                siblings_of_held = self._etf.get_pair_siblings(sym)
                must_sell = (
                    sym not in desired
                    or any(d in siblings_of_held for d in desired)
                )
                if must_sell:
                    s = prices.get(sym)
                    if s is not None and i < len(s):
                        self._execute_sell(sym, float(s.iloc[i]), positions, cash, trades)

            # Buy newly desired (after sibling sells freed capacity).
            for sym in desired:
                if sym in positions:
                    continue
                s = prices.get(sym)
                if s is None or i >= len(s):
                    continue
                eq = cash[0] + sum(
                    p.quantity * float(prices[p.symbol].iloc[i])
                    for p in positions.values()
                    if p.symbol in prices and i < len(prices[p.symbol])
                )
                # Determine reason from universe — is this a bear (inverse) ETF?
                is_bear_etf = self._etf.is_leveraged(sym) and is_bear_regime
                reason = "regime_bear" if is_bear_etf else "regime_bull"
                self._execute_buy(sym, float(s.iloc[i]), eq, reason, d,
                                  positions, cash, trades)

            # ---- 4) Sector rotation (top-N by trailing 20d return)
            top_sectors = self._pick_top_sectors(prices, i, sector_symbols)
            # Sell sector positions that fell out of top-N
            for sym in list(positions):
                if positions[sym].reason == "sector" and sym not in top_sectors:
                    s = prices.get(sym)
                    if s is not None and i < len(s):
                        self._execute_sell(sym, float(s.iloc[i]), positions, cash, trades)
            # Buy newly top sectors
            for sym in top_sectors:
                if sym in positions:
                    continue
                s = prices.get(sym)
                if s is None or i >= len(s):
                    continue
                eq = cash[0] + sum(
                    p.quantity * float(prices[p.symbol].iloc[i])
                    for p in positions.values()
                    if p.symbol in prices and i < len(prices[p.symbol])
                )
                self._execute_buy(sym, float(s.iloc[i]), eq, "sector", d,
                                  positions, cash, trades)

            # ---- 5) MTM equity
            eq = cash[0] + sum(
                p.quantity * float(prices[p.symbol].iloc[i])
                for p in positions.values()
                if p.symbol in prices and i < len(prices[p.symbol])
            )
            daily_equity.append(eq)

        # Final results
        final = daily_equity[-1] if daily_equity else self._cfg.initial_capital
        total_ret = (final / self._cfg.initial_capital - 1) * 100

        # MDD
        peak = daily_equity[0] if daily_equity else self._cfg.initial_capital
        mdd = 0.0
        for v in daily_equity:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (v - peak) / peak * 100
                if dd < mdd:
                    mdd = dd

        # Sharpe (daily returns ≥30 sample)
        returns = []
        for j in range(1, len(daily_equity)):
            prev = daily_equity[j - 1]
            if prev > 0:
                returns.append((daily_equity[j] - prev) / prev)
        if len(returns) >= 30:
            mu = sum(returns) / len(returns)
            var = sum((r - mu) ** 2 for r in returns) / max(1, len(returns) - 1)
            std = math.sqrt(var)
            sharpe = (mu / std) * math.sqrt(252) if std > 0 else 0.0
        else:
            sharpe = 0.0

        return ETFBacktestResult(
            total_return_pct=round(total_ret, 2),
            sharpe=round(sharpe, 2),
            max_drawdown_pct=round(mdd, 2),
            trades=trades[0],
            final_value=round(final, 2),
            daily_equity=daily_equity,
            initial_capital=self._cfg.initial_capital,
            regime_changes=regime_changes,
        )
