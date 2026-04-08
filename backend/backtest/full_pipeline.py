"""Full Pipeline Backtest Engine.

Simulates the complete live trading pipeline on historical data:
  1. SPY-based market state detection (regime-aware strategy weights)
  2. Rolling universe screening via IndicatorScreener
  3. Per-stock classification + adaptive weight blending
  4. 14 strategies → SignalCombiner (group consensus)
  5. Kelly position sizing with factor scores
  6. ATR-based dynamic SL/TP per stock
  7. Portfolio-level constraints (max positions, exposure, daily loss)

Usage:
    config = PipelineConfig(universe=["AAPL", "MSFT", ...])
    engine = FullPipelineBacktest(config)
    result = await engine.run(period="3y")
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.data_loader import BacktestDataLoader
from backtest.metrics import MetricsCalculator, BacktestMetrics, Trade
from data.indicator_service import IndicatorService
from data.market_state import MarketStateDetector, MarketRegime
from scanner.indicator_screener import IndicatorScreener
from strategies.registry import StrategyRegistry
from strategies.combiner import SignalCombiner
from strategies.base import Signal
from engine.stock_classifier import StockClassifier
from engine.adaptive_weights import AdaptiveWeightManager
from engine.risk_manager import RiskManager, RiskParams
from analytics.factor_model import MultiFactorModel
from analytics.position_sizing import KellyPositionSizer
from analytics.signal_quality import SignalQualityTracker
from strategies.config_loader import StrategyConfigLoader
from core.enums import SignalType

logger = logging.getLogger(__name__)

# Default narrow universe (S&P 100 subset + growth/value mix). 55 mega-caps.
# Kept as the legacy default for backwards compatibility — covers the symbol
# space the system *historically* used to backtest before live universe
# expansion. Most live trades happen in WIDE_UNIVERSE territory now (small/mid
# cap biotechs, energy, industrials surfaced by yfinance screeners + KIS
# ranking APIs), so for any decision-grade backtest set
# ``PipelineConfig.use_wide_universe=True`` or pass an explicit universe.
DEFAULT_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    # Tech / Semis
    "AMD", "INTC", "QCOM", "AMAT", "MU", "CRM", "ADBE", "ORCL",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "AXP", "C",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK", "TMO", "ABT",
    # Consumer
    "WMT", "HD", "COST", "NKE", "SBUX", "MCD", "PG", "KO",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrial
    "CAT", "BA", "HON", "GE", "UPS", "RTX", "DE",
    # Other
    "DIS", "NFLX", "PYPL", "V", "MA", "BRK-B",
]


# Wider universe (~155 symbols) covering mega-cap + S&P 500 mid-caps + the
# small-cap segments the live UniverseExpander typically surfaces (energy
# E&P, biotech, industrial, utilities, materials). Closer to the actual
# symbol space the live system trades. Use this when backtesting for
# decision-grade analysis — the narrow DEFAULT_UNIVERSE gives misleadingly
# high alpha estimates because it omits the small-cap segment where the
# strategy combiner spends most of its trade volume.
#
# Survivorship caveat: this list is curated from currently-listed tickers,
# so it has the same survivorship bias as any static historical universe
# (delisted names are missing). For production-grade evaluation, prefer
# `universe_path=` pointing to a snapshot taken via
# ``scripts/snapshot_universe.py``.
WIDE_UNIVERSE = sorted(set(DEFAULT_UNIVERSE) | {
    # Tech / Software / Cloud (additional)
    "CSCO", "IBM", "NOW", "INTU", "ADSK", "MRVL", "KLAC", "LRCX",
    "SNOW", "NET", "CRWD", "PANW", "OKTA", "ZS", "DDOG", "MDB", "ZM", "TEAM",
    # Semis additional
    "ASML", "TSM", "ARM", "ON", "MCHP", "STX", "WDC",
    # Finance additional
    "BRK-B", "V", "MA", "COF", "USB", "PNC", "SCHW", "TFC", "BK", "STT",
    # Healthcare / Pharma additional
    "BMY", "AMGN", "GILD", "BIIB", "REGN", "VRTX", "ISRG", "DHR",
    "SYK", "ZTS", "MDT", "BSX", "EW", "BAX",
    # Biotech / small pharma (live system trades these)
    "MRNA", "IONS", "BMRN", "INCY", "NBIX", "JAZZ", "EXEL", "RARE",
    "BPMC", "FOLD", "RGNX", "ARWR", "VKTX", "CRSP", "BEAM", "EDIT",
    "ELVN", "CNTA", "GRDN", "DFTX",
    # Consumer additional
    "PEP", "LOW", "TGT", "DG", "EL", "CL", "MDLZ", "KMB", "GIS", "K",
    "SBUX", "CMG", "YUM", "DPZ", "ULTA", "TJX", "ROST", "BBY",
    # Energy additional
    "OXY", "FANG", "CVE", "CNQ", "MPC", "VLO", "PSX", "DEC", "DVN",
    "HES", "PXD", "MRO", "APA", "EQT", "AR", "BW",
    # Industrial / Defense additional
    "LMT", "NOC", "GD", "TDG", "ETN", "EMR", "PH", "ITW", "ROK",
    "FAST", "PCAR", "CMI", "URI", "ADTN", "IRDM",
    # Materials
    "LIN", "FCX", "NEM", "NUE", "DOW", "DD", "APD", "CSTM", "ALB",
    "VMC", "MLM", "STLD",
    # Comm / Media
    "CMCSA", "T", "VZ", "TMUS", "WBD", "CHTR", "PARA",
    # Transport
    "FDX", "LUV", "DAL", "UAL", "AAL", "EXPD", "JBHT",
    # Utilities (incl ones the system has traded)
    "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "IDA", "ETR", "ED",
    # Real Estate
    "PLD", "AMT", "EQIX", "SPG", "O", "WELL", "PSA",
    # Sector ETFs (so dual_momentum / sector_rotation can pick them in backtest)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC",
})

# Default KR universe (KOSPI/KOSDAQ major stocks, yfinance format)
DEFAULT_KR_UNIVERSE = [
    # KOSPI 대형주
    "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005935.KS",
    "006400.KS", "051910.KS", "005380.KS", "000270.KS",
    # IT/플랫폼
    "035420.KS", "035720.KS",
    # 바이오/헬스
    "068270.KS", "207940.KS",
    # 금융
    "105560.KS", "055550.KS", "086790.KS", "032830.KS",
    # 자동차/부품
    "012330.KS",
    # 전자/소재
    "066570.KS", "003550.KS", "009150.KS", "003670.KS",
    # 에너지/산업
    "034020.KS", "015760.KS", "010130.KS", "011200.KS",
    # 소비/통신
    "033780.KS", "017670.KS", "028260.KS", "018260.KS",
    # KOSDAQ
    "247540.KQ", "086520.KQ", "263750.KQ", "196170.KQ",
    "403870.KQ", "352820.KQ", "058470.KQ", "357780.KQ",
    "036930.KQ", "039030.KQ", "145020.KQ",
]


def _default_universe_for_market(market: str, wide: bool = False) -> list[str]:
    """Return the default universe for the given market.

    Args:
        market: "US" or "KR".
        wide: When True, returns the wider US universe (~155 symbols
            covering mega + mid + small caps) closer to the live
            UniverseExpander output. KR currently has only one universe.
    """
    if market == "KR":
        return list(DEFAULT_KR_UNIVERSE)
    return list(WIDE_UNIVERSE if wide else DEFAULT_UNIVERSE)


@dataclass
class PipelineConfig:
    """Configuration for full pipeline backtest."""
    # Market
    market: str = "US"  # "US" or "KR"

    # Universe
    # Resolution priority (in __init__): explicit `universe` list >
    # `universe_path` (JSON snapshot) > wide/default constant.
    universe: list[str] | None = None  # None = auto from market
    regime_symbol: str | None = None  # None = auto (SPY or 069500.KS)
    use_wide_universe: bool = False  # US: True → WIDE_UNIVERSE (~155 symbols)
    universe_path: str | None = None  # JSON file with {"symbols": [...]}

    # Live state injection — load a previously-snapshotted SignalQualityTracker
    # so the backtest starts from accumulated live history rather than a
    # cold tracker. Generate via scripts/snapshot_signal_quality.py.
    # Without this, dynamic strategy gating + Kelly inputs in the backtest
    # diverge from live behavior because the tracker has no trade history.
    signal_quality_seed_path: str | None = None

    # Simulation
    # initial_equity uses raw price units (no FX conversion). For US (USD prices)
    # the default is 100_000 = $100k. For KR (KRW prices, e.g. 470,000원/share),
    # leaving this at 100_000 makes int(allocation/price) = 0 → 0 trades silently.
    # When None, FullPipelineBacktest auto-scales: US → 100_000, KR → 100_000_000.
    initial_equity: float | None = None
    slippage_pct: float = 0.05  # 0.05%
    commission_per_order: float = 0.0

    volume_adjusted_slippage: bool = True  # Scale slippage by participation rate

    # Screening
    screen_interval: int = 20  # Re-screen every N trading days
    max_watchlist: int = 30  # Max symbols in active watchlist
    min_screen_grade: str = "B"

    # Risk (mirrors RiskParams)
    max_position_pct: float = 0.10
    max_positions: int = 20
    max_exposure_pct: float = 0.90
    daily_loss_limit_pct: float = 0.03

    # SL/TP
    dynamic_sl_tp: bool = True
    default_stop_loss_pct: float = 0.08
    default_take_profit_pct: float = 0.20
    trailing_activation_pct: float = 0.05
    trailing_trail_pct: float = 0.03

    # Protective sells
    enable_regime_sells: bool = True  # Sell losing positions on regime deterioration

    # Strategy sell pairing
    paired_strategy_sells: bool = False  # Only sell when buy strategy votes SELL

    # Signal combiner
    min_active_ratio: float = 0.15  # Min fraction of strategies that must be active
    min_confidence: float = 0.50  # Min combined confidence to generate BUY/SELL

    # Kelly sizing (match live defaults from position_sizing.py)
    kelly_fraction: float = 0.40  # Fractional Kelly (matches live)
    confidence_exponent: float = 1.2  # Confidence scaling power (matches live)
    min_position_pct: float = 0.05  # Minimum position size (matches live)

    # Momentum factor tilt
    enable_momentum_tilt: bool = False  # Pass momentum z-scores to Kelly sizer
    momentum_update_interval: int = 20  # Recompute momentum every N days

    # Strategy quality amplification
    enable_quality_amplification: bool = False  # Boost weights of winning strategies
    quality_blend_alpha: float = 0.3  # 30% quality-based, 70% original weights
    min_trades_for_quality: int = 30  # Min trades before quality weights activate

    # Strategy gating
    enable_strategy_gating: bool = False  # Disable strategies with no edge

    # Re-entry after stop loss
    recovery_watch_days: int = 7  # Keep stopped-out symbols in eval set for N days

    # Trading gates (matching live evaluation_loop behavior)
    sell_cooldown_days: int = 1  # Block re-buy for N days after sell (0=disabled)
    whipsaw_max_losses: int = 2  # Block re-buy after N loss sells in 7 days
    min_hold_days: int = 1  # Minimum holding period in trading days
    hard_sl_pct: float = 0.15  # Hard SL bypasses min hold (e.g. -15%)

    # Held position evaluation (matching live STOCK-7 / STOCK-47)
    # Disabled by default — A/B testing showed these hurt KR returns
    held_sell_bias: float = 0.0  # Boost sell signals for held positions (0=disabled)
    held_min_confidence: float = 0.50  # Lower threshold for held exits (=min_confidence → no effect)
    stale_pnl_threshold: float = 0.0  # Sell on indifference below this PnL (<=0 disables)
    profit_protection_pct: float = 0.0  # Secure gains above this PnL (0=disabled)

    # Extended hours simulation (backtest-optimized)
    extended_hours_enabled: bool = False
    extended_hours_max_position_pct: float = 0.05  # 5% per position
    extended_hours_slippage_multiplier: float = 2.0  # 2x regular slippage
    extended_hours_fill_probability: float = 0.90  # 10% miss rate
    extended_hours_min_confidence: float = 0.55  # Lower bar OK: dip/spillover only

    # Daily buy limit + confidence escalation
    daily_buy_limit: int = 0  # 0 = unlimited
    enable_confidence_escalation: bool = False  # Dynamic confidence escalation

    # Strategy selection
    disabled_strategies: list[str] = field(default_factory=list)  # Strategies to skip

    # Cash parking: invest idle cash in SPY
    enable_cash_parking: bool = False   # Park idle cash in SPY
    cash_parking_symbol: str = "SPY"
    cash_parking_threshold: float = 0.30  # Park if cash > 30% of portfolio

    # Leveraged ETF allocation
    enable_leveraged_etf: bool = False
    etf_symbol: str = "TQQQ"           # Leveraged ETF to trade
    etf_inverse_symbol: str = "SQQQ"   # Inverse ETF for downtrend
    etf_max_allocation_pct: float = 0.20  # Max portfolio % in leveraged ETF
    etf_uptrend_regimes: list[str] = field(
        default_factory=lambda: ["strong_uptrend", "uptrend"],
    )

    # Sector concentration limit
    max_sector_pct: float = 1.0  # Max portfolio % in any single sector (1.0 = no cap)

    # Strategy config path
    strategy_config_path: str | None = None


@dataclass
class _Position:
    """Internal position tracking for backtest."""
    symbol: str
    quantity: int
    avg_price: float
    entry_date: str
    strategy_name: str
    highest_price: float
    stop_loss_pct: float
    take_profit_pct: float
    session: str = "regular"  # "regular" or "extended"
    entry_day_count: int = 0  # day_count at entry (for min hold check)
    sector: str = ""  # GICS sector for concentration tracking


@dataclass
class DailySnapshot:
    """Daily portfolio state for logging."""
    date: str
    equity: float
    cash: float
    n_positions: int
    regime: str
    n_watchlist: int


@dataclass
class PipelineResult:
    """Result of full pipeline backtest."""
    metrics: BacktestMetrics
    trades: list[Trade]
    equity_curve: pd.Series
    daily_snapshots: list[DailySnapshot]
    config: PipelineConfig
    strategy_stats: dict[str, dict]  # per-strategy trade counts

    def summary(self) -> str:
        m = self.metrics
        status = "PASS" if m.passes_minimum() else "FAIL"
        lines = [
            f"[{status}] Full Pipeline Backtest",
            f"  Period: {m.start_date} ~ {m.end_date} ({m.trading_days} days)",
            f"  Return: {m.total_return_pct:.1f}% | CAGR: {m.cagr:.1%}",
            f"  Sharpe: {m.sharpe_ratio:.2f} | Sortino: {m.sortino_ratio:.2f}",
            f"  MDD: {m.max_drawdown_pct:.1f}% ({m.max_drawdown_days} days)",
            f"  Trades: {m.total_trades} | Win Rate: {m.win_rate:.1f}%",
            f"  Profit Factor: {m.profit_factor:.2f}",
            f"  Avg Hold: {m.avg_holding_days:.0f} days",
            f"  Final Equity: ${m.final_equity:,.0f}",
            f"  Benchmark (SPY): {m.benchmark_return_pct:.1f}% | Alpha: {m.alpha:.1f}%",
        ]
        if m.extended_trades > 0:
            lines.append(
                f"  Extended Hours: {m.extended_trades} trades, "
                f"WR={m.extended_win_rate:.0f}%, "
                f"PnL=${m.extended_pnl:+,.0f}"
            )
        if self.strategy_stats:
            lines.append("  Strategy breakdown:")
            for name, stats in sorted(
                self.strategy_stats.items(),
                key=lambda x: x[1]["trades"], reverse=True,
            ):
                if stats["trades"] > 0:
                    lines.append(
                        f"    {name}: {stats['trades']} trades, "
                        f"WR={stats['win_rate']:.0f}%, "
                        f"PnL=${stats['pnl']:+,.0f}"
                    )
        return "\n".join(lines)


class FullPipelineBacktest:
    """Simulates the complete live trading pipeline on historical data."""

    def __init__(self, config: PipelineConfig | None = None):
        self._config = config or PipelineConfig()

        # Resolve market-dependent defaults
        cfg = self._config
        if cfg.universe is None:
            cfg.universe = self._resolve_default_universe(cfg)
        if cfg.regime_symbol is None:
            cfg.regime_symbol = "069500.KS" if cfg.market == "KR" else "SPY"
        if cfg.initial_equity is None:
            # Auto-scale equity to match the price units of the target market.
            # KR stock prices are in KRW (often 100k+ per share); a USD-style
            # 100_000 default makes int(allocation/price) = 0 and silently
            # produces 0 trades.
            cfg.initial_equity = 100_000_000.0 if cfg.market == "KR" else 100_000.0

        # Will be populated below; load signal-quality seed (if any) after
        # the tracker is created.
        self._data_loader = BacktestDataLoader()
        self._indicator_svc = IndicatorService()
        self._screener = IndicatorScreener(min_grade=self._config.min_screen_grade)
        self._market_state_detector = MarketStateDetector()
        self._classifier = StockClassifier()
        self._adaptive = AdaptiveWeightManager()
        self._factor_model = MultiFactorModel()
        self._signal_quality = SignalQualityTracker()
        self._load_signal_quality_seed(cfg)
        self._risk_manager = RiskManager(RiskParams(
            max_position_pct=self._config.max_position_pct,
            max_total_exposure_pct=self._config.max_exposure_pct,
            max_positions=self._config.max_positions,
            daily_loss_limit_pct=self._config.daily_loss_limit_pct,
            default_stop_loss_pct=self._config.default_stop_loss_pct,
            default_take_profit_pct=self._config.default_take_profit_pct,
        ))
        # Override Kelly sizer params for backtest tuning
        self._risk_manager._kelly = KellyPositionSizer(
            kelly_fraction=self._config.kelly_fraction,
            max_position_pct=self._config.max_position_pct,
            min_position_pct=self._config.min_position_pct,
            confidence_exponent=self._config.confidence_exponent,
        )

        # Initialize strategy registry and combiner
        config_loader = StrategyConfigLoader(self._config.strategy_config_path)
        self._registry = StrategyRegistry(config_loader=config_loader)
        consensus_cfg = config_loader.get_consensus_config()
        self._combiner = SignalCombiner(
            consensus_config=consensus_cfg,
            min_active_ratio=self._config.min_active_ratio,
        )

        # Portfolio state
        self._cash: float = 0.0
        self._positions: dict[str, _Position] = {}
        self._trades: list[Trade] = []
        self._equity_curve: list[float] = []
        self._equity_dates: list = []
        self._daily_snapshots: list[DailySnapshot] = []
        self._watchlist: list[str] = []
        self._prev_regime: str = "uptrend"

        # Momentum factor scores (updated periodically)
        self._factor_scores: dict[str, float] = {}  # symbol → composite z-score
        self._last_factor_update: int = -9999
        self._gated_strategies: set[str] = set()

        # Recovery watch: recently sold symbols stay in eval set
        # {symbol: day_count_when_sold}
        self._recovery_watch: dict[str, int] = {}
        self._day_count: int = 0

        # Trading gates (matching live evaluation_loop)
        self._sell_cooldown: dict[str, int] = {}  # symbol → day_count when sold
        self._loss_sell_history: dict[str, list[int]] = {}  # symbol → [day_counts]

        # Daily buy limit tracking
        self._daily_buy_count: int = 0
        self._daily_buy_date: str = ""

        # Sector concentration tracking
        self._sector_cache: dict[str, str] = {}  # symbol → sector

    def _load_signal_quality_seed(self, cfg: "PipelineConfig") -> None:
        """Optionally seed self._signal_quality from a JSON snapshot.

        Snapshot is generated by ``scripts/snapshot_signal_quality.py``
        which dumps the live tracker state (or rebuilds it from the trades
        DB table). Without seeding, the backtest's gating + Kelly inputs
        diverge from live behavior.

        Accepts two snapshot shapes:
            1. Bare tracker dict: ``{"version": 1, "trades": {...}}``
            2. Wrapper with metadata: ``{"version": 1, "tracker": {...}}``

        Failures are logged and the backtest continues with a cold tracker.
        """
        if not cfg.signal_quality_seed_path:
            return
        try:
            import json as _json
            from pathlib import Path as _Path

            path = _Path(cfg.signal_quality_seed_path)
            payload = _json.loads(path.read_text())
            # Unwrap the optional metadata layer
            if isinstance(payload, dict) and isinstance(payload.get("tracker"), dict):
                tracker_payload = payload["tracker"]
            else:
                tracker_payload = payload
            n = self._signal_quality.load_dict(tracker_payload)
            logger.info(
                "Seeded signal_quality with %d trade records from %s",
                n, cfg.signal_quality_seed_path,
            )
        except Exception as e:
            logger.warning(
                "Failed to load signal_quality seed %s: %s — using cold tracker",
                cfg.signal_quality_seed_path, e,
            )

    @staticmethod
    def _resolve_default_universe(cfg: "PipelineConfig") -> list[str]:
        """Resolve the universe when none was passed explicitly.

        Order: universe_path (JSON snapshot) > wide constant > narrow default.
        """
        if cfg.universe_path:
            try:
                import json as _json
                from pathlib import Path as _Path

                p = _Path(cfg.universe_path)
                payload = _json.loads(p.read_text())
                symbols = payload.get("symbols") if isinstance(payload, dict) else payload
                if isinstance(symbols, list) and symbols:
                    logger.info(
                        "Loaded %d symbols from universe snapshot %s",
                        len(symbols), cfg.universe_path,
                    )
                    return [str(s).upper() for s in symbols]
                logger.warning(
                    "Universe snapshot %s has no 'symbols' list — falling back",
                    cfg.universe_path,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load universe snapshot %s: %s — falling back",
                    cfg.universe_path, e,
                )
        return _default_universe_for_market(cfg.market, wide=cfg.use_wide_universe)

    async def run(
        self,
        period: str = "3y",
        start: str | None = None,
        end: str | None = None,
    ) -> PipelineResult:
        """Run the full pipeline backtest.

        Args:
            period: Data period (e.g. '3y', '5y')
            start: Start date YYYY-MM-DD (overrides period)
            end: End date YYYY-MM-DD

        Returns:
            PipelineResult with metrics, trades, equity curve
        """
        cfg = self._config

        # 1. Load all data upfront
        regime_sym = cfg.regime_symbol
        logger.info(
            "Loading data for %d universe symbols + %s (%s market)...",
            len(cfg.universe), regime_sym, cfg.market,
        )
        # Include leveraged ETF symbols in data load
        etf_symbols = []
        if cfg.enable_leveraged_etf:
            etf_symbols = [cfg.etf_symbol, cfg.etf_inverse_symbol]
        all_symbols = list(dict.fromkeys(
            [regime_sym] + cfg.universe + etf_symbols
        ))
        all_data = self._data_loader.load_multiple(
            all_symbols, period=period,
        )

        if regime_sym not in all_data:
            raise ValueError(
                f"Failed to load {regime_sym} data (required for market state)"
            )

        spy_data = all_data[regime_sym]
        stock_data = {s: d for s, d in all_data.items() if s != regime_sym}

        if not stock_data:
            raise ValueError("No stock data loaded")

        logger.info(
            "Data loaded: %d stocks, %d ~ %d bars",
            len(stock_data),
            min(len(d.df) for d in stock_data.values()),
            max(len(d.df) for d in stock_data.values()),
        )

        # 2. Find common date range
        spy_dates = spy_data.df.index
        common_start = spy_dates[250]  # Need 200+ bars for SMA200 + warmup
        common_end = spy_dates[-1]

        # 3. Initialize state
        self._cash = cfg.initial_equity
        self._positions.clear()
        self._trades.clear()
        self._equity_curve.clear()
        self._equity_dates.clear()
        self._daily_snapshots.clear()
        self._watchlist.clear()
        self._risk_manager.reset_daily()
        self._factor_scores.clear()
        self._last_factor_update = -9999
        self._gated_strategies.clear()
        self._recovery_watch.clear()
        self._daily_buy_count = 0
        self._daily_buy_date = ""

        # Pre-classify all stocks once
        for symbol, data in stock_data.items():
            if len(data.df) >= 60:
                profile = self._classifier.classify(data.df, symbol)
                self._adaptive.set_category(symbol, profile.category)

        # 4. Day-by-day simulation
        day_count = 0
        self._day_count = 0
        last_screen_day = -cfg.screen_interval  # Force screen on first day

        for date_idx in range(len(spy_dates)):
            date = spy_dates[date_idx]
            if date < common_start:
                continue

            day_count += 1
            self._day_count = day_count

            # Daily reset of PnL tracking
            if day_count % 1 == 0:
                # Actually reset only on real new days, but for daily bars this is every bar
                pass  # We reset in _end_of_day

            # 4a. Detect market state from SPY (exclude current bar to avoid look-ahead)
            spy_window = spy_data.df.iloc[:date_idx]
            market_state = self._market_state_detector.detect(spy_window)
            regime_str = market_state.regime.value
            self._risk_manager.set_eval_regime(regime_str)

            # Map regime to profile name
            profile_name = regime_str
            if profile_name == "strong_uptrend":
                profile_name = "strong_uptrend"

            # 4b. Screening: refresh watchlist periodically
            if day_count - last_screen_day >= cfg.screen_interval:
                self._watchlist = self._screen_universe(
                    stock_data, date_idx, cfg.max_watchlist,
                )
                last_screen_day = day_count
                logger.debug(
                    "Day %d: Screened %d → watchlist %d symbols",
                    day_count, len(stock_data), len(self._watchlist),
                )

            # 4b2. Update momentum factor scores periodically
            if cfg.enable_momentum_tilt:
                if day_count - self._last_factor_update >= cfg.momentum_update_interval:
                    self._update_factor_scores(stock_data, date_idx)
                    self._last_factor_update = day_count

            # 4b3. Update strategy gating
            if cfg.enable_strategy_gating:
                self._gated_strategies = set(
                    self._signal_quality.get_gated_strategies()
                )

            # Expire old recovery watch entries
            expired = [
                s for s, sold_day in self._recovery_watch.items()
                if day_count - sold_day > cfg.recovery_watch_days
            ]
            for s in expired:
                del self._recovery_watch[s]

            # Merge watchlist + held positions + recovery watch
            held = set(self._positions.keys())
            recovery = set(self._recovery_watch.keys()) - held
            eval_symbols = list(dict.fromkeys(
                self._watchlist + sorted(held) + sorted(recovery)
            ))

            # 4c. Regime-change protective sells
            if cfg.enable_regime_sells and self._positions:
                self._check_regime_sells(
                    stock_data, date_idx, date, regime_str,
                )
            self._prev_regime = regime_str

            # 4d. Extended hours: check SL at open price (gap-down defense)
            if cfg.extended_hours_enabled:
                self._check_extended_hours_sl(stock_data, date_idx, date)

            # 4d2. Check SL/TP/trailing stop on existing positions (regular)
            self._check_risk_exits(stock_data, date_idx, date)

            # 4e. Evaluate signals and execute
            buy_candidates: list[tuple[float, str, Signal]] = []

            for symbol in eval_symbols:
                if symbol not in stock_data:
                    continue
                sdata = stock_data[symbol]
                if date_idx >= len(sdata.df):
                    continue

                df_window = sdata.df.iloc[:date_idx]  # Exclude current bar (no look-ahead)
                if len(df_window) < 50:
                    continue

                # Run all strategies (skip disabled ones)
                strategies = self._registry.get_enabled()
                disabled = set(cfg.disabled_strategies)
                signals = []
                for strategy in strategies:
                    if strategy.name in disabled:
                        continue
                    try:
                        signal = await strategy.analyze(df_window, symbol)
                        signals.append(signal)
                    except Exception as e:
                        logger.debug("Strategy %s failed for %s: %s", strategy.name, symbol, e)

                # Get weights: market-state profile + stock category blending
                market_weights = self._registry.get_profile_weights(profile_name)
                weights = self._adaptive.get_weights(symbol, market_weights)

                # Amplify weights of winning strategies
                if cfg.enable_quality_amplification:
                    weights = self._get_quality_adjusted_weights(weights)

                # Soft-gate losing strategies (halve their weight)
                if cfg.enable_strategy_gating and self._gated_strategies:
                    for gated in self._gated_strategies:
                        if gated in weights:
                            weights[gated] *= 0.5

                # Combine signals — different thresholds for held vs new
                is_held = symbol in self._positions
                if is_held:
                    combined = self._combiner.combine(
                        signals, weights,
                        min_confidence=cfg.held_min_confidence,
                        min_active_ratio=cfg.min_active_ratio,
                        held_sell_bias=cfg.held_sell_bias,
                    )
                else:
                    combined = self._combiner.combine(
                        signals, weights, min_confidence=cfg.min_confidence,
                    )

                # Sell on indifference: losing held position + no strategy signal
                if (
                    is_held
                    and combined.signal_type == SignalType.HOLD
                    and cfg.stale_pnl_threshold < 0
                ):
                    pos = self._positions[symbol]
                    data = stock_data[symbol]
                    if date_idx < len(data.df):
                        cur_price = float(data.df.iloc[date_idx]["close"])
                        pnl_pct = (cur_price - pos.avg_price) / pos.avg_price
                        if pnl_pct < cfg.stale_pnl_threshold:
                            combined = Signal(
                                signal_type=SignalType.SELL,
                                confidence=0.50,
                                strategy_name="position_cleanup",
                                reason=f"Sell on indifference: P&L={pnl_pct:.1%}",
                            )

                # Profit protection: secure gains above threshold
                if (
                    is_held
                    and combined.signal_type == SignalType.HOLD
                    and cfg.profit_protection_pct > 0
                ):
                    pos = self._positions[symbol]
                    data = stock_data[symbol]
                    if date_idx < len(data.df):
                        cur_price = float(data.df.iloc[date_idx]["close"])
                        pnl_pct = (cur_price - pos.avg_price) / pos.avg_price
                        if pnl_pct >= cfg.profit_protection_pct:
                            combined = Signal(
                                signal_type=SignalType.SELL,
                                confidence=0.60,
                                strategy_name="profit_protection",
                                reason=f"Profit protection: P&L={pnl_pct:.1%}",
                            )

                # Execute SELLs immediately
                if combined.signal_type == SignalType.SELL:
                    if cfg.paired_strategy_sells and is_held:
                        # Paired mode: only sell if the BUY strategy votes SELL
                        buy_strat = self._positions[symbol].strategy_name
                        buy_strat_sells = any(
                            s.signal_type == SignalType.SELL
                            and s.strategy_name == buy_strat
                            for s in signals
                        )
                        if buy_strat_sells:
                            self._execute_sell(
                                symbol, stock_data, date_idx, date, combined,
                            )
                    else:
                        self._execute_sell(
                            symbol, stock_data, date_idx, date, combined,
                        )
                elif combined.signal_type == SignalType.BUY and not is_held:
                    # Only buy if not already held (BUY→HOLD remapping)
                    buy_candidates.append((combined.confidence, symbol, combined))

            # Execute BUYs ranked by confidence (highest first)
            if buy_candidates:
                buy_candidates.sort(key=lambda x: x[0], reverse=True)
                # Sell SPY parking to free cash for stock buys
                parking_sym = cfg.cash_parking_symbol
                if cfg.enable_cash_parking and parking_sym in self._positions:
                    data = stock_data.get(parking_sym)
                    if data and date_idx < len(data.df):
                        _row = data.df.iloc[date_idx]
                        price = float(_row["close"])
                        _vol = float(_row["volume"]) if "volume" in _row.index else 0.0
                        self._close_position(
                            parking_sym, price, date, "cash_unpark", volume=_vol,
                        )

                # Reset daily buy count on new day
                date_str = str(date)
                if date_str != self._daily_buy_date:
                    self._daily_buy_count = 0
                    self._daily_buy_date = date_str

                for _conf, symbol, combined in buy_candidates:
                    # Apply daily buy limit + confidence escalation
                    if not self._check_daily_buy_allowed(combined.confidence):
                        continue
                    held_before = symbol in self._positions
                    self._execute_buy(
                        symbol, stock_data, date_idx, date, combined,
                        market_state.regime,
                    )
                    if not held_before and symbol in self._positions:
                        self._daily_buy_count += 1

            # 4e2. Extended hours buy: high-confidence signals at open price
            if cfg.extended_hours_enabled and buy_candidates:
                self._execute_extended_hours_buys(
                    buy_candidates, stock_data, date_idx, date, market_state.regime,
                )

            # 4e3. Leveraged ETF regime allocation
            if cfg.enable_leveraged_etf:
                self._manage_leveraged_etf(
                    stock_data, date_idx, date, regime_str,
                )

            # 4e4. Cash parking — invest idle cash in SPY
            if cfg.enable_cash_parking:
                self._manage_cash_parking(stock_data, date_idx, date)

            # 4f. Update equity and snapshot
            equity = self._calculate_equity(stock_data, date_idx)
            self._equity_curve.append(equity)
            self._equity_dates.append(date)
            self._daily_snapshots.append(DailySnapshot(
                date=str(date),
                equity=round(equity, 2),
                cash=round(self._cash, 2),
                n_positions=len(self._positions),
                regime=regime_str,
                n_watchlist=len(self._watchlist),
            ))

            # End of day: reset daily PnL
            self._risk_manager.reset_daily()

        # 5. Close all remaining positions at last price
        self._close_all_positions(stock_data, len(spy_dates) - 1, spy_dates[-1])

        # 6. Calculate metrics
        equity_series = pd.Series(self._equity_curve, index=self._equity_dates)
        spy_returns = spy_data.df["close"].pct_change().dropna()
        # Align benchmark returns to our simulation period
        spy_returns = spy_returns.loc[
            (spy_returns.index >= common_start) & (spy_returns.index <= common_end)
        ]

        metrics = MetricsCalculator.calculate(
            equity_curve=equity_series,
            trades=self._trades,
            initial_equity=cfg.initial_equity,
            benchmark_returns=spy_returns,
        )

        # 7. Strategy breakdown
        strategy_stats = self._compute_strategy_stats()

        result = PipelineResult(
            metrics=metrics,
            trades=self._trades,
            equity_curve=equity_series,
            daily_snapshots=self._daily_snapshots,
            config=self._config,
            strategy_stats=strategy_stats,
        )

        logger.info("\n%s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Cash parking (invest idle cash in SPY)
    # ------------------------------------------------------------------

    def _manage_cash_parking(
        self, stock_data: dict, date_idx: int, date,
    ) -> None:
        """Park excess cash in SPY to reduce cash drag.

        When cash exceeds threshold, buy SPY with the excess.
        SPY is sold before stock buys to free up cash.
        """
        cfg = self._config
        parking_sym = cfg.cash_parking_symbol

        # Skip if already holding parking position
        if parking_sym in self._positions:
            return

        data = stock_data.get(parking_sym)
        if not data or date_idx >= len(data.df):
            return

        equity = self._calculate_equity(stock_data, date_idx)
        cash_pct = self._cash / equity if equity > 0 else 0

        # Only park if cash exceeds threshold
        if cash_pct < cfg.cash_parking_threshold:
            return

        # Park the excess cash (keep some buffer for opportunities)
        park_amount = self._cash - equity * 0.10  # Keep 10% cash buffer
        if park_amount <= 0:
            return

        price = float(data.df.iloc[date_idx]["close"])
        if price <= 0:
            return

        exec_price = price * (1 + cfg.slippage_pct / 100)
        quantity = int(park_amount / exec_price)
        if quantity <= 0:
            return

        cost = quantity * exec_price + cfg.commission_per_order
        if cost > self._cash:
            return

        self._cash -= cost
        self._positions[parking_sym] = _Position(
            symbol=parking_sym,
            quantity=quantity,
            avg_price=exec_price,
            entry_date=str(date),
            strategy_name="cash_parking",
            highest_price=exec_price,
            stop_loss_pct=9.99,   # no SL for parking
            take_profit_pct=9.99, # no TP for parking
        )

    # ------------------------------------------------------------------
    # Leveraged ETF management
    # ------------------------------------------------------------------

    def _manage_leveraged_etf(
        self,
        stock_data: dict,
        date_idx: int,
        date,
        regime: str,
    ) -> None:
        """Buy/sell leveraged ETF based on market regime.

        Uptrend → hold TQQQ (up to etf_max_allocation_pct)
        Downtrend → sell TQQQ, optionally hold SQQQ
        Sideways → sell both (cash)
        """
        cfg = self._config
        etf_long = cfg.etf_symbol
        etf_short = cfg.etf_inverse_symbol
        bullish = regime in cfg.etf_uptrend_regimes

        # Sell inverse ETF if holding in non-downtrend
        if etf_short in self._positions and regime not in ("downtrend", "weak_downtrend"):
            data = stock_data.get(etf_short)
            if data and date_idx < len(data.df):
                price = float(data.df.iloc[date_idx]["close"])
                self._close_position(etf_short, price, date, "etf_regime_exit")

        # Sell long ETF if regime turns bearish/sideways
        if etf_long in self._positions and not bullish:
            data = stock_data.get(etf_long)
            if data and date_idx < len(data.df):
                price = float(data.df.iloc[date_idx]["close"])
                self._close_position(etf_long, price, date, "etf_regime_exit")

        # Buy long ETF in uptrend if not already holding
        if bullish and etf_long not in self._positions:
            data = stock_data.get(etf_long)
            if data and date_idx < len(data.df):
                price = float(data.df.iloc[date_idx]["close"])
                if price <= 0:
                    return
                equity = self._calculate_equity(stock_data, date_idx)
                allocation = equity * cfg.etf_max_allocation_pct
                allocation = min(allocation, self._cash * 0.95)
                if allocation > 0:
                    exec_price = price * (1 + cfg.slippage_pct / 100)
                    quantity = int(allocation / exec_price)
                    if quantity > 0:
                        cost = quantity * exec_price + cfg.commission_per_order
                        if cost <= self._cash:
                            self._cash -= cost
                            self._positions[etf_long] = _Position(
                                symbol=etf_long,
                                quantity=quantity,
                                avg_price=exec_price,
                                entry_date=str(date),
                                strategy_name="etf_leverage",
                                highest_price=exec_price,
                                stop_loss_pct=0.15,  # wider SL for leveraged
                                take_profit_pct=1.00, # let it run
                            )

        # Buy inverse ETF in downtrend if not already holding
        if regime in ("downtrend", "weak_downtrend") and etf_short not in self._positions:
            data = stock_data.get(etf_short)
            if data and date_idx < len(data.df):
                price = float(data.df.iloc[date_idx]["close"])
                if price <= 0:
                    return
                equity = self._calculate_equity(stock_data, date_idx)
                # Smaller allocation for inverse (hedge)
                allocation = equity * cfg.etf_max_allocation_pct * 0.5
                allocation = min(allocation, self._cash * 0.95)
                if allocation > 0:
                    exec_price = price * (1 + cfg.slippage_pct / 100)
                    quantity = int(allocation / exec_price)
                    if quantity > 0:
                        cost = quantity * exec_price + cfg.commission_per_order
                        if cost <= self._cash:
                            self._cash -= cost
                            self._positions[etf_short] = _Position(
                                symbol=etf_short,
                                quantity=quantity,
                                avg_price=exec_price,
                                entry_date=str(date),
                                strategy_name="etf_inverse",
                                highest_price=exec_price,
                                stop_loss_pct=0.15,
                                take_profit_pct=0.30,
                            )

    # ------------------------------------------------------------------
    # Momentum factor scoring
    # ------------------------------------------------------------------

    def _update_factor_scores(
        self, stock_data: dict, date_idx: int,
    ) -> None:
        """Compute cross-sectional momentum z-scores for all stocks."""
        price_data = {}
        for symbol, data in stock_data.items():
            if date_idx < len(data.df):
                price_data[symbol] = data.df.iloc[:date_idx]  # exclude current bar

        if len(price_data) < 3:
            return

        # Use factor model which computes momentum + z-scores
        scores = self._factor_model.score_universe(price_data)
        self._factor_scores = {s.symbol: s.composite for s in scores}

    def _get_quality_adjusted_weights(
        self, base_weights: dict[str, float],
    ) -> dict[str, float]:
        """Blend base weights with signal quality performance weights."""
        cfg = self._config
        quality_weights = self._signal_quality.get_strategy_weights()
        if not quality_weights:
            return base_weights

        # Check if enough trades have been recorded
        total_trades = sum(
            self._signal_quality.get_metrics(name).total_trades
            for name in quality_weights
        )
        if total_trades < cfg.min_trades_for_quality:
            return base_weights

        # Blend: (1-alpha)*base + alpha*quality
        alpha = cfg.quality_blend_alpha
        blended = {}
        all_keys = set(base_weights) | set(quality_weights)
        for key in all_keys:
            base = base_weights.get(key, 0.0)
            qual = quality_weights.get(key, 0.0)
            blended[key] = (1 - alpha) * base + alpha * qual

        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    # ------------------------------------------------------------------
    # Screening
    # ------------------------------------------------------------------

    def _screen_universe(
        self,
        stock_data: dict[str, object],
        date_idx: int,
        max_symbols: int,
    ) -> list[str]:
        """Screen universe and return top symbols by indicator score."""
        scores = []
        for symbol, data in stock_data.items():
            if date_idx >= len(data.df):
                continue
            df_window = data.df.iloc[:date_idx]  # Exclude current bar
            if len(df_window) < 50:
                continue
            try:
                score = self._screener.score(df_window, symbol)
                scores.append(score)
            except Exception as e:
                logger.debug("Screener scoring failed for %s: %s", symbol, e)

        filtered = self._screener.filter_candidates(
            scores, max_candidates=max_symbols,
        )
        return [s.symbol for s in filtered]

    # ------------------------------------------------------------------
    # Risk exits (SL/TP/trailing stop)
    # ------------------------------------------------------------------

    def _check_risk_exits(
        self, stock_data: dict[str, object], date_idx: int, date,
    ) -> None:
        """Check SL/TP/trailing stop on all held positions."""
        # Skip cash parking and system-managed positions
        skip_symbols = {"cash_parking", "etf_leverage", "etf_inverse"}
        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            if pos.strategy_name in skip_symbols:
                continue
            if symbol not in stock_data:
                continue
            data = stock_data[symbol]
            if date_idx >= len(data.df):
                continue

            row = data.df.iloc[date_idx]
            pos = self._positions[symbol]
            price = float(row["close"])
            high = float(row["high"]) if "high" in row.index else price
            low = float(row["low"]) if "low" in row.index else price
            open_price = float(row["open"]) if "open" in row.index else price

            # Update highest price for trailing stop
            if high > pos.highest_price:
                pos.highest_price = high

            # Min hold check: skip non-hard-SL exits if held too briefly
            held_days = self._day_count - pos.entry_day_count
            min_hold = self._config.min_hold_days
            unrealized_pct = (low - pos.avg_price) / pos.avg_price
            is_hard_sl = unrealized_pct <= -self._config.hard_sl_pct

            # Stop-loss (gap-through: open below SL → fill at open)
            sl_price = pos.avg_price * (1 - pos.stop_loss_pct)
            if low <= sl_price:
                if held_days < min_hold and not is_hard_sl:
                    continue  # Min hold not met, not hard SL
                fill = open_price if open_price <= sl_price else sl_price
                self._close_position(symbol, fill, date, "stop_loss")
                continue

            # Take-profit (gap-through: open above TP → fill at open)
            tp_price = pos.avg_price * (1 + pos.take_profit_pct)
            if high >= tp_price:
                if held_days < min_hold:
                    continue  # Min hold not met for TP
                fill = open_price if open_price >= tp_price else tp_price
                self._close_position(symbol, fill, date, "take_profit")
                continue

            # Trailing stop (gap-through: open below trail → fill at open)
            cfg = self._config
            if cfg.trailing_activation_pct > 0 and cfg.trailing_trail_pct > 0:
                gain = (pos.highest_price - pos.avg_price) / pos.avg_price
                if gain >= cfg.trailing_activation_pct:
                    trail_price = pos.highest_price * (1 - cfg.trailing_trail_pct)
                    if low <= trail_price:
                        if held_days < min_hold and not is_hard_sl:
                            continue
                        fill = open_price if open_price <= trail_price else trail_price
                        self._close_position(
                            symbol, fill, date, "trailing_stop",
                        )

    # ------------------------------------------------------------------
    # Regime-change protective sells
    # ------------------------------------------------------------------

    def _check_regime_sells(
        self,
        stock_data: dict,
        date_idx: int,
        date,
        current_regime: str,
    ) -> None:
        """Sell losing positions when regime transitions to downtrend.

        Mirrors live evaluation_loop._check_protective_sells():
        when market moves from uptrend/strong_uptrend to downtrend,
        positions with negative PnL are closed to protect capital.
        """
        _BEARISH = {"downtrend", "weak_downtrend"}

        regime_worsened = (
            current_regime in _BEARISH
            and self._prev_regime not in _BEARISH
        )
        if not regime_worsened:
            return

        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            data = stock_data.get(symbol)
            if not data or date_idx >= len(data.df):
                continue

            row = data.df.iloc[date_idx]
            price = float(row["close"])
            pnl_pct = (price - pos.avg_price) / pos.avg_price

            if pnl_pct < 0:
                logger.debug(
                    "Regime sell %s: %s→%s, PnL=%.1f%%",
                    symbol, self._prev_regime, current_regime, pnl_pct * 100,
                )
                vol = float(row["volume"]) if "volume" in row.index else 0.0
                self._close_position(symbol, price, date, "regime_protect", volume=vol)

    # ------------------------------------------------------------------
    # Extended hours simulation
    # ------------------------------------------------------------------

    def _check_extended_hours_sl(
        self, stock_data: dict, date_idx: int, date,
    ) -> None:
        """Pre-market gap-down defense: realistic gap-through SL handling.

        When a stock gaps down through SL at open, the limit SL order
        would NOT fill at sl_price — it fills at open (gap-through).
        This method replaces the optimistic SL fill with a realistic one.
        The SL exit itself is recorded as regular (not extended) because
        it would happen in both scenarios; the difference is fill quality.

        Additionally, check for gap-up through TP at open (bonus exits).
        """
        cfg = self._config
        ext_slippage = cfg.slippage_pct * cfg.extended_hours_slippage_multiplier

        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            if pos.strategy_name in {"cash_parking", "etf_leverage", "etf_inverse"}:
                continue
            data = stock_data.get(symbol)
            if not data or date_idx >= len(data.df):
                continue

            row = data.df.iloc[date_idx]
            open_price = float(row["open"]) if "open" in row.index else float(row["close"])
            close_price = float(row["close"])

            sl_price = pos.avg_price * (1 - pos.stop_loss_pct)
            tp_price = pos.avg_price * (1 + pos.take_profit_pct)

            # Gap-down through SL: exit at open (realistic gap-through)
            # This position would be caught by regular SL too, but with
            # unrealistic fill at sl_price. Extended hours catches it at open
            # which is more realistic. If close > open, we saved losses
            # by exiting early; if close < open, regular SL at sl_price
            # was better but unrealistic.
            if open_price <= sl_price:
                # Exit at open with extended slippage
                exec_price = open_price * (1 - ext_slippage / 100)
                self._close_position_ext(
                    symbol, exec_price, date, pos, "extended_sl",
                )
                continue

            # Gap-up through TP at open: capture gap-up profit
            # Regular TP fills at tp_price but misses the extra gain
            # when stock opens above TP. Extended hours fills at open.
            if open_price >= tp_price:
                exec_price = open_price * (1 - ext_slippage / 100)
                if exec_price > tp_price:  # Only if better than regular TP
                    self._close_position_ext(
                        symbol, exec_price, date, pos, "extended_tp",
                    )

    def _close_position_ext(
        self, symbol: str, exec_price: float, date,
        pos: _Position, reason: str,
    ) -> None:
        """Close position via extended hours (records session='extended')."""
        cfg = self._config
        proceeds = pos.quantity * exec_price - cfg.commission_per_order
        self._cash += proceeds

        pnl = (exec_price - pos.avg_price) * pos.quantity
        pnl_pct = (exec_price - pos.avg_price) / pos.avg_price * 100

        try:
            entry = pd.Timestamp(pos.entry_date)
            exit_ = pd.Timestamp(str(date))
            holding_days = (exit_ - entry).days
        except Exception as e:
            logger.debug("Holding days calculation failed for %s: %s", symbol, e)
            holding_days = 0

        self._trades.append(Trade(
            symbol=symbol,
            side="SELL",
            entry_date=pos.entry_date,
            entry_price=pos.avg_price,
            exit_date=str(date),
            exit_price=exec_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            strategy_name=pos.strategy_name,
            session="extended",
        ))

        self._risk_manager.update_daily_pnl(pnl)
        self._signal_quality.record_trade(
            pos.strategy_name, symbol, pnl_pct / 100,
        )
        del self._positions[symbol]
        self._recovery_watch[symbol] = self._day_count

        logger.debug(
            "Extended %s %s at %.2f (PnL=%.1f%%)",
            reason, symbol, exec_price, pnl_pct,
        )

    def _execute_extended_hours_buys(
        self,
        buy_candidates: list[tuple[float, str, Signal]],
        stock_data: dict,
        date_idx: int,
        date,
        regime,
    ) -> None:
        """Execute extended hours buys when regular session can't.

        Two scenarios:
        1. Spillover: high-confidence candidates that couldn't be bought
           because max_positions was reached in regular session.
        2. Dip buys: symbols where today's open < yesterday's close (gap down),
           offering a better entry than regular session close price.

        Uses current bar's open price (observable pre-market data, no look-ahead).
        Conservative sizing, higher slippage, fill probability.
        """
        cfg = self._config
        ext_slippage = cfg.slippage_pct * cfg.extended_hours_slippage_multiplier

        # Count active (non-system) positions
        system_strategies = {"cash_parking", "etf_leverage", "etf_inverse"}
        active_positions = sum(
            1 for p in self._positions.values()
            if p.strategy_name not in system_strategies
        )

        ext_count = sum(
            1 for p in self._positions.values()
            if p.session == "extended"
        )

        for confidence, symbol, signal in buy_candidates:
            # Skip if already holding (bought in regular session)
            if symbol in self._positions:
                continue

            # Only high-confidence signals
            if confidence < cfg.extended_hours_min_confidence:
                continue

            # Max 5 extended hours positions
            if ext_count >= 5:
                break

            # Extended hours allows exceeding regular max_positions by up to 5
            if active_positions >= cfg.max_positions + 5:
                break

            # Fill probability check
            if random.random() > cfg.extended_hours_fill_probability:
                continue

            data = stock_data.get(symbol)
            if not data or date_idx < 1 or date_idx >= len(data.df):
                continue

            # Use current bar's open as extended hours entry price
            # (pre-market of today's session — observable, no look-ahead)
            current_row = data.df.iloc[date_idx]
            open_price = float(current_row["open"]) if "open" in current_row.index else None
            if not open_price or open_price <= 0:
                continue

            # Discount check: today's open vs yesterday's close (gap-down)
            prev_close = float(data.df.iloc[date_idx - 1]["close"])
            is_discount = open_price < prev_close * 0.995  # >0.5% gap down
            is_spillover = active_positions >= cfg.max_positions

            if not is_discount and not is_spillover:
                continue  # No advantage over regular session entry

            equity = self._calculate_equity(stock_data, date_idx)

            # Extended hours sizing: conservative
            max_allocation = equity * cfg.extended_hours_max_position_pct
            allocation = min(max_allocation, self._cash * 0.95)
            if allocation <= 0:
                continue

            exec_price = open_price * (1 + ext_slippage / 100)
            quantity = int(allocation / exec_price)
            if quantity <= 0:
                continue

            cost = quantity * exec_price + cfg.commission_per_order
            if cost > self._cash:
                continue

            # Determine SL/TP
            if cfg.dynamic_sl_tp:
                current_row = data.df.iloc[date_idx]
                atr_col = None
                for col in ("atr", "ATRr_14"):
                    if col in current_row.index and not pd.isna(current_row[col]):
                        atr_col = col
                        break
                if atr_col:
                    atr_val = float(current_row[atr_col])
                    sl_pct, tp_pct = self._risk_manager.calculate_dynamic_sl_tp(
                        open_price, atr_val,
                    )
                else:
                    sl_pct = cfg.default_stop_loss_pct
                    tp_pct = cfg.default_take_profit_pct
            else:
                sl_pct = cfg.default_stop_loss_pct
                tp_pct = cfg.default_take_profit_pct

            self._cash -= cost
            self._positions[symbol] = _Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=exec_price,
                entry_date=str(date),
                strategy_name=signal.strategy_name,
                highest_price=exec_price,
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
                session="extended",
                entry_day_count=self._day_count,
            )
            active_positions += 1
            ext_count += 1

            logger.debug(
                "Extended buy %s: %d @ %.2f (%s, conf=%.2f)",
                symbol, quantity, exec_price,
                "dip" if is_discount else "spillover", confidence,
            )

    # ------------------------------------------------------------------
    # Daily buy limit + confidence escalation
    # ------------------------------------------------------------------

    def _check_daily_buy_allowed(self, confidence: float) -> bool:
        """Check if a buy is allowed under daily limit + escalation rules.

        Mirrors live evaluation_loop._execute_signal() logic:
        - 0-60% usage: no extra requirement
        - 60-80% usage: confidence >= 0.65
        - 80-100% usage: confidence >= 0.75
        - Over limit: confidence >= 0.90 (override)
        """
        cfg = self._config
        limit = cfg.daily_buy_limit
        if limit <= 0:
            return True  # No limit

        if self._daily_buy_count >= limit:
            # Over limit: only ultra-high confidence override
            if cfg.enable_confidence_escalation and confidence >= 0.90:
                return True
            return False

        if not cfg.enable_confidence_escalation:
            return True  # Under limit, no escalation

        usage_ratio = self._daily_buy_count / limit
        if usage_ratio >= 0.8:
            return confidence >= 0.75
        elif usage_ratio >= 0.6:
            return confidence >= 0.65
        return True  # Under 60% usage: free

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _get_sector(self, symbol: str) -> str:
        """Look up GICS sector for a symbol (cached)."""
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).info
            sector = info.get("sector", "")
            self._sector_cache[symbol] = sector
            return sector
        except Exception:
            self._sector_cache[symbol] = ""
            return ""

    def _effective_slippage(self, volume: float, quantity: int) -> float:
        """Return slippage pct adjusted by participation rate if enabled."""
        base = self._config.slippage_pct
        if not self._config.volume_adjusted_slippage or volume <= 0:
            return base
        participation = quantity / volume
        if participation > 0.10:
            return base * 3.0
        if participation > 0.05:
            return base * 2.0
        if participation > 0.01:
            return base * 1.5
        return base

    def _execute_buy(
        self,
        symbol: str,
        stock_data: dict,
        date_idx: int,
        date,
        signal: Signal,
        regime: MarketRegime,
    ) -> None:
        """Execute a buy order with portfolio-level risk checks."""
        if symbol in self._positions:
            return  # Already holding

        cfg = self._config

        # Trading gate: sell cooldown
        if cfg.sell_cooldown_days > 0 and symbol in self._sell_cooldown:
            days_since = self._day_count - self._sell_cooldown[symbol]
            if days_since < cfg.sell_cooldown_days:
                return

        # Trading gate: whipsaw counter
        if cfg.whipsaw_max_losses > 0 and symbol in self._loss_sell_history:
            cutoff = self._day_count - 7
            recent = [d for d in self._loss_sell_history[symbol] if d >= cutoff]
            if len(recent) >= cfg.whipsaw_max_losses:
                return
        data = stock_data[symbol]
        if date_idx >= len(data.df):
            return

        price = float(data.df.iloc[date_idx]["close"])
        if price <= 0:
            return

        equity = self._calculate_equity(stock_data, date_idx)

        # Use Kelly-enhanced sizing (falls back to fixed if no trade history)
        strategy_name = signal.strategy_name
        metrics = self._signal_quality.get_metrics(strategy_name)
        win_rate, avg_win, avg_loss = metrics.kelly_inputs

        # Get momentum factor score for this stock
        factor_score = self._factor_scores.get(symbol, 0.0)

        # Count only stock positions (exclude parking, ETF)
        system_strategies = {"cash_parking", "etf_leverage", "etf_inverse"}
        active_positions = sum(
            1 for p in self._positions.values()
            if p.strategy_name not in system_strategies
        )

        sizing = self._risk_manager.calculate_kelly_position_size(
            symbol=symbol,
            price=price,
            portfolio_value=equity,
            cash_available=self._cash,
            current_positions=active_positions,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            signal_confidence=signal.confidence,
            factor_score=factor_score,
        )

        if not sizing.allowed:
            return

        # Apply slippage (volume-adjusted if enabled)
        row = data.df.iloc[date_idx]
        volume = float(row["volume"]) if "volume" in row.index else 0
        est_quantity = int(sizing.allocation_usd / price) if price > 0 else 0
        slippage = self._effective_slippage(volume, est_quantity)
        exec_price = price * (1 + slippage / 100)
        quantity = int(sizing.allocation_usd / exec_price)
        if quantity <= 0:
            return

        cost = quantity * exec_price + cfg.commission_per_order
        if cost > self._cash:
            return

        # Sector concentration check
        if cfg.max_sector_pct < 1.0:
            sector = self._get_sector(symbol)
            if sector:
                sector_value = sum(
                    p.quantity * p.avg_price
                    for p in self._positions.values()
                    if p.sector == sector
                    and p.strategy_name not in {"cash_parking", "etf_leverage", "etf_inverse"}
                )
                if (sector_value + cost) / equity > cfg.max_sector_pct:
                    return

        # Determine dynamic SL/TP
        if cfg.dynamic_sl_tp:
            atr_col = None
            row = data.df.iloc[date_idx]
            for col in ("atr", "ATRr_14"):
                if col in row.index and not pd.isna(row[col]):
                    atr_col = col
                    break
            if atr_col:
                atr_val = float(row[atr_col])
                sl_pct, tp_pct = self._risk_manager.calculate_dynamic_sl_tp(
                    price, atr_val,
                )
            else:
                sl_pct = cfg.default_stop_loss_pct
                tp_pct = cfg.default_take_profit_pct
        else:
            sl_pct = cfg.default_stop_loss_pct
            tp_pct = cfg.default_take_profit_pct

        # Execute
        self._cash -= cost
        self._positions[symbol] = _Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=exec_price,
            entry_date=str(date),
            strategy_name=strategy_name,
            highest_price=exec_price,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            entry_day_count=self._day_count,
            sector=self._get_sector(symbol) if cfg.max_sector_pct < 1.0 else "",
        )

    def _execute_sell(
        self,
        symbol: str,
        stock_data: dict,
        date_idx: int,
        date,
        signal: Signal,
    ) -> None:
        """Execute a sell signal."""
        if symbol not in self._positions:
            return

        data = stock_data.get(symbol)
        if not data or date_idx >= len(data.df):
            return

        # Min hold check for signal-based sells
        pos = self._positions[symbol]
        held_days = self._day_count - pos.entry_day_count
        if held_days < self._config.min_hold_days:
            price = float(data.df.iloc[date_idx]["close"])
            unrealized_pct = (price - pos.avg_price) / pos.avg_price
            if unrealized_pct > -self._config.hard_sl_pct:
                return  # Min hold not met, not hard SL

        row = data.df.iloc[date_idx]
        price = float(row["close"])
        vol = float(row["volume"]) if "volume" in row.index else 0.0
        self._close_position(symbol, price, date, "signal_sell", volume=vol)

    def _close_position(
        self, symbol: str, price: float, date, reason: str = "",
        volume: float = 0.0,
    ) -> None:
        """Close a position and record the trade."""
        pos = self._positions.get(symbol)
        if not pos:
            return

        cfg = self._config
        # Apply volume-adjusted slippage symmetrically (same as buy side)
        slippage = self._effective_slippage(volume, pos.quantity)
        exec_price = price * (1 - slippage / 100)
        proceeds = pos.quantity * exec_price - cfg.commission_per_order
        self._cash += proceeds

        pnl = (exec_price - pos.avg_price) * pos.quantity
        pnl_pct = (exec_price - pos.avg_price) / pos.avg_price * 100

        try:
            entry = pd.Timestamp(pos.entry_date)
            exit_ = pd.Timestamp(str(date))
            holding_days = (exit_ - entry).days
        except Exception as e:
            logger.debug("Holding days calculation failed for %s: %s", symbol, e)
            holding_days = 0

        self._trades.append(Trade(
            symbol=symbol,
            side="SELL",
            entry_date=pos.entry_date,
            entry_price=pos.avg_price,
            exit_date=str(date),
            exit_price=exec_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            strategy_name=pos.strategy_name,
            session=pos.session,
        ))

        # Update daily PnL for risk manager
        self._risk_manager.update_daily_pnl(pnl)

        # Record for signal quality tracking
        self._signal_quality.record_trade(
            pos.strategy_name, symbol, pnl_pct / 100,
        )

        del self._positions[symbol]

        # Trading gate tracking
        if reason not in ("end_of_backtest",):
            # Sell cooldown: record when this symbol was sold
            self._sell_cooldown[symbol] = self._day_count
            # Whipsaw tracking: record loss sells
            if pnl < 0:
                self._loss_sell_history.setdefault(symbol, []).append(
                    self._day_count,
                )

            # Add to recovery watch for re-entry evaluation
            self._recovery_watch[symbol] = self._day_count

    def _close_all_positions(
        self, stock_data: dict, date_idx: int, date,
    ) -> None:
        """Close all positions at end of backtest."""
        for symbol in list(self._positions.keys()):
            data = stock_data.get(symbol)
            if data and date_idx < len(data.df):
                price = float(data.df.iloc[date_idx]["close"])
            else:
                price = self._positions[symbol].avg_price
            self._close_position(symbol, price, date, "backtest_end")

    # ------------------------------------------------------------------
    # Portfolio valuation
    # ------------------------------------------------------------------

    def _calculate_equity(
        self, stock_data: dict, date_idx: int,
    ) -> float:
        """Calculate total portfolio value (cash + positions)."""
        position_value = 0.0
        for symbol, pos in self._positions.items():
            data = stock_data.get(symbol)
            if data and date_idx < len(data.df):
                price = float(data.df.iloc[date_idx]["close"])
            else:
                price = pos.avg_price
            position_value += pos.quantity * price
        return self._cash + position_value

    # ------------------------------------------------------------------
    # Strategy stats
    # ------------------------------------------------------------------

    def _compute_strategy_stats(self) -> dict[str, dict]:
        """Compute per-strategy trade statistics."""
        stats: dict[str, dict] = {}
        for trade in self._trades:
            name = trade.strategy_name or "unknown"
            if name not in stats:
                stats[name] = {
                    "trades": 0, "wins": 0, "losses": 0,
                    "pnl": 0.0, "win_rate": 0.0,
                }
            s = stats[name]
            s["trades"] += 1
            s["pnl"] += trade.pnl
            if trade.pnl > 0:
                s["wins"] += 1
            else:
                s["losses"] += 1

        for s in stats.values():
            if s["trades"] > 0:
                s["win_rate"] = s["wins"] / s["trades"] * 100

        return stats
