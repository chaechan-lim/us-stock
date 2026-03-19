"""Backtest order simulator.

Simulates order execution with slippage, tracks positions and equity.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.metrics import Trade
from strategies.base import Signal
from core.enums import SignalType

logger = logging.getLogger(__name__)


@dataclass
class SimPosition:
    symbol: str
    quantity: float
    avg_price: float
    entry_date: str
    strategy_name: str = ""
    highest_price: float = 0.0  # Track peak for trailing stop


@dataclass
class SimConfig:
    initial_equity: float = 100_000.0
    slippage_pct: float = 0.05  # 0.05% default
    commission_per_order: float = 0.0  # KIS US stocks: $0
    fx_spread_pct: float = 0.25  # KRW/USD spread
    max_position_pct: float = 0.10  # max 10% per position
    max_total_positions: int = 20
    # Stop-loss / take-profit / trailing stop
    stop_loss_pct: float = 0.0  # 0 = disabled, e.g. 0.08 = 8% stop
    take_profit_pct: float = 0.0  # 0 = disabled, e.g. 0.20 = 20% TP
    trailing_stop_activation_pct: float = 0.0  # 0 = disabled
    trailing_stop_trail_pct: float = 0.0  # trail distance from peak


class BacktestSimulator:
    """Simulates trading on historical data."""

    def __init__(self, config: SimConfig | None = None):
        self._config = config or SimConfig()
        self._equity = self._config.initial_equity
        self._cash = self._config.initial_equity
        self._positions: dict[str, SimPosition] = {}
        self._trades: list[Trade] = []
        self._equity_curve: list[float] = []
        self._equity_dates: list = []

    @property
    def trades(self) -> list[Trade]:
        return self._trades

    @property
    def equity_curve(self) -> pd.Series:
        if not self._equity_curve:
            return pd.Series(dtype=float)
        return pd.Series(self._equity_curve, index=self._equity_dates)

    @property
    def positions(self) -> dict[str, SimPosition]:
        return self._positions

    def run(
        self,
        df: pd.DataFrame,
        signals: dict[int, Signal],
        symbol: str,
    ) -> None:
        """Run simulation on a single symbol.

        Args:
            df: OHLCV DataFrame with indicators
            signals: Dict mapping row index -> Signal
            symbol: Stock symbol
        """
        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]
            price = float(row["close"])
            high = float(row["high"]) if "high" in row.index else price
            low = float(row["low"]) if "low" in row.index else price

            # Check SL/TP/trailing stop on existing positions
            self._check_risk_exits(symbol, low, high, price, date)

            signal = signals.get(i)
            if signal:
                self._process_signal(signal, symbol, price, date)

            # Update highest price for trailing stop
            pos = self._positions.get(symbol)
            if pos and high > pos.highest_price:
                pos.highest_price = high

            # Update equity
            self._update_equity(price, symbol, date)

    def _check_risk_exits(
        self, symbol: str, low: float, high: float, close: float, date
    ) -> None:
        """Check stop-loss, take-profit, and trailing stop triggers."""
        pos = self._positions.get(symbol)
        if not pos:
            return

        cfg = self._config

        # Stop-loss: triggered if low breaches SL level
        if cfg.stop_loss_pct > 0:
            sl_price = pos.avg_price * (1 - cfg.stop_loss_pct)
            if low <= sl_price:
                self._close_position(symbol, sl_price, date)
                return

        # Take-profit: triggered if high breaches TP level
        if cfg.take_profit_pct > 0:
            tp_price = pos.avg_price * (1 + cfg.take_profit_pct)
            if high >= tp_price:
                self._close_position(symbol, tp_price, date)
                return

        # Trailing stop
        if cfg.trailing_stop_activation_pct > 0 and cfg.trailing_stop_trail_pct > 0:
            peak = max(pos.highest_price, high)
            gain_from_entry = (peak - pos.avg_price) / pos.avg_price
            if gain_from_entry >= cfg.trailing_stop_activation_pct:
                trail_price = peak * (1 - cfg.trailing_stop_trail_pct)
                if low <= trail_price:
                    self._close_position(symbol, trail_price, date)
                    return

    def _process_signal(
        self, signal: Signal, symbol: str, price: float, date
    ) -> None:
        if signal.signal_type == SignalType.BUY:
            self._open_position(symbol, price, date, signal)
        elif signal.signal_type == SignalType.SELL:
            self._close_position(symbol, price, date)

    def _open_position(
        self, symbol: str, price: float, date, signal: Signal
    ) -> None:
        if symbol in self._positions:
            return  # Already holding
        if len(self._positions) >= self._config.max_total_positions:
            return

        # Position sizing
        max_allocation = self._equity * self._config.max_position_pct
        allocation = min(max_allocation, self._cash * 0.95)  # Keep 5% buffer
        if allocation <= 0:
            return

        # Apply slippage (buy higher)
        exec_price = price * (1 + self._config.slippage_pct / 100)
        quantity = int(allocation / exec_price)
        if quantity <= 0:
            return

        cost = quantity * exec_price + self._config.commission_per_order
        if cost > self._cash:
            return

        self._cash -= cost
        self._positions[symbol] = SimPosition(
            symbol=symbol,
            quantity=quantity,
            avg_price=exec_price,
            entry_date=str(date),
            strategy_name=signal.strategy_name,
            highest_price=exec_price,
        )

    def _close_position(self, symbol: str, price: float, date) -> None:
        pos = self._positions.get(symbol)
        if not pos:
            return

        # Apply slippage (sell lower)
        exec_price = price * (1 - self._config.slippage_pct / 100)
        proceeds = pos.quantity * exec_price - self._config.commission_per_order
        self._cash += proceeds

        pnl = (exec_price - pos.avg_price) * pos.quantity
        pnl_pct = (exec_price - pos.avg_price) / pos.avg_price * 100

        # Calculate holding days
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
        ))

        del self._positions[symbol]

    def _update_equity(self, price: float, symbol: str, date) -> None:
        position_value = sum(
            pos.quantity * (price if pos.symbol == symbol else pos.avg_price)
            for pos in self._positions.values()
        )
        self._equity = self._cash + position_value
        self._equity_curve.append(self._equity)
        self._equity_dates.append(date)

    def reset(self) -> None:
        """Reset simulator state."""
        self._equity = self._config.initial_equity
        self._cash = self._config.initial_equity
        self._positions.clear()
        self._trades.clear()
        self._equity_curve.clear()
        self._equity_dates.clear()
