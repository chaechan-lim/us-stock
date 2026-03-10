"""ETF Universe Manager.

Manages leveraged/inverse ETF mappings, sector ETFs, and risk rules
for the ETF trading engine.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "etf_universe.yaml"


@dataclass
class LeveragedPair:
    base: str
    bull: str | None
    bear: str | None
    leverage: int = 3


@dataclass
class SectorETF:
    name: str
    etf: str
    leveraged: str | None = None
    top_holdings: list[str] = field(default_factory=list)
    mid_holdings: list[str] = field(default_factory=list)


@dataclass
class ETFRiskRules:
    max_hold_days: int = 10
    max_portfolio_pct: float = 0.30
    max_single_etf_pct: float = 0.15
    require_stop_loss: bool = True
    default_stop_loss_pct: float = 0.08


class ETFUniverse:
    """Manage ETF universe with leveraged pairs, sectors, and risk rules."""

    def __init__(self, config_path: str | Path | None = None):
        self._path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._pairs: dict[str, LeveragedPair] = {}
        self._sectors: dict[str, SectorETF] = {}
        self._safe_haven: list[str] = []
        self._volatility: list[str] = []
        self._risk_rules = ETFRiskRules()
        # Map from index name -> base ETF symbol (e.g. "KOSPI200" -> "069500")
        self._base_symbols: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning("ETF universe config not found: %s", self._path)
            return

        with open(self._path) as f:
            data = yaml.safe_load(f)

        # Leveraged pairs
        for base, pair in (data.get("leveraged_pairs") or {}).items():
            self._pairs[base] = LeveragedPair(
                base=base,
                bull=pair.get("bull"),
                bear=pair.get("bear"),
                leverage=pair.get("leverage", 3),
            )
            # Store base ETF symbol (explicit 'base' field in KR config,
            # or the key itself is the base ETF in US config like QQQ/SPY)
            base_sym = pair.get("base", base)
            self._base_symbols[base] = base_sym

        # Sectors
        for name, sec in (data.get("sectors") or {}).items():
            self._sectors[name] = SectorETF(
                name=name,
                etf=sec["etf"],
                leveraged=sec.get("leveraged"),
                top_holdings=sec.get("top_holdings", []),
                mid_holdings=sec.get("mid_holdings", []),
            )

        self._safe_haven = data.get("safe_haven", [])
        self._volatility = data.get("volatility", [])

        # Exchange code overrides (default: AMEX for most ETFs)
        self._exchanges: dict[str, str] = data.get("exchanges", {})

        # Risk rules
        rules = data.get("risk_rules", {})
        self._risk_rules = ETFRiskRules(**rules)

        logger.info(
            "ETF universe loaded: %d pairs, %d sectors",
            len(self._pairs), len(self._sectors),
        )

    def get_bull_etf(self, base: str) -> str | None:
        pair = self._pairs.get(base)
        return pair.bull if pair else None

    def get_bear_etf(self, base: str) -> str | None:
        pair = self._pairs.get(base)
        return pair.bear if pair else None

    def get_pair(self, base: str) -> LeveragedPair | None:
        return self._pairs.get(base)

    def get_regime_etfs(self, regime: str) -> list[str]:
        """Get ETFs for a given regime (bull/bear)."""
        etfs = []
        for pair in self._pairs.values():
            if regime == "bull" and pair.bull:
                etfs.append(pair.bull)
            elif regime == "bear" and pair.bear:
                etfs.append(pair.bear)
        return etfs

    def get_sector(self, name: str) -> SectorETF | None:
        return self._sectors.get(name)

    def get_all_sectors(self) -> dict[str, SectorETF]:
        return dict(self._sectors)

    def get_sector_etf_symbols(self) -> list[str]:
        """Get all standard sector ETF symbols."""
        return [s.etf for s in self._sectors.values()]

    def get_sector_for_symbol(self, symbol: str) -> str | None:
        """Find which sector a stock belongs to."""
        for name, sec in self._sectors.items():
            if symbol in sec.top_holdings:
                return name
        return None

    def get_exchange(self, symbol: str) -> str:
        """Get KIS exchange code for an ETF symbol.

        Most ETFs trade on NYSE Arca (AMEX). Overrides loaded from config.
        """
        return self._exchanges.get(symbol, "AMEX")

    def get_pair_siblings(self, symbol: str) -> list[str]:
        """Get all sibling ETFs in the same leveraged pair (excluding self).

        For example, if symbol is KODEX 코스닥150 (229200, base),
        returns [233740, 251340] (bull and bear).
        Used to enforce mutual exclusivity: only one member of a pair
        should be held at any time.
        """
        for index_name, pair in self._pairs.items():
            base_sym = self._base_symbols.get(index_name, index_name)
            all_members = [m for m in (base_sym, pair.bull, pair.bear) if m]
            if symbol in all_members:
                return [m for m in all_members if m != symbol]
        return []

    def is_leveraged(self, symbol: str) -> bool:
        """Check if a symbol is a leveraged ETF."""
        for pair in self._pairs.values():
            if symbol in (pair.bull, pair.bear):
                return True
        for sec in self._sectors.values():
            if symbol == sec.leveraged:
                return True
        return False

    @property
    def risk_rules(self) -> ETFRiskRules:
        return self._risk_rules

    @property
    def safe_haven(self) -> list[str]:
        return list(self._safe_haven)

    @property
    def all_etf_symbols(self) -> list[str]:
        """All known ETF symbols."""
        symbols = set()
        for pair in self._pairs.values():
            symbols.add(pair.base)
            if pair.bull:
                symbols.add(pair.bull)
            if pair.bear:
                symbols.add(pair.bear)
        for sec in self._sectors.values():
            symbols.add(sec.etf)
            if sec.leveraged:
                symbols.add(sec.leveraged)
        symbols.update(self._safe_haven)
        symbols.update(self._volatility)
        return sorted(symbols)
