"""Scanner API endpoints."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from api.dependencies import get_market_data
from data.market_data_service import MarketDataService
from data.indicator_service import IndicatorService
from scanner.indicator_screener import IndicatorScreener

router = APIRouter(prefix="/scanner", tags=["scanner"])


class ScanRequest(BaseModel):
    symbols: list[str]
    min_grade: str = "B"
    max_candidates: int = 20


@router.post("/run")
async def run_scan(
    req: ScanRequest,
    market_data: MarketDataService = Depends(get_market_data),
):
    """Run indicator screening on a list of symbols."""
    screener = IndicatorScreener(min_grade=req.min_grade)
    indicator_svc = IndicatorService()
    scores = []

    for symbol in req.symbols:
        try:
            df = await market_data.get_ohlcv(symbol, limit=250)
            if df.empty:
                continue
            df = indicator_svc.add_all_indicators(df)
            score = screener.score(df, symbol)
            scores.append(score)
        except Exception:
            continue

    filtered = screener.filter_candidates(scores, max_candidates=req.max_candidates)
    return [
        {
            "symbol": s.symbol,
            "total_score": s.total_score,
            "trend_score": s.trend_score,
            "momentum_score": s.momentum_score,
            "volatility_volume_score": s.volatility_volume_score,
            "support_resistance_score": s.support_resistance_score,
            "grade": s.grade,
            "details": s.details,
        }
        for s in filtered
    ]


@router.get("/sectors")
async def sector_performance():
    """Get sector ETF performance data as array for frontend."""
    from data.external_data_service import ExternalDataService
    svc = ExternalDataService()
    data = await svc.get_sector_performance()
    if not data:
        return []
    # Convert {name: {symbol, return_1d, ...}} to [{sector, etf_symbol, ...}]
    result = []
    for sector_name, info in data.items():
        if isinstance(info, dict):
            result.append({
                "sector": sector_name,
                "etf_symbol": info.get("symbol", ""),
                "return_1w": info.get("return_1w", 0),
                "return_1m": info.get("return_1m", 0),
            })
    return result
