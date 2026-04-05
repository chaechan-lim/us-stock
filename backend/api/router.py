"""API router aggregator."""

from fastapi import APIRouter

from api.accounts import router as accounts_router
from api.backtest_api import router as backtest_router
from api.engine_api import router as engine_router
from api.market import router as market_router
from api.news import router as news_router
from api.orders import router as orders_router
from api.portfolio import router as portfolio_router
from api.positions import router as positions_router
from api.scanner_api import router as scanner_router
from api.strategies import router as strategies_router
from api.trades import router as trades_router
from api.watchlist import router as watchlist_router
from api.ws import router as ws_router

api_router = APIRouter()

api_router.include_router(accounts_router)
api_router.include_router(portfolio_router)
api_router.include_router(positions_router)
api_router.include_router(orders_router)
api_router.include_router(market_router)
api_router.include_router(strategies_router)
api_router.include_router(scanner_router)
api_router.include_router(engine_router)
api_router.include_router(watchlist_router)
api_router.include_router(trades_router)
api_router.include_router(backtest_router)
api_router.include_router(ws_router)
api_router.include_router(news_router)
