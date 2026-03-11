import axios from 'axios'
import type {
  PortfolioSummary,
  Position,
  TickerData,
  ChartData,
  Strategy,
  ScanResult,
  EngineStatus,
  WatchlistResponse,
  Trade,
  TradeSummary,
  ETFStatus,
} from '../types'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 15_000,
})

// Portfolio
export const fetchPortfolioSummary = (market = 'ALL') =>
  api.get<PortfolioSummary>('/portfolio/summary', { params: { market } }).then(r => r.data)

export const fetchPositions = (market = 'ALL') =>
  api.get<Position[]>('/portfolio/positions', { params: { market } }).then(r => r.data)

// Market
export const fetchPrice = (symbol: string) =>
  api.get<TickerData>(`/market/price/${symbol}`).then(r => r.data)

export const fetchChart = (symbol: string, timeframe = '1D', limit = 200, market = 'US') =>
  api.get<ChartData>(`/market/chart/${symbol}`, {
    params: { timeframe, limit, market },
  }).then(r => r.data)

// Stock names
export const fetchStockNames = (symbols: string[], market = 'US') =>
  api.get<Record<string, string>>('/market/names', {
    params: { symbols: symbols.join(','), market },
  }).then(r => r.data)

// Strategies
export const fetchStrategies = () =>
  api.get<Strategy[]>('/strategies/').then(r => r.data)

export const reloadStrategies = () =>
  api.post('/strategies/reload').then(r => r.data)

// Scanner
export const runScan = (symbols: string[], minGrade = 'B', maxCandidates = 20) =>
  api.post<ScanResult[]>('/scanner/run', {
    symbols,
    min_grade: minGrade,
    max_candidates: maxCandidates,
  }).then(r => r.data)

export const fetchSectorPerformance = (market: string = 'US') =>
  api.get('/scanner/sectors', { params: { market } }).then(r => r.data)

// Engine
export const fetchEngineStatus = () =>
  api.get<EngineStatus>('/engine/status').then(r => r.data)

export const startEngine = () =>
  api.post('/engine/start').then(r => r.data)

export const stopEngine = () =>
  api.post('/engine/stop').then(r => r.data)

// Watchlist
export const fetchWatchlist = (market = 'US') =>
  api.get<WatchlistResponse>('/watchlist/', { params: { market } }).then(r => r.data)

export const addToWatchlist = (symbol: string, market = 'US') =>
  api.post<WatchlistResponse>('/watchlist/', { symbol, market }).then(r => r.data)

export const removeFromWatchlist = (symbol: string, market = 'US') =>
  api.delete<WatchlistResponse>(`/watchlist/${symbol}`, { params: { market } }).then(r => r.data)

// Trades
export const fetchTrades = (limit = 50, market?: string) =>
  api.get<Trade[]>('/trades/', { params: { limit, ...(market && { market }) } }).then(r => r.data)

export const fetchTradeSummary = (market?: string) =>
  api.get<TradeSummary>('/trades/summary', { params: { ...(market && { market }) } }).then(r => r.data)

// Backtest
export const runBacktest = (params: {
  strategy_name: string
  symbol: string
  period?: string
  initial_equity?: number
}) => api.post('/backtest/run', params).then(r => r.data)

export const fetchBacktestStrategies = () =>
  api.get('/backtest/strategies').then(r => r.data)

// Portfolio history
export const fetchEquityHistory = (days = 30, market = 'US') =>
  api.get('/portfolio/equity-history', { params: { days, market } }).then(r => r.data)

// Recovery
export const fetchRecoveryStatus = () =>
  api.get('/engine/recovery').then(r => r.data)

// Backtest results store
export const fetchBacktestResults = (strategy?: string, symbol?: string) =>
  api.get('/backtest/results', { params: { strategy, symbol } }).then(r => r.data)

export const deleteBacktestResult = (key: string) =>
  api.delete(`/backtest/results/${key}`).then(r => r.data)

// Optimization
export const runOptimization = (params: {
  strategy_name?: string
  symbols?: string[]
  period?: string
  metric?: string
}) => api.post('/backtest/optimize', params, { timeout: 300_000 }).then(r => r.data)

export const fetchParamGrids = () =>
  api.get('/backtest/optimize/grids').then(r => r.data)

// Engine extras
export const fetchMacroIndicators = () =>
  api.get('/engine/macro').then(r => r.data)

export const fetchAdaptiveWeights = () =>
  api.get('/engine/adaptive-weights').then(r => r.data)

export const fetchMarketState = () =>
  api.get('/engine/market-state').then(r => r.data)

export const runEvaluation = () =>
  api.post('/engine/evaluate', {}, { timeout: 120_000 }).then(r => r.data)

// ETF Engine
export const fetchETFStatus = (market = 'US') =>
  api.get<ETFStatus>(market === 'KR' ? '/engine/etf/kr' : '/engine/etf').then(r => r.data)

// Trade Summary (daily/weekly/monthly)
export interface PeriodSummary {
  trades: number
  wins: number
  losses: number
  pnl: number
  win_rate: number
}
export interface TradeSummaryPeriods {
  today: PeriodSummary
  week: PeriodSummary
  month: PeriodSummary
  all_time: PeriodSummary
  total_buys: number
  total_sells: number
}
export const fetchTradeSummaryPeriods = (market?: string) =>
  api.get<TradeSummaryPeriods>('/portfolio/trade-summary', { params: { ...(market && { market }) } }).then(r => r.data)

// News Sentiment
export interface SentimentSignal {
  symbol: string
  sentiment: number
  impact: string
  category: string
  sector_impact: string[]
  key_event: string
  trading_signal: string
  time_sensitivity: string
  is_actionable: boolean
}
export interface SentimentSummary {
  symbol_sentiments: Record<string, number>
  sector_sentiments: Record<string, number>
  market_sentiment: number
  actionable_count: number
  analyzed_count: number
}
export interface NewsSentimentData {
  summary: SentimentSummary
  signals: SentimentSignal[]
  updated_at: string | null
  kr?: {
    summary: SentimentSummary
    signals: SentimentSignal[]
    updated_at: string | null
  }
}
export const fetchNewsSentiment = () =>
  api.get<NewsSentimentData>('/news/sentiment').then(r => r.data)

// Market Events
export interface EarningsEvent {
  symbol: string
  date: string
  hour: string
  eps_estimate: number | null
  eps_actual: number | null
  revenue_estimate: number | null
  revenue_actual: number | null
}
export interface MacroEvent {
  date: string
  event_type: string
  description: string
}
export interface InsiderSignal {
  symbol: string
  signal: string
  total_value: number
  count: number
  top_buyer?: string
  top_seller?: string
}
export interface EventCalendarData {
  earnings: EarningsEvent[]
  macro: MacroEvent[]
  insider: InsiderSignal[]
}
export const fetchMarketEvents = () =>
  api.get<EventCalendarData>('/market/events').then(r => r.data)

// Signals
export interface SignalEntry {
  timestamp: string
  symbol: string
  signal: string
  confidence: number
  strategy: string
  market_state: string
  market: string
}
export const fetchSignals = (market = 'ALL', limit = 100) =>
  api.get<SignalEntry[]>('/engine/signals', { params: { market, limit } }).then(r => r.data)
