import axios from 'axios'
import type {
  AccountInfo,
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

// SEC-B1 (2026-06-06): backend bearer middleware applies to every
// non-exempt path including GETs. Frontend supplies the token via
// Vite env (VITE_API_TOKEN). Without it the dashboard cannot reach
// the API in live mode — leave undefined for dev / paper.
const apiToken = (import.meta as ImportMeta & { env: { VITE_API_TOKEN?: string } })
  .env?.VITE_API_TOKEN
if (apiToken) {
  api.defaults.headers.common['Authorization'] = `Bearer ${apiToken}`
}

// HIGH-14 (2026-06-07): global response interceptor for hot error
// classes. Surfaces a CustomEvent any layout banner can subscribe to,
// without coupling axios to a specific toast library. Per-call .catch
// handlers still see the original error so they can fall through.
export interface ApiErrorEvent {
  status: number
  url: string | undefined
  message: string
  level: 'auth' | 'rate' | 'server'
}
function emitApiError(detail: ApiErrorEvent): void {
  try {
    window.dispatchEvent(new CustomEvent('apiError', { detail }))
  } catch {
    /* SSR / non-DOM environments — ignore */
  }
}
api.interceptors.response.use(
  r => r,
  err => {
    const status = err?.response?.status ?? 0
    const url = err?.config?.url
    if (status === 401 || status === 403) {
      emitApiError({
        status, url,
        message: status === 401
          ? '인증 토큰이 없거나 만료되었습니다. VITE_API_TOKEN 확인 필요.'
          : '권한이 없습니다 (403).',
        level: 'auth',
      })
    } else if (status === 429) {
      emitApiError({
        status, url,
        message: '요청이 너무 많습니다. 잠시 후 다시 시도하세요 (429).',
        level: 'rate',
      })
    } else if (status >= 500) {
      emitApiError({
        status, url,
        message: `서버 오류 (${status}). 백엔드 로그를 확인하세요.`,
        level: 'server',
      })
    }
    return Promise.reject(err)
  },
)

// Accounts
export const fetchAccounts = () =>
  api.get<AccountInfo[]>('/accounts/').then(r => r.data)

// Portfolio
export const fetchPortfolioSummary = (market = 'ALL', accountId?: string | null) =>
  api.get<PortfolioSummary>('/portfolio/summary', {
    params: { market, ...(accountId ? { account_id: accountId } : {}) },
  }).then(r => r.data)

export const fetchPositions = (market = 'ALL', accountId?: string | null) =>
  api.get<Position[]>('/portfolio/positions', {
    params: { market, ...(accountId ? { account_id: accountId } : {}) },
  }).then(r => r.data)

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
export const fetchTrades = (opts: { limit?: number; market?: string; offset?: number; accountId?: string | null } = {}) => {
  const { limit = 50, market, offset = 0, accountId } = opts
  return api.get<Trade[]>('/trades/', {
    params: {
      limit,
      offset,
      ...(market && { market }),
      ...(accountId ? { account_id: accountId } : {}),
    },
  }).then(r => r.data)
}

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

// Portfolio returns (daily/weekly/monthly)
export interface PeriodReturn {
  change: number
  pct: number
  realized_kr?: number
  realized_us?: number
}
export interface PortfolioReturns {
  daily: PeriodReturn | null
  weekly: PeriodReturn | null
  monthly: PeriodReturn | null
}
export const fetchPortfolioReturns = () =>
  api.get<PortfolioReturns>('/portfolio/returns').then(r => r.data)

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
  pnl_pct: number | null
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
export const fetchTradeSummaryPeriods = (market?: string, accountId?: string | null) =>
  api.get<TradeSummaryPeriods>('/portfolio/trade-summary', {
    params: {
      ...(market && { market }),
      ...(accountId ? { account_id: accountId } : {}),
    },
  }).then(r => r.data)

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
export const fetchMarketEvents = (market: string = 'US') =>
  api.get<EventCalendarData>('/market/events', { params: { market } }).then(r => r.data)

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

// Performance metrics (cost-aware, equity-based hierarchy)
export interface PerformanceMetrics {
  window_days: number
  market: string
  equity: {
    start_equity: number
    end_equity: number
    net_return_pct: number
    annualized_return_pct: number
    max_drawdown_pct: number
    max_dd_recovery_days: number
    calmar_ratio: number
    sharpe_ratio: number
    sortino_ratio: number
    exposure_pct: number
    sample_days: number
    sufficient_samples: boolean
    intraday_max_drawdown_pct: number
    intraday_sample_count: number
  }
  trades: {
    total_trades: number
    wins: number
    losses: number
    win_rate: number
    avg_win: number
    avg_loss: number
    gross_profit: number
    gross_loss: number
    gross_pf: number | null
    expectancy: number
    estimated_fees: number
    estimated_slippage: number
    net_profit: number
    net_pf: number | null
    round_trips: number
    round_trip_wins: number
    round_trip_losses: number
    round_trip_win_rate: number
    round_trip_avg_pnl: number
  }
  benchmark: {
    symbol: string
    label: string
    return_pct: number | null
    alpha_pct: number | null
    adjusted_return_pct: number | null
    adjusted_alpha_pct: number | null
  }
  target: {
    target_exposure_pct: number | null
    current_exposure_pct: number
    exposure_gap_pct: number | null
  }
}
export const fetchPerformanceMetrics = (days = 30, market?: string) =>
  api.get<PerformanceMetrics>('/portfolio/metrics', {
    params: { days, ...(market ? { market } : {}) },
  }).then(r => r.data)

// Daily post-market analysis artifacts
export interface DailyAnalysisArtifact {
  version: number
  date: string
  generated_at: string
  title: string
  body: string
  level: string  // "info" | "warning" | "critical"
  daily: {
    date: string
    buys: number
    sells: number
    cleanups: number
    pnl_usd: number
    cleanup_pnl_usd: number
    baseline_5d_avg: {
      sells_per_day: number
      cleanups_per_day: number
      pnl_per_day: number
    }
  }
  spy_pct: number | null
  positions: { total?: number; gain?: number; flat?: number; loss?: number }
  funnel: Record<string, any>
}
export interface DailyAnalysisList {
  artifacts: DailyAnalysisArtifact[]
  count: number
}
export const fetchDailyAnalyses = (days = 7) =>
  api.get<DailyAnalysisList>('/analysis/daily', { params: { days } }).then(r => r.data)

// F1 attribution funnel
export interface RejectionFunnelMarket {
  buy_signals_total: number
  buys_placed: number
  rejected_total: number
  fill_rate: number | null
  rejections: Record<string, number>
  daily_buy_count: number
  daily_buy_limit: number
  daily_buy_date: string
}
export type RejectionFunnel = Partial<Record<'US' | 'KR', RejectionFunnelMarket>>
export const fetchRejectionFunnel = () =>
  api.get<RejectionFunnel>('/engine/rejection-funnel').then(r => r.data)

// Self-evolution recommendations
export interface AgentRecommendation {
  id: number
  created_at: string
  agent_type: string
  param_path: string
  current_value: any
  proposed_value: any
  rationale: string | null
  expected_effect: string | null
  confidence: string | null
  risk: string | null
  backtest_result: any
  status: string
  applied_at: string | null
  rejected_reason: string | null
  notes: string | null
}
export const fetchRecommendations = (status: string = 'pending', limit = 50) =>
  api.get<AgentRecommendation[]>('/recommendations/', {
    params: { status, limit },
  }).then(r => r.data)
export const acceptRecommendation = (id: number, notes?: string) =>
  api.post<AgentRecommendation>(`/recommendations/${id}/accept`, { notes }).then(r => r.data)
export const rejectRecommendation = (id: number, reason?: string) =>
  api.post<AgentRecommendation>(`/recommendations/${id}/reject`, { reason }).then(r => r.data)
