export interface PortfolioSummary {
  market: string
  balance: {
    currency: string
    total: number
    available: number
    locked?: number
  }
  usd_balance?: {
    total: number
    available: number
  }
  positions_count: number
  total_position_value?: number
  total_unrealized_pnl: number
  total_unrealized_pnl_usd?: number
  total_equity: number
}

export interface PositionWithMarket extends Position {
  market: string
}

export interface Position {
  symbol: string
  name?: string
  exchange: string
  quantity: number
  avg_price: number
  current_price: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
  market?: string
}

export interface TickerData {
  symbol: string
  price: number
  change_pct: number
  volume: number
}

export interface ChartData {
  symbol: string
  timeframe: string
  data: OHLCVRow[]
}

export interface OHLCVRow {
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Strategy {
  name: string
  display_name: string
  applicable_market_types: string[]
  timeframe: string
  params: Record<string, unknown>
}

export interface ScanResult {
  symbol: string
  total_score: number
  trend_score: number
  momentum_score: number
  volatility_volume_score: number
  support_resistance_score: number
  grade: string
  details: Record<string, unknown>
}

export interface EngineStatus {
  running: boolean
  market_phase: string
  market_time_et?: string
  kr_market_phase?: string
  kr_market_time_kst?: string
  tasks?: {
    name: string
    interval_sec: number
    phases: string[] | null
    last_run: string | null
    active: boolean
  }[]
}

export interface WatchlistItem {
  symbol: string
  name?: string
  market?: string
}

export interface WatchlistResponse {
  symbols: string[]
  items?: WatchlistItem[]
}

export interface Trade {
  symbol: string
  name?: string
  side: string
  quantity: number
  price: number
  filled_price: number | null
  strategy: string
  status: string
  pnl: number | null
  created_at: string
}

export interface TradeSummary {
  total_trades: number
  wins: number
  losses: number
  total_pnl: number
  win_rate: number
}

export interface ETFManagedPosition {
  reason: string
  sector: string
  hold_days: number
}

export interface ETFStatus {
  status?: string
  last_regime: string | null
  top_sectors: string[]
  managed_positions: Record<string, ETFManagedPosition>
  risk_params: {
    max_hold_days: number
    max_portfolio_pct: number
    max_single_etf_pct: number
  }
  bear_config: {
    min_distance_pct: number
    min_confidence: number
    size_ratio: number
  }
}
