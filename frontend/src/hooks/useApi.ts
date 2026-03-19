import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import * as api from '../api/client'

export function usePortfolioSummary(market = 'ALL') {
  return useQuery({
    queryKey: ['portfolio', 'summary', market],
    queryFn: () => api.fetchPortfolioSummary(market),
    refetchInterval: 30_000,
  })
}

export function usePositions(market = 'ALL') {
  return useQuery({
    queryKey: ['portfolio', 'positions', market],
    queryFn: () => api.fetchPositions(market),
    refetchInterval: 30_000,
  })
}

export function usePortfolioReturns() {
  return useQuery({
    queryKey: ['portfolio', 'returns'],
    queryFn: api.fetchPortfolioReturns,
    refetchInterval: 60_000,
  })
}

export function usePrice(symbol: string) {
  return useQuery({
    queryKey: ['market', 'price', symbol],
    queryFn: () => api.fetchPrice(symbol),
    enabled: !!symbol,
  })
}

export function useChart(symbol: string, timeframe = '1D', market = 'US') {
  return useQuery({
    queryKey: ['market', 'chart', symbol, timeframe, market],
    queryFn: () => api.fetchChart(symbol, timeframe, 200, market),
    enabled: !!symbol,
    refetchInterval: 60_000,
  })
}

export function useStrategies() {
  return useQuery({
    queryKey: ['strategies'],
    queryFn: api.fetchStrategies,
    refetchInterval: 30_000,
  })
}

export function useEngineStatus() {
  return useQuery({
    queryKey: ['engine', 'status'],
    queryFn: api.fetchEngineStatus,
    refetchInterval: 30_000,
  })
}

export function useWatchlist(market = 'US') {
  return useQuery({
    queryKey: ['watchlist', market],
    queryFn: () => api.fetchWatchlist(market),
  })
}

export function useAddToWatchlist(market = 'US') {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (symbol: string) => api.addToWatchlist(symbol, market),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['watchlist'] }),
  })
}

export function useRemoveFromWatchlist(market = 'US') {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (symbol: string) => api.removeFromWatchlist(symbol, market),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['watchlist'] }),
  })
}

export function useTrades(limit = 50, market?: string, offset = 0) {
  return useQuery({
    queryKey: ['trades', limit, market, offset],
    queryFn: () => api.fetchTrades(limit, market, offset),
    refetchInterval: 15_000,
  })
}

export function useTradeSummary(market?: string) {
  return useQuery({
    queryKey: ['trades', 'summary', market],
    queryFn: () => api.fetchTradeSummary(market),
    refetchInterval: 15_000,
  })
}

export function useEngineControl() {
  const qc = useQueryClient()
  const startMutation = useMutation({
    mutationFn: api.startEngine,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engine'] }),
  })
  const stopMutation = useMutation({
    mutationFn: api.stopEngine,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engine'] }),
  })
  return { start: startMutation, stop: stopMutation }
}

export function useEquityHistory(days = 30, market = 'US') {
  return useQuery({
    queryKey: ['portfolio', 'equity-history', days, market],
    queryFn: () => api.fetchEquityHistory(days, market),
    refetchInterval: 300_000,
  })
}

export function useBacktestStrategies() {
  return useQuery({
    queryKey: ['backtest', 'strategies'],
    queryFn: api.fetchBacktestStrategies,
  })
}

export function useETFStatus(market = 'US') {
  return useQuery({
    queryKey: ['engine', 'etf', market],
    queryFn: () => api.fetchETFStatus(market),
    refetchInterval: 30_000,
  })
}

export function useNewsSentiment() {
  return useQuery({
    queryKey: ['news', 'sentiment'],
    queryFn: api.fetchNewsSentiment,
    refetchInterval: 60_000,
  })
}

export function useMarketEvents(market: string = 'US') {
  return useQuery({
    queryKey: ['market', 'events', market],
    queryFn: () => api.fetchMarketEvents(market),
    refetchInterval: 300_000,
  })
}
