import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import clsx from 'clsx'
import * as api from '../api/client'
import { useBacktestStrategies } from '../hooks/useApi'

interface BacktestMetrics {
  total_return_pct: number
  sharpe_ratio: number
  max_drawdown_pct: number
  win_rate: number
  total_trades: number
  profit_factor: number
}

interface BacktestTrade {
  entry_date: string
  exit_date: string
  side: string
  pnl: number
  pnl_pct: number
}

interface EquityPoint {
  date: string
  equity: number
}

interface BacktestResult {
  metrics: BacktestMetrics
  equity_curve: EquityPoint[]
  trades: BacktestTrade[]
}

const PERIODS = ['1y', '2y', '3y', '5y'] as const

function formatUSD(n: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(n)
}

function formatPct(n: number, decimals = 2) {
  const sign = n >= 0 ? '+' : ''
  return `${sign}${n.toFixed(decimals)}%`
}

export default function BacktestPanel() {
  const { data: strategies, isLoading: loadingStrategies } = useBacktestStrategies()
  const [strategyName, setStrategyName] = useState('')
  const [symbol, setSymbol] = useState('')
  const [period, setPeriod] = useState<string>('1y')

  const backtest = useMutation({
    mutationFn: () =>
      api.runBacktest({
        strategy_name: strategyName,
        symbol: symbol.toUpperCase(),
        period,
      }) as Promise<BacktestResult>,
  })

  const result = backtest.data

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-gray-900 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">Backtest Configuration</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Strategy Selector */}
          <div>
            <label className="block text-xs text-gray-400 uppercase tracking-wide mb-1">
              Strategy
            </label>
            <select
              value={strategyName}
              onChange={e => setStrategyName(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            >
              <option value="">Select strategy...</option>
              {(strategies ?? []).map((s: { name: string; display_name?: string }) => (
                <option key={s.name} value={s.name}>
                  {s.display_name ?? s.name}
                </option>
              ))}
            </select>
            {loadingStrategies && (
              <span className="text-xs text-gray-500 mt-1">Loading...</span>
            )}
          </div>

          {/* Symbol Input */}
          <div>
            <label className="block text-xs text-gray-400 uppercase tracking-wide mb-1">
              Symbol
            </label>
            <input
              type="text"
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
              placeholder="e.g. AAPL"
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Period Selector */}
          <div>
            <label className="block text-xs text-gray-400 uppercase tracking-wide mb-1">
              Period
            </label>
            <div className="flex gap-1">
              {PERIODS.map(p => (
                <button
                  key={p}
                  onClick={() => setPeriod(p)}
                  className={clsx(
                    'px-3 py-2 text-sm rounded transition-colors',
                    period === p
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
                  )}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          {/* Run Button */}
          <div className="flex items-end">
            <button
              onClick={() => backtest.mutate()}
              disabled={!strategyName || !symbol || backtest.isPending}
              className={clsx(
                'w-full px-4 py-2 rounded text-sm font-medium transition-colors',
                !strategyName || !symbol || backtest.isPending
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-500'
              )}
            >
              {backtest.isPending ? 'Running...' : 'Run Backtest'}
            </button>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {backtest.isPending && (
        <div className="bg-gray-900 rounded-lg p-8 text-center">
          <div className="inline-block w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-3" />
          <p className="text-gray-400">Running backtest for {symbol.toUpperCase()} with {strategyName}...</p>
        </div>
      )}

      {/* Error */}
      {backtest.isError && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
          <p className="text-red-400 text-sm">
            Backtest failed: {backtest.error instanceof Error ? backtest.error.message : 'Unknown error'}
          </p>
        </div>
      )}

      {/* Results */}
      {result && !backtest.isPending && (
        <>
          {/* Metrics Cards */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <MetricCard
              title="Total Return"
              value={formatPct(result.metrics.total_return_pct)}
              positive={result.metrics.total_return_pct >= 0}
            />
            <MetricCard
              title="Sharpe Ratio"
              value={result.metrics.sharpe_ratio.toFixed(2)}
              positive={result.metrics.sharpe_ratio >= 1}
            />
            <MetricCard
              title="Max Drawdown"
              value={formatPct(-Math.abs(result.metrics.max_drawdown_pct))}
              positive={Math.abs(result.metrics.max_drawdown_pct) < 25}
            />
            <MetricCard
              title="Win Rate"
              value={formatPct(result.metrics.win_rate, 1)}
              positive={result.metrics.win_rate >= 50}
            />
            <MetricCard
              title="Total Trades"
              value={String(result.metrics.total_trades)}
            />
            <MetricCard
              title="Profit Factor"
              value={result.metrics.profit_factor.toFixed(2)}
              positive={result.metrics.profit_factor >= 1}
            />
          </div>

          {/* Equity Curve */}
          {result.equity_curve && result.equity_curve.length > 0 && (
            <div className="bg-gray-900 rounded-lg p-4">
              <h2 className="text-lg font-semibold mb-4">Equity Curve</h2>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={result.equity_curve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#9CA3AF', fontSize: 11 }}
                    tickLine={{ stroke: '#4B5563' }}
                  />
                  <YAxis
                    tick={{ fill: '#9CA3AF', fontSize: 11 }}
                    tickLine={{ stroke: '#4B5563' }}
                    tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '0.5rem',
                      color: '#F9FAFB',
                    }}
                    formatter={(value: number) => [formatUSD(value), 'Equity']}
                  />
                  <Line
                    type="monotone"
                    dataKey="equity"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Trade List */}
          {result.trades && result.trades.length > 0 && (
            <div className="bg-gray-900 rounded-lg p-4">
              <h2 className="text-lg font-semibold mb-3">
                Recent Trades
                <span className="text-sm text-gray-400 font-normal ml-2">
                  (last {Math.min(result.trades.length, 20)})
                </span>
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-gray-400 border-b border-gray-800">
                    <tr>
                      <th className="text-left py-2">Entry Date</th>
                      <th className="text-left py-2">Exit Date</th>
                      <th className="text-left py-2">Side</th>
                      <th className="text-right py-2">P&L</th>
                      <th className="text-right py-2">P&L %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.trades.slice(-20).map((t, i) => (
                      <tr key={i} className="border-b border-gray-800/50">
                        <td className="py-2">{t.entry_date}</td>
                        <td className="py-2">{t.exit_date}</td>
                        <td className="py-2">
                          <span
                            className={clsx(
                              'px-2 py-0.5 rounded text-xs font-medium',
                              t.side === 'long'
                                ? 'bg-green-900/40 text-green-400'
                                : 'bg-red-900/40 text-red-400'
                            )}
                          >
                            {t.side.toUpperCase()}
                          </span>
                        </td>
                        <td className={clsx('text-right py-2', t.pnl >= 0 ? 'text-green-400' : 'text-red-400')}>
                          {t.pnl >= 0 ? '+' : ''}{formatUSD(t.pnl)}
                        </td>
                        <td className={clsx('text-right py-2', t.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400')}>
                          {formatPct(t.pnl_pct)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* Saved Results */}
      <SavedResults />
    </div>
  )
}

function SavedResults() {
  const queryClient = useQueryClient()
  const { data: savedResults, isLoading } = useQuery({
    queryKey: ['backtest-results'],
    queryFn: () => api.fetchBacktestResults(),
  })

  const deleteMutation = useMutation({
    mutationFn: api.deleteBacktestResult,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['backtest-results'] }),
  })

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Saved Results</h2>
        <button
          onClick={() => queryClient.invalidateQueries({ queryKey: ['backtest-results'] })}
          className="px-3 py-1 text-xs bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
        >
          Refresh
        </button>
      </div>

      {isLoading && <p className="text-gray-500 text-sm">Loading saved results...</p>}

      {savedResults && Array.isArray(savedResults) && savedResults.length === 0 && (
        <p className="text-gray-500 text-sm">No saved backtest results.</p>
      )}

      {savedResults && Array.isArray(savedResults) && savedResults.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-gray-400 border-b border-gray-800">
              <tr>
                <th className="text-left py-2">Strategy</th>
                <th className="text-left py-2">Symbol</th>
                <th className="text-left py-2">Period</th>
                <th className="text-right py-2">Return%</th>
                <th className="text-right py-2">Sharpe</th>
                <th className="text-right py-2">MDD%</th>
                <th className="text-right py-2">Trades</th>
                <th className="text-center py-2">Cached</th>
                <th className="text-right py-2"></th>
              </tr>
            </thead>
            <tbody>
              {savedResults.map((r: any) => (
                <tr key={r.key} className="border-b border-gray-800/50">
                  <td className="py-2 font-medium">{r.strategy ?? '-'}</td>
                  <td className="py-2">{r.symbol ?? '-'}</td>
                  <td className="py-2">{r.period ?? '-'}</td>
                  <td className={clsx(
                    'text-right py-2',
                    (r.total_return_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'
                  )}>
                    {r.total_return_pct != null ? formatPct(r.total_return_pct) : '-'}
                  </td>
                  <td className="text-right py-2">
                    {r.sharpe_ratio != null ? r.sharpe_ratio.toFixed(2) : '-'}
                  </td>
                  <td className="text-right py-2 text-red-400">
                    {r.max_drawdown_pct != null ? formatPct(-Math.abs(r.max_drawdown_pct)) : '-'}
                  </td>
                  <td className="text-right py-2">{r.total_trades ?? '-'}</td>
                  <td className="text-center py-2">
                    {r.cached && (
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-blue-900/40 text-blue-400">
                        cached
                      </span>
                    )}
                  </td>
                  <td className="text-right py-2">
                    <button
                      onClick={() => deleteMutation.mutate(r.key)}
                      disabled={deleteMutation.isPending}
                      className="px-2 py-1 text-xs text-red-400 hover:text-red-300 hover:bg-red-900/30 rounded transition-colors"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function MetricCard({
  title,
  value,
  positive,
}: {
  title: string
  value: string
  positive?: boolean
}) {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <div className="text-xs text-gray-400 uppercase tracking-wide">{title}</div>
      <div
        className={clsx(
          'text-xl font-bold mt-1',
          positive === undefined
            ? 'text-white'
            : positive
              ? 'text-green-400'
              : 'text-red-400'
        )}
      >
        {value}
      </div>
    </div>
  )
}
