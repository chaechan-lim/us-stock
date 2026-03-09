import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { useTrades } from '../hooks/useApi'
import { formatCurrency } from '../utils/format'

interface StrategyMetrics {
  strategy: string
  tradeCount: number
  wins: number
  losses: number
  winRate: number
  totalPnl: number
  avgPnl: number
}

export default function StrategyPerformance() {
  const currency = 'USD'
  const { data: trades, isLoading } = useTrades(200)

  const strategyMetrics = useMemo(() => {
    if (!trades || trades.length === 0) return []

    const grouped = new Map<string, { pnls: number[]; wins: number; losses: number }>()

    for (const t of trades) {
      const key = t.strategy || 'unknown'
      if (!grouped.has(key)) {
        grouped.set(key, { pnls: [], wins: 0, losses: 0 })
      }
      const g = grouped.get(key)!
      const pnl = t.pnl ?? 0
      g.pnls.push(pnl)
      if (pnl > 0) g.wins++
      else if (pnl < 0) g.losses++
    }

    const metrics: StrategyMetrics[] = []
    for (const [strategy, g] of grouped) {
      const tradeCount = g.pnls.length
      const totalPnl = g.pnls.reduce((sum, p) => sum + p, 0)
      metrics.push({
        strategy,
        tradeCount,
        wins: g.wins,
        losses: g.losses,
        winRate: tradeCount > 0 ? (g.wins / tradeCount) * 100 : 0,
        totalPnl,
        avgPnl: tradeCount > 0 ? totalPnl / tradeCount : 0,
      })
    }

    return metrics.sort((a, b) => b.totalPnl - a.totalPnl)
  }, [trades])

  const yTickFormatter = (v: number) => `$${v.toLocaleString()}`

  if (isLoading) {
    return <div className="text-gray-500">Loading trade data...</div>
  }

  if (!trades || trades.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg p-8 text-center">
        <p className="text-gray-500">No trades available for strategy analysis.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Bar Chart */}
      {strategyMetrics.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4">Strategy P&L Comparison</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={strategyMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="strategy"
                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                tickLine={{ stroke: '#4B5563' }}
              />
              <YAxis
                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                tickLine={{ stroke: '#4B5563' }}
                tickFormatter={yTickFormatter}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.5rem',
                  color: '#F9FAFB',
                }}
                formatter={(value: number) => [formatCurrency(value, currency), 'Total P&L']}
              />
              <Bar dataKey="totalPnl" radius={[4, 4, 0, 0]}>
                {strategyMetrics.map((entry, index) => (
                  <Cell
                    key={index}
                    fill={entry.totalPnl >= 0 ? '#22c55e' : '#ef4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Metrics Table */}
      <div className="bg-gray-900 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-3">Strategy Metrics</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-gray-400 border-b border-gray-800">
              <tr>
                <th className="text-left py-2">Strategy</th>
                <th className="text-right py-2">Trades</th>
                <th className="text-right py-2">Wins</th>
                <th className="text-right py-2">Losses</th>
                <th className="text-right py-2">Win Rate</th>
                <th className="text-right py-2">Total P&L</th>
                <th className="text-right py-2">Avg P&L</th>
              </tr>
            </thead>
            <tbody>
              {strategyMetrics.map(m => (
                <tr key={m.strategy} className="border-b border-gray-800/50">
                  <td className="py-2 font-medium">{m.strategy}</td>
                  <td className="text-right py-2">{m.tradeCount}</td>
                  <td className="text-right py-2 text-green-400">{m.wins}</td>
                  <td className="text-right py-2 text-red-400">{m.losses}</td>
                  <td className="text-right py-2">
                    <span className={m.winRate >= 50 ? 'text-green-400' : 'text-red-400'}>
                      {m.winRate.toFixed(1)}%
                    </span>
                  </td>
                  <td className="text-right py-2">
                    <span className={m.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {m.totalPnl >= 0 ? '+' : ''}{formatCurrency(m.totalPnl, currency)}
                    </span>
                  </td>
                  <td className="text-right py-2">
                    <span className={m.avgPnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {m.avgPnl >= 0 ? '+' : ''}{formatCurrency(m.avgPnl, currency)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
