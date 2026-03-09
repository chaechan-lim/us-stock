import { useState } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import clsx from 'clsx'
import { useEquityHistory } from '../hooks/useApi'
import { useMarket } from '../contexts/MarketContext'
import { formatCurrency } from '../utils/format'

interface EquityPoint {
  date: string
  total_value_usd: number
}

const PERIODS = [
  { label: '7d', days: 7 },
  { label: '30d', days: 30 },
  { label: '90d', days: 90 },
  { label: '365d', days: 365 },
] as const

export default function PortfolioChart() {
  const { market, currency } = useMarket()
  const [days, setDays] = useState(30)
  const { data, isLoading, isError } = useEquityHistory(days, market)

  const history: EquityPoint[] = data ?? []

  const currentValue = history.length > 0 ? history[history.length - 1].total_value_usd : 0
  const startValue = history.length > 0 ? history[0].total_value_usd : 0
  const changeAbs = currentValue - startValue
  const changePct = startValue > 0 ? (changeAbs / startValue) * 100 : 0
  const isPositive = changeAbs >= 0

  const yTickFormatter = currency === 'KRW'
    ? (v: number) => `${(v / 10000).toFixed(0)}만`
    : (v: number) => `$${(v / 1000).toFixed(0)}k`

  if (isLoading) {
    return <div className="text-gray-500">Loading equity history...</div>
  }

  if (isError) {
    return (
      <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
        <p className="text-red-400 text-sm">Failed to load equity history.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Value Summary + Period Selector */}
      <div className="bg-gray-900 rounded-lg p-4">
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="text-xs text-gray-400 uppercase tracking-wide">Portfolio Value</div>
            <div className="text-3xl font-bold mt-1">{formatCurrency(currentValue, currency)}</div>
            {history.length > 1 && (
              <div className={clsx('text-sm mt-1', isPositive ? 'text-green-400' : 'text-red-400')}>
                {isPositive ? '+' : ''}{formatCurrency(changeAbs, currency)}{' '}
                ({isPositive ? '+' : ''}{changePct.toFixed(2)}%)
                <span className="text-gray-500 ml-1">
                  past {days}d
                </span>
              </div>
            )}
          </div>
          <div className="flex gap-1">
            {PERIODS.map(p => (
              <button
                key={p.label}
                onClick={() => setDays(p.days)}
                className={clsx(
                  'px-3 py-1.5 text-sm rounded transition-colors',
                  days === p.days
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
                )}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Chart */}
        {history.length > 0 ? (
          <ResponsiveContainer width="100%" height={350}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="date"
                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                tickLine={{ stroke: '#4B5563' }}
              />
              <YAxis
                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                tickLine={{ stroke: '#4B5563' }}
                tickFormatter={yTickFormatter}
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.5rem',
                  color: '#F9FAFB',
                }}
                formatter={(value: number) => [formatCurrency(value, currency), 'Value']}
                labelFormatter={label => `Date: ${label}`}
              />
              <Area
                type="monotone"
                dataKey="total_value_usd"
                stroke="#3b82f6"
                strokeWidth={2}
                fill="url(#equityGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-gray-500 text-center py-12">
            No equity history data available for this period.
          </div>
        )}
      </div>
    </div>
  )
}
