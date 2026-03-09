import { useETFStatus } from '../hooks/useApi'
import { useMarket } from '../contexts/MarketContext'
import MarketToggle from './MarketToggle'
import clsx from 'clsx'

const REGIME_COLORS: Record<string, string> = {
  strong_uptrend: 'text-green-400',
  uptrend: 'text-green-300',
  sideways: 'text-yellow-400',
  downtrend: 'text-red-400',
  strong_downtrend: 'text-red-500',
}

const REGIME_LABELS: Record<string, string> = {
  strong_uptrend: 'Strong Uptrend',
  uptrend: 'Uptrend',
  sideways: 'Sideways',
  downtrend: 'Downtrend',
  strong_downtrend: 'Strong Downtrend',
}

const REASON_LABELS: Record<string, string> = {
  regime_uptrend: 'Regime (Bull)',
  regime_downtrend: 'Regime (Bear)',
  sector_rotation: 'Sector Rotation',
}

export default function ETFPanel() {
  const { market } = useMarket()
  const { data, isLoading, error } = useETFStatus(market)

  if (isLoading) {
    return <p className="text-gray-500 text-sm">Loading ETF status...</p>
  }

  if (error) {
    return <p className="text-red-400 text-sm">Failed to load ETF status.</p>
  }

  if (!data || data.status === 'not_configured') {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold">ETF Engine</h2>
          <MarketToggle />
        </div>
        <p className="text-gray-500 text-sm">ETF engine is not configured.</p>
      </div>
    )
  }

  const positions = Object.entries(data.managed_positions || {})
  const regime = data.last_regime

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">ETF Engine ({market})</h2>

      {/* Regime & Top Sectors */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-2">Market Regime</h3>
          <p className={clsx('text-2xl font-bold', regime ? REGIME_COLORS[regime] ?? 'text-gray-300' : 'text-gray-500')}>
            {regime ? REGIME_LABELS[regime] ?? regime : 'Unknown'}
          </p>
        </div>

        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-2">Top Sectors</h3>
          {data.top_sectors && data.top_sectors.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {data.top_sectors.map(s => (
                <span key={s} className="bg-blue-900/40 text-blue-300 text-sm px-2.5 py-1 rounded">
                  {s}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No sector data</p>
          )}
        </div>
      </div>

      {/* Managed Positions */}
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Managed ETF Positions</h3>
        {positions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="text-left py-2 pr-4">Symbol</th>
                  <th className="text-left py-2 pr-4">Reason</th>
                  <th className="text-left py-2 pr-4">Sector</th>
                  <th className="text-right py-2">Hold Days</th>
                </tr>
              </thead>
              <tbody>
                {positions.map(([symbol, pos]) => {
                  const holdDays = Math.round(pos.hold_days * 10) / 10
                  const isOverLimit = holdDays > (data.risk_params?.max_hold_days ?? 10)
                  return (
                    <tr key={symbol} className="border-b border-gray-800/50">
                      <td className="py-2 pr-4 font-medium">{symbol}</td>
                      <td className="py-2 pr-4 text-gray-400">
                        {REASON_LABELS[pos.reason] ?? pos.reason}
                      </td>
                      <td className="py-2 pr-4 text-gray-400">{pos.sector || '—'}</td>
                      <td className={clsx('py-2 text-right', isOverLimit ? 'text-red-400' : 'text-gray-300')}>
                        {holdDays}d
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-sm">No managed ETF positions.</p>
        )}
      </div>

      {/* Risk Parameters & Bear Config */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Risk Parameters</h3>
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-gray-500">Max Hold Days</dt>
              <dd>{data.risk_params.max_hold_days}d</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Max Portfolio %</dt>
              <dd>{(data.risk_params.max_portfolio_pct * 100).toFixed(0)}%</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Max Single ETF %</dt>
              <dd>{(data.risk_params.max_single_etf_pct * 100).toFixed(0)}%</dd>
            </div>
          </dl>
        </div>

        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Bear Entry Config</h3>
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-gray-500">Min Distance</dt>
              <dd>{data.bear_config.min_distance_pct}%</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Min Confidence</dt>
              <dd>{(data.bear_config.min_confidence * 100).toFixed(0)}%</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Size Ratio</dt>
              <dd>{(data.bear_config.size_ratio * 100).toFixed(0)}%</dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  )
}
