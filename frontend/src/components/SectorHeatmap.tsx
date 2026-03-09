import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import * as api from '../api/client'
import { useMarket } from '../contexts/MarketContext'
import MarketToggle from './MarketToggle'

interface SectorData {
  sector: string
  etf_symbol: string
  return_1w: number
  return_1m: number
}

function getHeatColor(value: number): string {
  const clamped = Math.max(-10, Math.min(10, value))
  const intensity = Math.abs(clamped) / 10

  if (value >= 0) {
    // Green scale: from gray-800 to green-500
    const r = Math.round(31 - intensity * 9)    // 31 -> 22
    const g = Math.round(41 + intensity * 160)   // 41 -> 201
    const b = Math.round(55 - intensity * 15)    // 55 -> 40
    const a = 0.3 + intensity * 0.5
    return `rgba(${r}, ${g}, ${b}, ${a})`
  } else {
    // Red scale: from gray-800 to red-500
    const r = Math.round(31 + intensity * 208)   // 31 -> 239
    const g = Math.round(41 - intensity * 3)     // 41 -> 38
    const b = Math.round(55 - intensity * 27)    // 55 -> 28
    const a = 0.3 + intensity * 0.5
    return `rgba(${r}, ${g}, ${b}, ${a})`
  }
}

function formatPct(n: number) {
  const sign = n >= 0 ? '+' : ''
  return `${sign}${n.toFixed(2)}%`
}

export default function SectorHeatmap() {
  const { market } = useMarket()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['scanner', 'sectors', market],
    queryFn: api.fetchSectorPerformance,
    refetchInterval: 300_000,
    enabled: market === 'US',
  })

  if (market === 'KR') {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold">Sector Heatmap</h2>
          <MarketToggle />
        </div>
        <div className="bg-gray-900 rounded-lg p-8 text-center">
          <p className="text-gray-500">Sector heatmap is available for US market only.</p>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return <div className="text-gray-500">Loading sector data...</div>
  }

  if (isError) {
    return (
      <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
        <p className="text-red-400 text-sm">Failed to load sector performance.</p>
      </div>
    )
  }

  const sectors: SectorData[] = (data ?? [])
    .slice()
    .sort((a: SectorData, b: SectorData) => b.return_1m - a.return_1m)

  return (
    <div className="space-y-6">
      <div className="bg-gray-900 rounded-lg p-4">
        <div className="flex items-center gap-3 mb-4">
          <h2 className="text-lg font-semibold">Sector Performance</h2>
          <MarketToggle />
        </div>

        {sectors.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No sector data available.</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
            {sectors.map(s => (
              <div
                key={s.etf_symbol}
                className="rounded-lg p-4 border border-gray-800 transition-transform hover:scale-105"
                style={{ backgroundColor: getHeatColor(s.return_1m) }}
              >
                <div className="font-semibold text-white text-sm">{s.sector}</div>
                <div className="text-xs text-gray-300 mt-0.5">{s.etf_symbol}</div>
                <div className="mt-3 space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-300">1W</span>
                    <span
                      className={clsx(
                        'font-medium',
                        s.return_1w >= 0 ? 'text-green-300' : 'text-red-300'
                      )}
                    >
                      {formatPct(s.return_1w)}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-300">1M</span>
                    <span
                      className={clsx(
                        'font-medium',
                        s.return_1m >= 0 ? 'text-green-300' : 'text-red-300'
                      )}
                    >
                      {formatPct(s.return_1m)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Legend */}
        <div className="mt-4 flex items-center justify-center gap-2 text-xs text-gray-400">
          <span className="text-red-400">Negative</span>
          <div className="flex gap-0.5">
            {[-8, -5, -2, 0, 2, 5, 8].map(v => (
              <div
                key={v}
                className="w-6 h-3 rounded-sm"
                style={{ backgroundColor: getHeatColor(v) }}
              />
            ))}
          </div>
          <span className="text-green-400">Positive</span>
        </div>
      </div>
    </div>
  )
}
