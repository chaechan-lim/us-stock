import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useWatchlist } from '../hooks/useApi'
import { useMarket } from '../contexts/MarketContext'
import { runScan } from '../api/client'
import type { ScanResult } from '../types'
import clsx from 'clsx'

const US_POPULAR = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'CRM',
  'TQQQ', 'SOXL', 'SQQQ', 'SOXS', 'QQQ', 'SPY', 'IWM', 'XLK', 'XLF', 'XLE',
]

const KR_POPULAR = [
  '005930', '000660', '373220', '207940', '005380', '000270', '035420', '035720',
  '068270', '105560', '055550', '066570', '006400', '051910', '012330', '003550',
  '247540', '086520', '377300', '196170',
]

function gradeColor(grade: string) {
  switch (grade) {
    case 'A': return 'text-green-400 bg-green-900/40'
    case 'B': return 'text-blue-400 bg-blue-900/40'
    case 'C': return 'text-yellow-400 bg-yellow-900/40'
    case 'D': return 'text-orange-400 bg-orange-900/40'
    default:  return 'text-red-400 bg-red-900/40'
  }
}

export default function ScannerPanel() {
  const { market } = useMarket()
  const { data: watchlist } = useWatchlist(market)
  const [results, setResults] = useState<ScanResult[]>([])
  const [minGrade, setMinGrade] = useState('B')
  const [symbolSource, setSymbolSource] = useState<'popular' | 'watchlist'>('popular')

  const POPULAR_SYMBOLS = market === 'KR' ? KR_POPULAR : US_POPULAR

  const scanMutation = useMutation({
    mutationFn: () => {
      const symbols = symbolSource === 'watchlist'
        ? (watchlist?.symbols ?? [])
        : POPULAR_SYMBOLS
      return runScan(symbols, minGrade)
    },
    onSuccess: setResults,
  })

  const symbols = symbolSource === 'watchlist'
    ? (watchlist?.symbols ?? [])
    : POPULAR_SYMBOLS

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Stock Scanner</h2>

      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-400">Scan:</label>
          <select
            value={symbolSource}
            onChange={e => setSymbolSource(e.target.value as 'popular' | 'watchlist')}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm"
          >
            <option value="popular">Popular ({POPULAR_SYMBOLS.length})</option>
            <option value="watchlist">Watchlist ({watchlist?.symbols?.length ?? 0})</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-400">Min Grade:</label>
          <select
            value={minGrade}
            onChange={e => setMinGrade(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm"
          >
            {['A', 'B', 'C', 'D', 'F'].map(g => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </div>

        <button
          onClick={() => scanMutation.mutate()}
          disabled={scanMutation.isPending || symbols.length === 0}
          className="px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 rounded
                     transition-colors disabled:opacity-50"
        >
          {scanMutation.isPending ? 'Scanning...' : `Run Scan (${symbols.length} symbols)`}
        </button>
      </div>

      {scanMutation.isError && (
        <div className="text-red-400 text-sm">
          Scan failed: {(scanMutation.error as Error).message}
        </div>
      )}

      {results.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium">Results ({results.length})</h3>
          </div>
          <table className="w-full text-sm">
            <thead className="text-gray-400 border-b border-gray-700">
              <tr>
                <th className="text-left py-2 px-3">Symbol</th>
                <th className="text-center py-2 px-3">Grade</th>
                <th className="text-right py-2 px-3">Total</th>
                <th className="text-right py-2 px-3">Trend</th>
                <th className="text-right py-2 px-3">Momentum</th>
                <th className="text-right py-2 px-3">Vol/Volume</th>
                <th className="text-right py-2 px-3">S/R</th>
              </tr>
            </thead>
            <tbody>
              {results.map(r => (
                <tr key={r.symbol} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="py-2 px-3 font-medium">{r.symbol}</td>
                  <td className="py-2 px-3 text-center">
                    <span className={clsx('px-2 py-0.5 rounded text-xs font-bold', gradeColor(r.grade))}>
                      {r.grade}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-right">{r.total_score.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right text-gray-300">{r.trend_score.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right text-gray-300">{r.momentum_score.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right text-gray-300">{r.volatility_volume_score.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right text-gray-300">{r.support_resistance_score.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!scanMutation.isPending && results.length === 0 && (
        <p className="text-gray-500 text-sm">Run a scan to find trading candidates.</p>
      )}
    </div>
  )
}
