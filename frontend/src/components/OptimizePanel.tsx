import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import * as api from '../api/client'
import { useBacktestStrategies } from '../hooks/useApi'
import { useMarket } from '../contexts/MarketContext'
import MarketToggle from './MarketToggle'

interface PerSymbolResult {
  symbol: string
  sharpe: number
  cagr: number
  max_drawdown?: number
}

interface OptimizeResult {
  strategy: string
  best_params: Record<string, unknown>
  avg_sharpe: number
  avg_cagr: number
  grid_combos: number
  elapsed: number
  per_symbol?: PerSymbolResult[]
}

export default function OptimizePanel() {
  useMarket() // keep context for MarketToggle
  const { data: strategies, isLoading: loadingStrategies } = useBacktestStrategies()
  const [strategyName, setStrategyName] = useState('')

  const optimize = useMutation({
    mutationFn: () =>
      api.runOptimization({
        strategy_name: strategyName || undefined,
      }),
  })

  const results: OptimizeResult[] | null = optimize.data
    ? Array.isArray(optimize.data)
      ? optimize.data
      : [optimize.data]
    : null

  return (
    <div className="space-y-6">
      {/* Run Optimization */}
      <div className="bg-gray-900 rounded-lg p-4">
        <div className="flex items-center gap-3 mb-4">
          <h2 className="text-lg font-semibold">Run Optimization</h2>
          <MarketToggle />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
              <option value="">All strategies</option>
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

          {/* Spacer */}
          <div />

          {/* Run Button */}
          <div className="flex items-end">
            <button
              onClick={() => optimize.mutate()}
              disabled={optimize.isPending}
              className={clsx(
                'w-full px-4 py-2 rounded text-sm font-medium transition-colors',
                optimize.isPending
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-500'
              )}
            >
              {optimize.isPending ? 'Optimizing...' : 'Run Optimize'}
            </button>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {optimize.isPending && (
        <div className="bg-gray-900 rounded-lg p-8 text-center">
          <div className="inline-block w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-3" />
          <p className="text-gray-400">
            Running optimization{strategyName ? ` for ${strategyName}` : ' for all strategies'}...
          </p>
          <p className="text-gray-500 text-xs mt-2">This may take several minutes.</p>
        </div>
      )}

      {/* Error */}
      {optimize.isError && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
          <p className="text-red-400 text-sm">
            Optimization failed: {optimize.error instanceof Error ? optimize.error.message : 'Unknown error'}
          </p>
        </div>
      )}

      {/* Results */}
      {results && !optimize.isPending && (
        <OptimizeResults results={results} />
      )}

      {/* Param Grids Viewer */}
      <ParamGridsViewer />
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Optimization Results                                               */
/* ------------------------------------------------------------------ */

function OptimizeResults({ results }: { results: OptimizeResult[] }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null)

  if (results.length === 1) {
    const r = results[0]
    return (
      <div className="bg-gray-900 rounded-lg p-4 space-y-4">
        <h2 className="text-lg font-semibold">
          Optimization Result
          <span className="text-sm text-gray-400 font-normal ml-2">{r.strategy}</span>
        </h2>

        {/* Metric cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard title="Avg Sharpe" value={r.avg_sharpe.toFixed(3)} positive={r.avg_sharpe >= 1} />
          <MetricCard title="Avg CAGR" value={`${(r.avg_cagr * 100).toFixed(2)}%`} positive={r.avg_cagr >= 0.12} />
          <MetricCard title="Grid Combos" value={String(r.grid_combos)} />
          <MetricCard title="Elapsed" value={formatElapsed(r.elapsed)} />
        </div>

        {/* Best params */}
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">Best Parameters</h3>
          <ParamTable params={r.best_params} />
        </div>

        {/* Per-symbol results */}
        {r.per_symbol && r.per_symbol.length > 0 && (
          <PerSymbolTable rows={r.per_symbol} />
        )}
      </div>
    )
  }

  // Multiple strategy results (full optimization)
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-4">Optimization Results</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-gray-400 border-b border-gray-800">
            <tr>
              <th className="text-left py-2">Strategy</th>
              <th className="text-right py-2">Avg Sharpe</th>
              <th className="text-right py-2">Avg CAGR</th>
              <th className="text-right py-2">Grid Combos</th>
              <th className="text-right py-2">Elapsed</th>
              <th className="text-left py-2 pl-4">Best Params</th>
              <th className="text-center py-2">Details</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, idx) => (
              <>
                <tr key={r.strategy} className="border-b border-gray-800/50">
                  <td className="py-2 font-medium">{r.strategy}</td>
                  <td className={clsx('text-right py-2', r.avg_sharpe >= 1 ? 'text-green-400' : 'text-red-400')}>
                    {r.avg_sharpe.toFixed(3)}
                  </td>
                  <td className={clsx('text-right py-2', r.avg_cagr >= 0.12 ? 'text-green-400' : 'text-red-400')}>
                    {(r.avg_cagr * 100).toFixed(2)}%
                  </td>
                  <td className="text-right py-2">{r.grid_combos}</td>
                  <td className="text-right py-2 text-gray-400">{formatElapsed(r.elapsed)}</td>
                  <td className="py-2 pl-4">
                    <span className="text-xs text-gray-300 font-mono">
                      {Object.entries(r.best_params)
                        .map(([k, v]) => `${k}=${String(v)}`)
                        .join(', ')}
                    </span>
                  </td>
                  <td className="text-center py-2">
                    {r.per_symbol && r.per_symbol.length > 0 && (
                      <button
                        onClick={() => setExpandedIdx(expandedIdx === idx ? null : idx)}
                        className="px-2 py-1 text-xs text-blue-400 hover:text-blue-300 hover:bg-blue-900/30 rounded transition-colors"
                      >
                        {expandedIdx === idx ? 'Hide' : 'Show'}
                      </button>
                    )}
                  </td>
                </tr>
                {expandedIdx === idx && r.per_symbol && (
                  <tr key={`${r.strategy}-detail`}>
                    <td colSpan={7} className="py-2 px-4">
                      <PerSymbolTable rows={r.per_symbol} />
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

function ParamTable({ params }: { params: Record<string, unknown> }) {
  const entries = Object.entries(params)
  if (entries.length === 0) return <p className="text-gray-500 text-sm">No parameters</p>

  return (
    <div className="overflow-x-auto">
      <table className="text-sm">
        <thead className="text-gray-400 border-b border-gray-800">
          <tr>
            {entries.map(([key]) => (
              <th key={key} className="text-left py-1 pr-6 font-medium">{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            {entries.map(([key, val]) => (
              <td key={key} className="py-1 pr-6 font-mono text-blue-400">{String(val)}</td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  )
}

function PerSymbolTable({ rows }: { rows: PerSymbolResult[] }) {
  return (
    <div className="mt-2">
      <h4 className="text-xs text-gray-400 uppercase tracking-wide mb-1">Per-Symbol Results</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-gray-400 border-b border-gray-800">
            <tr>
              <th className="text-left py-1">Symbol</th>
              <th className="text-right py-1">Sharpe</th>
              <th className="text-right py-1">CAGR</th>
              {rows[0]?.max_drawdown !== undefined && (
                <th className="text-right py-1">Max DD</th>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.symbol} className="border-b border-gray-800/30">
                <td className="py-1 font-medium">{row.symbol}</td>
                <td className={clsx('text-right py-1', row.sharpe >= 1 ? 'text-green-400' : 'text-red-400')}>
                  {row.sharpe.toFixed(3)}
                </td>
                <td className={clsx('text-right py-1', row.cagr >= 0.12 ? 'text-green-400' : 'text-red-400')}>
                  {(row.cagr * 100).toFixed(2)}%
                </td>
                {row.max_drawdown !== undefined && (
                  <td className="text-right py-1 text-red-400">
                    {(row.max_drawdown * 100).toFixed(2)}%
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
    <div className="bg-gray-800 rounded-lg p-3">
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

/* ------------------------------------------------------------------ */
/*  Param Grids Viewer                                                 */
/* ------------------------------------------------------------------ */

function ParamGridsViewer() {
  const { data: grids, isLoading, isError } = useQuery({
    queryKey: ['param-grids'],
    queryFn: api.fetchParamGrids,
  })

  if (isLoading) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-3">Parameter Grids</h2>
        <p className="text-gray-500 text-sm">Loading parameter grids...</p>
      </div>
    )
  }

  if (isError || !grids) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-3">Parameter Grids</h2>
        <p className="text-gray-500 text-sm">Failed to load parameter grids.</p>
      </div>
    )
  }

  const gridEntries = Object.entries(grids as Record<string, Record<string, unknown[]>>)

  if (gridEntries.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-3">Parameter Grids</h2>
        <p className="text-gray-500 text-sm">No parameter grids defined.</p>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-4">Parameter Grids</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {gridEntries.map(([strategy, params]) => (
          <div key={strategy} className="bg-gray-800 rounded-lg p-3">
            <h3 className="text-sm font-medium text-blue-400 mb-2">{strategy}</h3>
            <div className="space-y-1">
              {Object.entries(params).map(([param, values]) => (
                <div key={param} className="flex items-start gap-2 text-xs">
                  <span className="text-gray-400 font-mono whitespace-nowrap">{param}:</span>
                  <span className="text-gray-300 font-mono break-all">
                    [{Array.isArray(values) ? values.map(String).join(', ') : String(values)}]
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}m ${secs.toFixed(0)}s`
}
