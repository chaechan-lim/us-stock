import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { usePortfolioSummary, usePositions, useEngineStatus } from '../hooks/useApi'
import { usePriceStream } from '../hooks/usePriceStream'
import { fetchMacroIndicators, fetchMarketState, fetchTradeSummaryPeriods } from '../api/client'
import { formatCurrency } from '../utils/format'

function PnLText({ value, currency }: { value: number; currency: string }) {
  const color = value >= 0 ? 'text-green-400' : 'text-red-400'
  const sign = value >= 0 ? '+' : ''
  return <span className={color}>{sign}{formatCurrency(value, currency)}</span>
}

function PctText({ value }: { value: number }) {
  const color = value >= 0 ? 'text-green-400' : 'text-red-400'
  const sign = value >= 0 ? '+' : ''
  return <span className={color}>{sign}{value.toFixed(2)}%</span>
}

export default function Dashboard() {
  const { data: summary, isLoading } = usePortfolioSummary()
  const { data: positions } = usePositions()
  const { data: engineStatus } = useEngineStatus()
  const { data: usTradeSummary } = useQuery({
    queryKey: ['portfolio', 'trade-summary', 'US'],
    queryFn: () => fetchTradeSummaryPeriods('US'),
    refetchInterval: 60_000,
  })
  const { data: krTradeSummary } = useQuery({
    queryKey: ['portfolio', 'trade-summary', 'KR'],
    queryFn: () => fetchTradeSummaryPeriods('KR'),
    refetchInterval: 60_000,
  })
  const symbols = useMemo(
    () => (positions ?? []).map(p => p.symbol),
    [positions],
  )
  const { prices, connected } = usePriceStream(symbols)

  // Determine active markets from engine status
  const usPhase = engineStatus?.market_phase ?? 'closed'
  const krPhase = engineStatus?.kr_market_phase ?? 'closed'
  const usActive = usPhase === 'regular'
  const krActive = krPhase === 'regular'

  // Sort positions: active market first, then by absolute P&L descending
  const sortedPositions = useMemo(() => {
    if (!positions) return []
    return [...positions].sort((a, b) => {
      const aMkt = (a as { market?: string }).market ?? 'US'
      const bMkt = (b as { market?: string }).market ?? 'US'
      const aActive = (aMkt === 'KR' && krActive) || (aMkt === 'US' && usActive)
      const bActive = (bMkt === 'KR' && krActive) || (bMkt === 'US' && usActive)
      if (aActive !== bActive) return aActive ? -1 : 1
      // Within same group, sort by absolute unrealized P&L descending
      return Math.abs(b.unrealized_pnl ?? 0) - Math.abs(a.unrealized_pnl ?? 0)
    })
  }, [positions, usActive, krActive])

  if (isLoading || !summary) {
    return <div className="text-gray-500">Loading...</div>
  }

  const hasUsd = summary.usd_balance && summary.usd_balance.total > 0
  const rate = summary.exchange_rate ?? 1450
  // Fallback: KR total + USD total * rate (backend always provides total_equity,
  // but keep fallback for robustness). USD total already includes cash + positions.
  const totalEquity = summary.total_equity ??
    (summary.balance.total + (summary.usd_balance?.total ?? 0) * rate)

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card
          title="Total Equity"
          value={formatCurrency(totalEquity, 'KRW')}
          sub={hasUsd
            ? `KRW ${formatCurrency(summary.balance.total, 'KRW')} + USD ${formatCurrency(summary.usd_balance!.total, 'USD')} (₩${rate.toFixed(0)})`
            : undefined
          }
        />
        <Card
          title="Available Cash"
          value={formatCurrency(summary.available_cash ?? summary.balance.available, 'KRW')}
          sub={hasUsd
            ? `KRW ${formatCurrency(summary.balance.available, 'KRW')} / USD ${formatCurrency(summary.usd_balance!.available, 'USD')}`
            : undefined
          }
        />
        <Card title="Positions" value={String(summary.positions_count)} />
        <Card
          title="Unrealized P&L"
          value={
            <>
              <PnLText value={summary.total_unrealized_pnl} currency="KRW" />
              {(summary.total_unrealized_pnl_usd ?? 0) !== 0 && (
                <span className="text-sm ml-1">
                  / <PnLText value={summary.total_unrealized_pnl_usd!} currency="USD" />
                </span>
              )}
            </>
          }
          sub={summary.total_unrealized_pnl_pct != null
            ? `${summary.total_unrealized_pnl_pct >= 0 ? '+' : ''}${summary.total_unrealized_pnl_pct.toFixed(2)}%`
            : undefined
          }
          subClassName={summary.total_unrealized_pnl_pct != null
            ? (summary.total_unrealized_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400')
            : undefined
          }
        />
      </div>

      {/* Realized P&L by Period */}
      {(usTradeSummary || krTradeSummary) && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <PnLCard label="Today" us={usTradeSummary?.today} kr={krTradeSummary?.today} />
          <PnLCard label="This Week" us={usTradeSummary?.week} kr={krTradeSummary?.week} />
          <PnLCard label="This Month" us={usTradeSummary?.month} kr={krTradeSummary?.month} />
          <PnLCard label="All Time" us={usTradeSummary?.all_time} kr={krTradeSummary?.all_time} />
        </div>
      )}

      {/* All Positions (US + KR) — active market first */}
      {sortedPositions.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Holdings</h2>
            <div className="flex items-center gap-3">
              {krActive && (
                <span className="text-xs text-purple-400 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
                  KR Open
                </span>
              )}
              {usActive && (
                <span className="text-xs text-blue-400 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                  US Open
                </span>
              )}
              {connected && (
                <span className="text-xs text-green-500 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                  Live
                </span>
              )}
            </div>
          </div>
          <div className="overflow-x-auto -mx-4 px-4">
          <table className="w-full text-sm min-w-[700px]">
            <thead className="text-gray-400 border-b border-gray-800">
              <tr>
                <th className="text-left py-2">Symbol</th>
                <th className="text-center py-2">Mkt</th>
                <th className="text-right py-2">Qty</th>
                <th className="text-right py-2">Avg Price</th>
                <th className="text-right py-2">Current</th>
                <th className="text-right py-2">P&L</th>
                <th className="text-right py-2">P&L %</th>
                <th className="text-right py-2">SL / TP</th>
              </tr>
            </thead>
            <tbody>
              {sortedPositions.map(p => {
                const mkt = (p as { market?: string }).market ?? 'US'
                const cur = mkt === 'KR' ? 'KRW' : 'USD'
                const live = prices[p.symbol]
                const currentPrice = live?.price ?? p.current_price
                const pnl = (currentPrice - p.avg_price) * p.quantity
                const pnlPct = p.avg_price > 0
                  ? ((currentPrice - p.avg_price) / p.avg_price) * 100
                  : 0
                const isActive = (mkt === 'KR' && krActive) || (mkt === 'US' && usActive)
                const ext = p as { stop_loss_pct?: number; take_profit_pct?: number; trailing_active?: boolean }
                const slPct = ext.stop_loss_pct ?? 0.08
                const tpPct = ext.take_profit_pct ?? 0.20
                return (
                  <tr key={p.symbol} className={`border-b border-gray-800/50 ${isActive ? '' : 'opacity-60'}`}>
                    <td className="py-2 font-medium">
                      {p.symbol}
                      {p.name && <span className="text-gray-500 text-xs ml-1">{p.name}</span>}
                    </td>
                    <td className="py-2 text-center">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${mkt === 'KR' ? 'bg-purple-900/40 text-purple-300' : 'bg-blue-900/40 text-blue-300'}`}>
                        {mkt}
                      </span>
                    </td>
                    <td className="text-right">{p.quantity}</td>
                    <td className="text-right">{formatCurrency(p.avg_price, cur)}</td>
                    <td className="text-right">{formatCurrency(currentPrice, cur)}</td>
                    <td className="text-right">
                      <PnLText value={pnl} currency={cur} />
                    </td>
                    <td className="text-right">
                      <PctText value={pnlPct} />
                    </td>
                    <td className="text-right text-xs text-gray-500">
                      <span className="text-red-400/70">-{(slPct * 100).toFixed(0)}%</span>
                      {' / '}
                      <span className="text-green-400/70">+{(tpPct * 100).toFixed(0)}%</span>
                      {ext.trailing_active && (
                        <span className="ml-1 text-yellow-400/70" title="Trailing stop active">T</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          </div>
        </div>
      )}

      {/* Market State & Macro Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MarketStateCard />
        <MacroIndicatorsCard />
      </div>
    </div>
  )
}

function MarketStateCard() {
  const { data: marketState, isLoading } = useQuery({
    queryKey: ['engine', 'market-state'],
    queryFn: fetchMarketState,
    refetchInterval: 60_000,
  })

  const phaseColor = (phase: string) => {
    if (phase === 'regular') return 'text-green-400'
    if (phase === 'pre_market') return 'text-blue-400'
    if (phase === 'after_hours') return 'text-orange-400'
    return 'text-gray-400'
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">Market State</h2>
      {isLoading && <p className="text-gray-500 text-sm">Loading...</p>}
      {marketState && (
        <div className="space-y-3">
          {/* US Market */}
          <div>
            <div className="text-xs text-blue-400 font-medium mb-1">US Market</div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-500">Phase</span>
                <p className={`font-medium ${phaseColor(marketState.market_phase ?? '')}`}>
                  {marketState.market_phase ?? '-'}
                </p>
              </div>
              <div>
                <span className="text-gray-500">Regime</span>
                <p className="text-white font-medium">{marketState.regime ?? '-'}</p>
              </div>
              <div>
                <span className="text-gray-500">SPY</span>
                <p className="text-white font-medium">
                  {marketState.spy_price != null ? `$${Number(marketState.spy_price).toFixed(2)}` : '-'}
                </p>
              </div>
              <div>
                <span className="text-gray-500">VIX</span>
                <p className="text-white font-medium">
                  {marketState.vix_level != null ? Number(marketState.vix_level).toFixed(2) : '-'}
                </p>
              </div>
            </div>
          </div>
          {/* KR Market */}
          <div className="border-t border-gray-800 pt-2">
            <div className="text-xs text-purple-400 font-medium mb-1">KR Market</div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-500">Phase</span>
                <p className={`font-medium ${phaseColor(marketState.kr_market_phase ?? '')}`}>
                  {marketState.kr_market_phase ?? '-'}
                </p>
              </div>
              <div>
                <span className="text-gray-500">Regime</span>
                <p className="text-white font-medium">{marketState.kr_regime ?? '-'}</p>
              </div>
              {marketState.kr_index_price != null && (
                <div>
                  <span className="text-gray-500">KODEX 200</span>
                  <p className="text-white font-medium">
                    ₩{Number(marketState.kr_index_price).toLocaleString()}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function MacroIndicatorsCard() {
  const { data: macro, isLoading } = useQuery({
    queryKey: ['engine', 'macro'],
    queryFn: fetchMacroIndicators,
    refetchInterval: 60_000,
  })

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">Macro Indicators</h2>
      {isLoading && <p className="text-gray-500 text-sm">Loading...</p>}
      {macro && (
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-500">Fed Funds Rate</span>
            <p className="text-white font-medium">
              {macro.fed_funds_rate != null ? `${Number(macro.fed_funds_rate).toFixed(2)}%` : '-'}
            </p>
          </div>
          <div>
            <span className="text-gray-500">Treasury 10Y</span>
            <p className="text-white font-medium">
              {macro.treasury_10y != null ? `${Number(macro.treasury_10y).toFixed(2)}%` : '-'}
            </p>
          </div>
          <div>
            <span className="text-gray-500">Yield Spread</span>
            <p className="text-white font-medium">
              {macro.yield_spread != null ? `${Number(macro.yield_spread).toFixed(2)}%` : '-'}
            </p>
          </div>
          <div>
            <span className="text-gray-500">CPI YoY</span>
            <p className="text-white font-medium">
              {macro.cpi_yoy != null ? `${Number(macro.cpi_yoy).toFixed(2)}%` : '-'}
            </p>
          </div>
          <div className="col-span-2">
            <span className="text-gray-500">Unemployment Rate</span>
            <p className="text-white font-medium">
              {macro.unemployment_rate != null ? `${Number(macro.unemployment_rate).toFixed(1)}%` : '-'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

function Card({ title, value, sub, subClassName }: { title: string; value: React.ReactNode; sub?: string; subClassName?: string }) {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <div className="text-xs text-gray-400 uppercase tracking-wide">{title}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {sub && <div className={`text-xs mt-0.5 ${subClassName ?? 'text-gray-500'}`}>{sub}</div>}
    </div>
  )
}

interface PeriodData { pnl: number; pnl_pct: number | null; trades: number; wins: number; losses: number; win_rate: number }

function PnLLine({ pnl, pnlPct, currency, trades, wins, losses }: { pnl: number; pnlPct?: number | null; currency: string; trades: number; wins: number; losses: number }) {
  const color = pnl >= 0 ? 'text-green-400' : 'text-red-400'
  const sign = pnl >= 0 ? '+' : ''
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className={`text-lg font-bold ${color}`}>{sign}{formatCurrency(pnl, currency)}</span>
        <span className="text-xs text-gray-500">{trades}T {wins}W/{losses}L</span>
      </div>
      {pnlPct != null && (
        <div className={`text-xs ${pnlPct >= 0 ? 'text-green-400/70' : 'text-red-400/70'}`}>
          avg {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%
        </div>
      )}
    </div>
  )
}

function PnLCard({ label, us, kr }: { label: string; us?: PeriodData; kr?: PeriodData }) {
  const hasKr = (kr?.trades ?? 0) > 0
  const hasUs = (us?.trades ?? 0) > 0

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <div className="text-xs text-gray-400 uppercase tracking-wide mb-2">{label}</div>
      {!hasKr && !hasUs ? (
        <div className="text-lg text-gray-600">—</div>
      ) : (
        <div className="space-y-1">
          {hasKr && (
            <div>
              <div className="text-[10px] text-purple-400 mb-0.5">KR</div>
              <PnLLine pnl={kr!.pnl} pnlPct={kr!.pnl_pct} currency="KRW" trades={kr!.trades} wins={kr!.wins} losses={kr!.losses} />
            </div>
          )}
          {hasUs && (
            <div>
              <div className="text-[10px] text-blue-400 mb-0.5">US</div>
              <PnLLine pnl={us!.pnl} pnlPct={us!.pnl_pct} currency="USD" trades={us!.trades} wins={us!.wins} losses={us!.losses} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
