import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { usePortfolioSummary, usePositions, useEngineStatus, usePortfolioReturns } from '../hooks/useApi'
import { usePriceStream } from '../hooks/usePriceStream'
import { fetchMacroIndicators, fetchMarketState, fetchTradeSummaryPeriods } from '../api/client'
import type { PeriodReturn } from '../api/client'
import { formatCurrency } from '../utils/format'
import { useAccount } from '../contexts/AccountContext'

function PnLText({ value, currency }: { value: number; currency: string }) {
  const color = value >= 0 ? 'text-emerald-600' : 'text-rose-600'
  const sign = value >= 0 ? '+' : ''
  return <span className={`font-medium ${color}`}>{sign}{formatCurrency(value, currency)}</span>
}

function PctBadge({ value }: { value: number }) {
  const bg = value >= 0 ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
  const sign = value >= 0 ? '+' : ''
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded-md text-[11px] font-semibold ${bg}`}>
      {sign}{value.toFixed(2)}%
    </span>
  )
}

export default function Dashboard() {
  const { selectedAccountId, selectedAccount } = useAccount()
  const { data: summary, isLoading } = usePortfolioSummary('ALL', selectedAccountId)
  const { data: positions } = usePositions('ALL', selectedAccountId)
  const { data: engineStatus } = useEngineStatus()
  const { data: returns } = usePortfolioReturns()
  const { data: usTradeSummary } = useQuery({
    queryKey: ['portfolio', 'trade-summary', 'US', selectedAccountId ?? 'all'],
    queryFn: () => fetchTradeSummaryPeriods('US', selectedAccountId),
    refetchInterval: 60_000,
  })
  const { data: krTradeSummary } = useQuery({
    queryKey: ['portfolio', 'trade-summary', 'KR', selectedAccountId ?? 'all'],
    queryFn: () => fetchTradeSummaryPeriods('KR', selectedAccountId),
    refetchInterval: 60_000,
  })
  const symbols = useMemo(() => (positions ?? []).map(p => p.symbol), [positions])
  const { prices, connected } = usePriceStream(symbols)

  const usPhase = engineStatus?.market_phase ?? 'closed'
  const krPhase = engineStatus?.kr_market_phase ?? 'closed'
  const usActive = usPhase === 'regular'
  const krActive = krPhase === 'regular'

  const sortedPositions = useMemo(() => {
    if (!positions) return []
    return [...positions].sort((a, b) => {
      const aMkt = (a as { market?: string }).market ?? 'US'
      const bMkt = (b as { market?: string }).market ?? 'US'
      const aActive = (aMkt === 'KR' && krActive) || (aMkt === 'US' && usActive)
      const bActive = (bMkt === 'KR' && krActive) || (bMkt === 'US' && usActive)
      if (aActive !== bActive) return aActive ? -1 : 1
      return Math.abs(b.unrealized_pnl ?? 0) - Math.abs(a.unrealized_pnl ?? 0)
    })
  }, [positions, usActive, krActive])

  if (isLoading || !summary) {
    return <div className="flex items-center justify-center h-40 text-gray-400">Loading...</div>
  }

  const hasUsd = summary.usd_balance && summary.usd_balance.total > 0
  const rate = summary.exchange_rate ?? 1450
  const totalEquity = summary.total_equity ??
    (summary.balance.total + (summary.usd_balance?.total ?? 0) * rate)

  return (
    <div className="min-h-screen bg-gray-50 -mx-4 -mt-4 px-4 pt-4 pb-8 sm:px-6">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 mb-5">
        <MarketPill label="US" phase={usPhase} />
        <MarketPill label="KR" phase={krPhase} />
        {connected && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-100 text-emerald-700">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            Live
          </span>
        )}
        {selectedAccount && (
          <span className="text-[11px] px-2 py-0.5 rounded-full font-medium bg-gray-200 text-gray-600">
            {selectedAccount.name}
          </span>
        )}
      </div>

      {/* Equity Hero Card */}
      <EquityCard
        totalEquity={totalEquity}
        hasUsd={!!hasUsd}
        krwTotal={summary.balance.total}
        usdTotal={summary.usd_balance?.total ?? 0}
        rate={rate}
        returns={returns}
      />

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-3 mt-4">
        <StatCard label="Cash" value={formatCurrency(summary.available_cash ?? summary.balance.available, 'KRW')} />
        <StatCard label="Positions" value={String(summary.positions_count)} />
        <StatCard
          label="Unrealized"
          value={(() => {
            const krw = summary.total_unrealized_pnl ?? 0
            const usd = summary.total_unrealized_pnl_usd ?? 0
            const combined = krw + usd * (summary.exchange_rate ?? 1450)
            const color = combined >= 0 ? 'text-emerald-600' : 'text-rose-600'
            return <span className={color}>{combined >= 0 ? '+' : ''}{formatCurrency(combined, 'KRW')}</span>
          })()}
          sub={
            <div className="flex items-center gap-1.5 flex-wrap">
              {summary.total_unrealized_pnl_pct != null && <PctBadge value={summary.total_unrealized_pnl_pct} />}
              {(summary.total_unrealized_pnl_usd ?? 0) !== 0 && (
                <span className={`text-[10px] ${(summary.total_unrealized_pnl_usd ?? 0) >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                  US {(summary.total_unrealized_pnl_usd ?? 0) >= 0 ? '+' : ''}{formatCurrency(summary.total_unrealized_pnl_usd!, 'USD')}
                </span>
              )}
            </div>
          }
        />
      </div>

      {/* Realized P&L */}
      {(usTradeSummary || krTradeSummary) && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          <PnLCard label="Today" us={usTradeSummary?.today} kr={krTradeSummary?.today} />
          <PnLCard label="Week" us={usTradeSummary?.week} kr={krTradeSummary?.week} />
          <PnLCard label="Month" us={usTradeSummary?.month} kr={krTradeSummary?.month} />
          <PnLCard label="All" us={usTradeSummary?.all_time} kr={krTradeSummary?.all_time} />
        </div>
      )}

      {/* Holdings */}
      {sortedPositions.length > 0 && (
        <div className="mt-4 bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-100">
            <h2 className="text-sm font-semibold text-gray-900">Holdings</h2>
          </div>

          {/* Mobile */}
          <div className="divide-y divide-gray-100 md:hidden">
            {sortedPositions.map(p => {
              const mkt = (p as { market?: string }).market ?? 'US'
              const cur = mkt === 'KR' ? 'KRW' : 'USD'
              const live = prices[p.symbol]
              const currentPrice = live?.price ?? p.current_price
              const pnl = (currentPrice - p.avg_price) * p.quantity
              const pnlPct = p.avg_price > 0 ? ((currentPrice - p.avg_price) / p.avg_price) * 100 : 0
              const isActive = (mkt === 'KR' && krActive) || (mkt === 'US' && usActive)
              const ext = p as { stop_loss_pct?: number; take_profit_pct?: number; trailing_active?: boolean }
              return (
                <div key={p.symbol} className={`px-4 py-3 ${isActive ? '' : 'opacity-40'}`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 min-w-0">
                      <MktTag mkt={mkt} />
                      <div className="min-w-0">
                        <div className="flex items-center gap-1">
                          <span className="font-semibold text-sm text-gray-900">{p.symbol}</span>
                          {ext.trailing_active && <span className="text-[10px] font-medium text-amber-600 bg-amber-50 px-1 rounded">T</span>}
                        </div>
                        {p.name && <div className="text-[11px] text-gray-400 truncate">{p.name}</div>}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-sm font-semibold ${pnl >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                        {pnl >= 0 ? '+' : ''}{formatCurrency(pnl, cur)}
                      </div>
                      <PctBadge value={pnlPct} />
                    </div>
                  </div>
                  <div className="flex items-center justify-between mt-1 text-[11px] text-gray-400">
                    <span>{p.quantity}주 · avg {formatCurrency(p.avg_price, cur)}</span>
                    <span>
                      SL <span className="text-rose-400">-{((ext.stop_loss_pct ?? 0.08) * 100).toFixed(0)}%</span>
                      {' / '}
                      TP <span className="text-emerald-400">+{((ext.take_profit_pct ?? 0.20) * 100).toFixed(0)}%</span>
                    </span>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Desktop */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 text-xs text-gray-500">
                  <th className="text-left py-2.5 px-4 font-medium">Symbol</th>
                  <th className="text-center py-2.5 font-medium">Mkt</th>
                  <th className="text-right py-2.5 font-medium">Qty</th>
                  <th className="text-right py-2.5 font-medium">Avg</th>
                  <th className="text-right py-2.5 font-medium">Now</th>
                  <th className="text-right py-2.5 font-medium">P&L</th>
                  <th className="text-right py-2.5 font-medium">%</th>
                  <th className="text-right py-2.5 px-4 font-medium">SL / TP</th>
                </tr>
              </thead>
              <tbody>
                {sortedPositions.map(p => {
                  const mkt = (p as { market?: string }).market ?? 'US'
                  const cur = mkt === 'KR' ? 'KRW' : 'USD'
                  const live = prices[p.symbol]
                  const currentPrice = live?.price ?? p.current_price
                  const pnl = (currentPrice - p.avg_price) * p.quantity
                  const pnlPct = p.avg_price > 0 ? ((currentPrice - p.avg_price) / p.avg_price) * 100 : 0
                  const isActive = (mkt === 'KR' && krActive) || (mkt === 'US' && usActive)
                  const ext = p as { stop_loss_pct?: number; take_profit_pct?: number; trailing_active?: boolean }
                  return (
                    <tr key={p.symbol} className={`border-b border-gray-50 hover:bg-gray-50/50 transition ${isActive ? '' : 'opacity-40'}`}>
                      <td className="py-2.5 px-4">
                        <span className="font-medium text-gray-900">{p.symbol}</span>
                        {p.name && <span className="text-gray-400 text-xs ml-1.5 hidden lg:inline">{p.name}</span>}
                      </td>
                      <td className="text-center"><MktTag mkt={mkt} /></td>
                      <td className="text-right tabular-nums text-gray-700">{p.quantity}</td>
                      <td className="text-right tabular-nums text-gray-700">{formatCurrency(p.avg_price, cur)}</td>
                      <td className="text-right tabular-nums text-gray-900 font-medium">{formatCurrency(currentPrice, cur)}</td>
                      <td className="text-right"><PnLText value={pnl} currency={cur} /></td>
                      <td className="text-right"><PctBadge value={pnlPct} /></td>
                      <td className="text-right px-4 text-xs text-gray-400">
                        <span className="text-rose-400">-{((ext.stop_loss_pct ?? 0.08) * 100).toFixed(0)}%</span>
                        {' / '}
                        <span className="text-emerald-400">+{((ext.take_profit_pct ?? 0.20) * 100).toFixed(0)}%</span>
                        {ext.trailing_active && <span className="ml-1 text-amber-500 font-medium">T</span>}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Market & Macro */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <MarketStateCard />
        <MacroCard />
      </div>
    </div>
  )
}

/* ─── Shared Components ─── */

function MarketPill({ label, phase }: { label: string; phase: string }) {
  const isOpen = phase === 'regular'
  const isPre = phase === 'pre_market'
  const isAfter = phase === 'after_hours'

  const style = isOpen
    ? 'bg-emerald-100 text-emerald-700'
    : isPre
      ? 'bg-sky-100 text-sky-700'
      : isAfter
        ? 'bg-amber-100 text-amber-700'
        : 'bg-gray-100 text-gray-500'

  const dot = isOpen
    ? 'bg-emerald-500'
    : isPre
      ? 'bg-sky-500'
      : isAfter
        ? 'bg-amber-500'
        : 'bg-gray-400'

  const phaseLabel = isOpen ? 'Open' : isPre ? 'Pre' : isAfter ? 'After' : 'Closed'

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold ${style}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dot} ${isOpen ? 'animate-pulse' : ''}`} />
      {label} {phaseLabel}
    </span>
  )
}

function MktTag({ mkt }: { mkt: string }) {
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold ${
      mkt === 'KR' ? 'bg-violet-100 text-violet-700' : 'bg-sky-100 text-sky-700'
    }`}>{mkt}</span>
  )
}

function StatCard({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 px-3 py-2.5">
      <div className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">{label}</div>
      <div className="text-sm font-bold text-gray-900 mt-0.5 truncate">{value}</div>
      {sub && <div className="mt-0.5">{sub}</div>}
    </div>
  )
}

type ReturnPeriod = 'daily' | 'weekly' | 'monthly'

function EquityCard({
  totalEquity, hasUsd, krwTotal, usdTotal, rate, returns,
}: {
  totalEquity: number; hasUsd: boolean; krwTotal: number; usdTotal: number; rate: number
  returns?: { daily: PeriodReturn | null; weekly: PeriodReturn | null; monthly: PeriodReturn | null }
}) {
  const [period, setPeriod] = useState<ReturnPeriod>('daily')
  const labels: Record<ReturnPeriod, string> = { daily: '1D', weekly: '1W', monthly: '1M' }
  const ret = returns?.[period] ?? null

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
      <div className="text-xs text-gray-400 font-semibold uppercase tracking-wider">Total Equity</div>
      <div className="text-3xl font-extrabold text-gray-900 tracking-tight mt-1">
        {formatCurrency(totalEquity, 'KRW')}
      </div>
      {hasUsd && (
        <div className="text-xs text-gray-400 mt-1">
          KRW {formatCurrency(krwTotal, 'KRW')} · USD {formatCurrency(usdTotal, 'USD')} · ₩{rate.toFixed(0)}
        </div>
      )}
      {returns && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <div className="flex items-center gap-1 mb-2">
            {(['daily', 'weekly', 'monthly'] as ReturnPeriod[]).map(p => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-2.5 py-1 text-[11px] rounded-full font-semibold transition ${
                  period === p
                    ? 'bg-gray-900 text-white'
                    : 'text-gray-400 hover:bg-gray-100 hover:text-gray-600'
                }`}
              >
                {labels[p]}
              </button>
            ))}
          </div>
          {ret ? (
            <div>
              <div className="text-[10px] text-gray-400 mb-0.5">실현 손익 (Realized)</div>
              <div className="flex items-baseline gap-2 flex-wrap">
                <span className={`text-lg font-bold ${ret.change >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                  {ret.change >= 0 ? '+' : ''}{formatCurrency(ret.change, 'KRW')}
                </span>
                {(ret as any).realized_us != null && (ret as any).realized_us !== 0 && (
                  <span className="text-[10px] text-gray-400">
                    KR {formatCurrency((ret as any).realized_kr ?? 0, 'KRW')} · US {(ret as any).realized_us >= 0 ? '+' : ''}${(ret as any).realized_us}
                  </span>
                )}
              </div>
            </div>
          ) : (
            <div className="text-xs text-gray-300">No data yet</div>
          )}
        </div>
      )}
    </div>
  )
}

function MarketStateCard() {
  const { data: ms, isLoading } = useQuery({
    queryKey: ['engine', 'market-state'],
    queryFn: fetchMarketState,
    refetchInterval: 60_000,
  })

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4">
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Market</h2>
      {isLoading && <p className="text-gray-300 text-sm">Loading...</p>}
      {ms && (
        <div className="space-y-3 text-sm">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
            <InfoRow label="US" value={ms.market_phase ?? '-'} color={phaseTextColor(ms.market_phase ?? '')} />
            <InfoRow label="Regime" value={ms.regime ?? '-'} />
            <InfoRow label="SPY" value={ms.spy_price != null ? `$${Number(ms.spy_price).toFixed(2)}` : '-'} />
            <InfoRow label="VIX" value={ms.vix_level != null ? Number(ms.vix_level).toFixed(1) : '-'} />
          </div>
          <div className="border-t border-gray-100 pt-2">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
              <InfoRow label="KR" value={ms.kr_market_phase ?? '-'} color={phaseTextColor(ms.kr_market_phase ?? '')} />
              <InfoRow label="Regime" value={ms.kr_regime ?? '-'} />
              {ms.kr_index_price != null && (
                <InfoRow label="KODEX" value={`₩${Number(ms.kr_index_price).toLocaleString()}`} />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function MacroCard() {
  const { data: m, isLoading } = useQuery({
    queryKey: ['engine', 'macro'],
    queryFn: fetchMacroIndicators,
    refetchInterval: 60_000,
  })

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4">
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Macro</h2>
      {isLoading && <p className="text-gray-300 text-sm">Loading...</p>}
      {m && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
          <InfoRow label="Fed Rate" value={m.fed_funds_rate != null ? `${Number(m.fed_funds_rate).toFixed(2)}%` : '-'} />
          <InfoRow label="10Y" value={m.treasury_10y != null ? `${Number(m.treasury_10y).toFixed(2)}%` : '-'} />
          <InfoRow label="Spread" value={m.yield_spread != null ? `${Number(m.yield_spread).toFixed(2)}%` : '-'} />
          <InfoRow label="CPI" value={m.cpi_yoy != null ? `${Number(m.cpi_yoy).toFixed(2)}%` : '-'} />
          <InfoRow label="Unemp." value={m.unemployment_rate != null ? `${Number(m.unemployment_rate).toFixed(1)}%` : '-'} />
        </div>
      )}
    </div>
  )
}

function InfoRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-gray-400 text-xs">{label}</span>
      <span className={`font-medium ${color ?? 'text-gray-900'}`}>{value}</span>
    </div>
  )
}

function phaseTextColor(phase: string) {
  if (phase === 'regular') return 'text-emerald-600'
  if (phase === 'pre_market') return 'text-sky-600'
  if (phase === 'after_hours') return 'text-amber-600'
  return 'text-gray-400'
}

interface PeriodData { pnl: number; pnl_pct: number | null; trades: number; wins: number; losses: number; win_rate: number }

function PnLCard({ label, us, kr }: { label: string; us?: PeriodData; kr?: PeriodData }) {
  const hasKr = (kr?.trades ?? 0) > 0
  const hasUs = (us?.trades ?? 0) > 0

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-3">
      <div className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider mb-1.5">{label}</div>
      {!hasKr && !hasUs ? (
        <div className="text-sm text-gray-300">—</div>
      ) : (
        <div className="space-y-1.5">
          {hasKr && <PnLLine tag="KR" pnl={kr!.pnl} pnlPct={kr!.pnl_pct} currency="KRW" trades={kr!.trades} wins={kr!.wins} losses={kr!.losses} />}
          {hasUs && <PnLLine tag="US" pnl={us!.pnl} pnlPct={us!.pnl_pct} currency="USD" trades={us!.trades} wins={us!.wins} losses={us!.losses} />}
        </div>
      )}
    </div>
  )
}

function PnLLine({ tag, pnl, pnlPct, currency, trades, wins, losses }: {
  tag: string; pnl: number; pnlPct?: number | null; currency: string
  trades: number; wins: number; losses: number
}) {
  const tagStyle = tag === 'KR' ? 'bg-violet-100 text-violet-700' : 'bg-sky-100 text-sky-700'
  return (
    <div>
      <div className="flex items-center gap-1.5">
        <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-semibold ${tagStyle}`}>{tag}</span>
        <span className={`text-sm font-bold ${pnl >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
          {pnl >= 0 ? '+' : ''}{formatCurrency(pnl, currency)}
        </span>
        <span className="text-[10px] text-gray-400 ml-auto">{trades}T {wins}W/{losses}L</span>
      </div>
      {pnlPct != null && (
        <div className="ml-7 mt-0.5"><PctBadge value={pnlPct} /></div>
      )}
    </div>
  )
}
