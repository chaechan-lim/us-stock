import { useQuery } from '@tanstack/react-query'
import { useTrades } from '../hooks/useApi'
import { fetchTradeSummaryPeriods, type PeriodSummary } from '../api/client'
import { formatCurrency } from '../utils/format'
import clsx from 'clsx'

export default function TradeHistory() {
  const { data: rawTrades, isLoading } = useTrades(100)
  const { data: periodSummary } = useQuery({
    queryKey: ['trade-summary-periods'],
    queryFn: () => fetchTradeSummaryPeriods(),
    refetchInterval: 60_000,
  })
  const trades = rawTrades?.filter(t => t.status !== 'pending')

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Trade History</h2>

      {/* Period Summary Cards */}
      {periodSummary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <PeriodCard label="Today" data={periodSummary.today} />
          <PeriodCard label="This Week" data={periodSummary.week} />
          <PeriodCard label="This Month" data={periodSummary.month} />
          <PeriodCard label="All Time" data={periodSummary.all_time} />
        </div>
      )}

      {/* Overall Stats */}
      {periodSummary && (
        <div className="flex gap-4 text-sm text-gray-400">
          <span>Total BUYs: <b className="text-white">{periodSummary.total_buys}</b></span>
          <span>Total SELLs: <b className="text-white">{periodSummary.total_sells}</b></span>
        </div>
      )}

      {isLoading ? (
        <div className="text-gray-500">Loading trades...</div>
      ) : !trades || trades.length === 0 ? (
        <div className="text-gray-500 text-sm">No trades recorded yet.</div>
      ) : (
        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
          <table className="w-full text-sm min-w-[750px]">
            <thead className="text-gray-400 border-b border-gray-700">
              <tr>
                <th className="text-left py-2 px-3">Time</th>
                <th className="text-left py-2 px-3">Symbol</th>
                <th className="text-center py-2 px-3">Side</th>
                <th className="text-right py-2 px-3">Qty</th>
                <th className="text-right py-2 px-3">Price</th>
                <th className="text-left py-2 px-3">Strategy</th>
                <th className="text-center py-2 px-3">Status</th>
                <th className="text-right py-2 px-3">P&L</th>
                <th className="text-right py-2 px-3">P&L %</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, i) => (
                <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="py-2 px-3 text-gray-400 text-xs">
                    {t.created_at ? new Date(t.created_at).toLocaleString() : '-'}
                  </td>
                  <td className="py-2 px-3 font-medium">
                    {t.symbol}
                    {t.name && <span className="text-gray-500 text-xs ml-1">{t.name}</span>}
                  </td>
                  <td className="py-2 px-3 text-center">
                    <span className={clsx(
                      'px-2 py-0.5 rounded text-xs font-bold',
                      t.side === 'BUY' ? 'bg-green-900/40 text-green-400' : 'bg-red-900/40 text-red-400'
                    )}>
                      {t.side}
                    </span>
                    {t.session && t.session !== 'regular' && (
                      <span className={clsx('ml-1 px-1 py-0.5 rounded text-[10px] font-semibold', {
                        'bg-orange-900/40 text-orange-300': t.session === 'pre_market',
                        'bg-purple-900/40 text-purple-300': t.session === 'after_hours',
                        'bg-indigo-900/40 text-indigo-300': t.session === 'extended_nxt',
                      })}>
                        {t.session === 'pre_market' ? 'PRE' : t.session === 'after_hours' ? 'AH' : 'NXT'}
                      </span>
                    )}
                  </td>
                  <td className="py-2 px-3 text-right">{t.quantity}</td>
                  <td className="py-2 px-3 text-right">
                    {t.filled_price ? formatCurrency(t.filled_price, t.market === 'KR' ? 'KRW' : 'USD') : t.price ? formatCurrency(t.price, t.market === 'KR' ? 'KRW' : 'USD') : '-'}
                  </td>
                  <td className="py-2 px-3 text-gray-400 text-xs">{t.strategy}</td>
                  <td className="py-2 px-3 text-center">
                    <span className={clsx('text-xs', {
                      'text-green-400': t.status === 'filled',
                      'text-yellow-400': t.status === 'pending',
                      'text-red-400': t.status === 'failed',
                      'text-gray-400': t.status === 'cancelled',
                    })}>
                      {t.status}
                    </span>
                  </td>
                  <td className={clsx('py-2 px-3 text-right', {
                    'text-green-400': t.pnl != null && t.pnl > 0,
                    'text-red-400': t.pnl != null && t.pnl < 0,
                    'text-gray-500': t.pnl == null,
                  })}>
                    {t.pnl != null ? formatCurrency(t.pnl, t.market === 'KR' ? 'KRW' : 'USD') : '-'}
                  </td>
                  <td className={clsx('py-2 px-3 text-right text-xs', {
                    'text-green-400': t.pnl_pct != null && t.pnl_pct > 0,
                    'text-red-400': t.pnl_pct != null && t.pnl_pct < 0,
                    'text-gray-500': t.pnl_pct == null,
                  })}>
                    {t.pnl_pct != null ? `${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%` : '-'}
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

function PeriodCard({ label, data }: { label: string; data: PeriodSummary }) {
  const pnlColor = data.pnl >= 0 ? 'text-green-400' : 'text-red-400'
  const pnlSign = data.pnl >= 0 ? '+' : ''
  return (
    <div className="bg-gray-900 rounded-lg p-3">
      <div className="text-xs text-gray-400 uppercase tracking-wide">{label}</div>
      <div className={clsx('text-lg font-bold mt-1', data.trades === 0 ? 'text-gray-600' : pnlColor)}>
        {data.trades === 0 ? '-' : `${pnlSign}${formatCurrency(data.pnl, 'KRW')}`}
      </div>
      {data.trades > 0 && (
        <>
          {data.pnl_pct != null && (
            <div className={clsx('text-xs mt-0.5', data.pnl_pct >= 0 ? 'text-green-400/70' : 'text-red-400/70')}>
              {data.pnl_pct >= 0 ? '+' : ''}{data.pnl_pct.toFixed(2)}%
            </div>
          )}
          <div className="text-xs text-gray-500 mt-0.5">
            {data.wins}W {data.losses}L ({data.win_rate.toFixed(0)}%) / {data.trades} trades
          </div>
        </>
      )}
    </div>
  )
}
