import { useTrades, useTradeSummary } from '../hooks/useApi'
import { formatCurrency } from '../utils/format'
import clsx from 'clsx'

export default function TradeHistory() {
  const { data: rawTrades, isLoading } = useTrades(100)
  const { data: summary } = useTradeSummary()
  const trades = rawTrades?.filter(t => t.status !== 'pending')

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Trade History</h2>

      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <StatCard label="Total Trades" value={String(summary.total_trades)} />
          <StatCard label="Wins" value={String(summary.wins)} color="text-green-400" />
          <StatCard label="Losses" value={String(summary.losses)} color="text-red-400" />
          <StatCard
            label="Total P&L"
            value={formatCurrency(summary.total_pnl, summary.currency === 'KRW' ? 'KRW' : 'USD')}
            color={summary.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}
          />
          <StatCard label="Win Rate" value={`${summary.win_rate.toFixed(1)}%`} />
        </div>
      )}

      {isLoading ? (
        <div className="text-gray-500">Loading trades...</div>
      ) : !trades || trades.length === 0 ? (
        <div className="text-gray-500 text-sm">No trades recorded yet.</div>
      ) : (
        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
          <table className="w-full text-sm min-w-[650px]">
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
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-gray-900 rounded-lg p-3">
      <div className="text-xs text-gray-400 uppercase">{label}</div>
      <div className={clsx('text-xl font-bold mt-1', color || 'text-white')}>{value}</div>
    </div>
  )
}
