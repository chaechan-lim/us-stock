import { usePositions } from '../hooks/useApi'
import { formatCurrency } from '../utils/format'

export default function PositionList() {
  const { data: positions, isLoading } = usePositions()

  if (isLoading) return <div className="text-gray-500">Loading positions...</div>
  if (!positions || positions.length === 0) {
    return <div className="text-gray-500">No open positions.</div>
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-4">All Positions</h2>
      <div className="overflow-x-auto">
      <table className="w-full text-sm min-w-[900px]">
        <thead className="text-gray-400 border-b border-gray-700">
          <tr>
            <th className="text-left py-2 px-3">Symbol</th>
            <th className="text-center py-2 px-3">Mkt</th>
            <th className="text-right py-2 px-3">Qty</th>
            <th className="text-right py-2 px-3">Avg Price</th>
            <th className="text-right py-2 px-3">Current</th>
            <th className="text-right py-2 px-3">P&L</th>
            <th className="text-right py-2 px-3">P&L %</th>
            <th className="text-right py-2 px-3">Stop Loss</th>
            <th className="text-right py-2 px-3">Target</th>
          </tr>
        </thead>
        <tbody>
          {positions.map(p => {
            const mkt = p.market ?? 'US'
            const cur = mkt === 'KR' ? 'KRW' : 'USD'
            const pnlColor = p.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
            const slPrice = p.stop_loss_pct ? p.avg_price * (1 - p.stop_loss_pct) : null
            const tpPrice = p.take_profit_pct ? p.avg_price * (1 + p.take_profit_pct) : null
            const slPct = p.stop_loss_pct ? (p.stop_loss_pct * 100).toFixed(0) : null
            const tpPct = p.take_profit_pct ? (p.take_profit_pct * 100).toFixed(0) : null
            return (
              <tr key={p.symbol} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="py-2 px-3 font-medium">
                  <div className="flex items-center gap-1.5">
                    <span className={`text-[10px] px-1 py-0.5 rounded ${mkt === 'KR' ? 'bg-purple-900/40 text-purple-300' : 'bg-blue-900/40 text-blue-300'}`}>
                      {mkt}
                    </span>
                    {p.symbol}
                    {p.name && <span className="text-gray-500 text-xs">{p.name}</span>}
                  </div>
                </td>
                <td className="py-2 px-3 text-center text-gray-400 text-xs">{p.exchange}</td>
                <td className="py-2 px-3 text-right">{p.quantity}</td>
                <td className="py-2 px-3 text-right">{formatCurrency(p.avg_price, cur)}</td>
                <td className="py-2 px-3 text-right">{formatCurrency(p.current_price, cur)}</td>
                <td className={`py-2 px-3 text-right ${pnlColor}`}>
                  {p.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(p.unrealized_pnl, cur)}
                </td>
                <td className={`py-2 px-3 text-right ${pnlColor}`}>
                  {p.unrealized_pnl_pct >= 0 ? '+' : ''}{p.unrealized_pnl_pct.toFixed(2)}%
                </td>
                <td className="py-2 px-3 text-right">
                  {slPrice != null ? (
                    <span className="text-red-400/80 text-xs">
                      {formatCurrency(slPrice, cur)}
                      <span className="text-gray-500 ml-0.5">(-{slPct}%)</span>
                    </span>
                  ) : <span className="text-gray-600">—</span>}
                </td>
                <td className="py-2 px-3 text-right">
                  {tpPrice != null ? (
                    <span className="text-green-400/80 text-xs">
                      {formatCurrency(tpPrice, cur)}
                      <span className="text-gray-500 ml-0.5">(+{tpPct}%)</span>
                    </span>
                  ) : <span className="text-gray-600">—</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      </div>
    </div>
  )
}
