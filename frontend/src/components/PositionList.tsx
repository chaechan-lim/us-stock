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
      <table className="w-full text-sm">
        <thead className="text-gray-400 border-b border-gray-700">
          <tr>
            <th className="text-left py-2 px-3">Symbol</th>
            <th className="text-center py-2 px-3">Mkt</th>
            <th className="text-left py-2 px-3">Exchange</th>
            <th className="text-right py-2 px-3">Quantity</th>
            <th className="text-right py-2 px-3">Avg Price</th>
            <th className="text-right py-2 px-3">Current Price</th>
            <th className="text-right py-2 px-3">Value</th>
            <th className="text-right py-2 px-3">P&L</th>
            <th className="text-right py-2 px-3">P&L %</th>
          </tr>
        </thead>
        <tbody>
          {positions.map(p => {
            const mkt = (p as { market?: string }).market ?? 'US'
            const cur = mkt === 'KR' ? 'KRW' : 'USD'
            const value = p.quantity * p.current_price
            const pnlColor = p.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
            return (
              <tr key={p.symbol} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="py-2 px-3 font-medium">{p.symbol}</td>
                <td className="py-2 px-3 text-center">
                  <span className={`text-xs px-1.5 py-0.5 rounded ${mkt === 'KR' ? 'bg-purple-900/40 text-purple-300' : 'bg-blue-900/40 text-blue-300'}`}>
                    {mkt}
                  </span>
                </td>
                <td className="py-2 px-3 text-gray-400">{p.exchange}</td>
                <td className="py-2 px-3 text-right">{p.quantity}</td>
                <td className="py-2 px-3 text-right">{formatCurrency(p.avg_price, cur)}</td>
                <td className="py-2 px-3 text-right">{formatCurrency(p.current_price, cur)}</td>
                <td className="py-2 px-3 text-right">{formatCurrency(value, cur)}</td>
                <td className={`py-2 px-3 text-right ${pnlColor}`}>
                  {p.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(p.unrealized_pnl, cur)}
                </td>
                <td className={`py-2 px-3 text-right ${pnlColor}`}>
                  {p.unrealized_pnl_pct >= 0 ? '+' : ''}{p.unrealized_pnl_pct.toFixed(2)}%
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
