import { useState } from 'react'
import { useWatchlist, useAddToWatchlist, useRemoveFromWatchlist } from '../hooks/useApi'
import { useMarket } from '../contexts/MarketContext'

export default function WatchlistPanel() {
  const { market } = useMarket()
  const { data: watchlist } = useWatchlist(market)
  const addMutation = useAddToWatchlist(market)
  const removeMutation = useRemoveFromWatchlist(market)
  const [input, setInput] = useState('')

  const handleAdd = () => {
    if (input.trim()) {
      addMutation.mutate(input.trim().toUpperCase())
      setInput('')
    }
  }

  const placeholder = market === 'KR' ? 'Add symbol (e.g. 005930)' : 'Add symbol (e.g. AAPL)'

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Watchlist</h2>

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleAdd()}
          placeholder={placeholder}
          className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm
                     focus:outline-none focus:border-blue-500 w-48"
        />
        <button
          onClick={handleAdd}
          disabled={!input.trim() || addMutation.isPending}
          className="px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 rounded
                     transition-colors disabled:opacity-50"
        >
          Add
        </button>
      </div>

      {watchlist?.symbols && watchlist.symbols.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {watchlist.symbols.map(symbol => (
            <div
              key={symbol}
              className="flex items-center gap-2 bg-gray-900 rounded-lg px-3 py-2"
            >
              <span className="font-medium text-sm">{symbol}</span>
              <button
                onClick={() => removeMutation.mutate(symbol)}
                className="text-gray-500 hover:text-red-400 text-xs"
              >
                x
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-500 text-sm">No symbols in watchlist.</p>
      )}
    </div>
  )
}
