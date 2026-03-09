import clsx from 'clsx'
import { useMarket } from '../contexts/MarketContext'

export default function MarketToggle() {
  const { market, setMarket } = useMarket()
  return (
    <div className="flex bg-gray-800 rounded-lg p-0.5">
      <button
        onClick={() => setMarket('US')}
        className={clsx(
          'px-3 py-1 text-xs font-medium rounded-md transition-colors',
          market === 'US'
            ? 'bg-blue-600 text-white'
            : 'text-gray-400 hover:text-white'
        )}
      >
        US
      </button>
      <button
        onClick={() => setMarket('KR')}
        className={clsx(
          'px-3 py-1 text-xs font-medium rounded-md transition-colors',
          market === 'KR'
            ? 'bg-blue-600 text-white'
            : 'text-gray-400 hover:text-white'
        )}
      >
        KR
      </button>
    </div>
  )
}
