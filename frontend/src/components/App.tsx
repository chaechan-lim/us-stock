import { useState } from 'react'
import Dashboard from './Dashboard'
import PositionList from './PositionList'
import StrategyPanel from './StrategyPanel'
import EngineControl from './EngineControl'
import WatchlistPanel from './WatchlistPanel'
import ScannerPanel from './ScannerPanel'
import LogPanel from './LogPanel'
import TradeHistory from './TradeHistory'
import StockChart from './StockChart'
import BacktestPanel from './BacktestPanel'
import OptimizePanel from './OptimizePanel'
import PortfolioChart from './PortfolioChart'
import SectorHeatmap from './SectorHeatmap'
import StrategyPerformance from './StrategyPerformance'
import ETFPanel from './ETFPanel'
import clsx from 'clsx'

type Tab = 'dashboard' | 'positions' | 'trades' | 'chart' | 'strategies' | 'scanner' | 'watchlist' | 'logs' | 'backtest' | 'optimize' | 'portfolio' | 'sectors' | 'performance' | 'etf'

const TABS: { key: Tab; label: string }[] = [
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'portfolio', label: 'Portfolio' },
  { key: 'positions', label: 'Positions' },
  { key: 'trades', label: 'Trades' },
  { key: 'chart', label: 'Chart' },
  { key: 'strategies', label: 'Strategies' },
  { key: 'performance', label: 'Performance' },
  { key: 'backtest', label: 'Backtest' },
  { key: 'optimize', label: 'Optimize' },
  { key: 'scanner', label: 'Scanner' },
  { key: 'sectors', label: 'Sectors' },
  { key: 'etf', label: 'ETF' },
  { key: 'watchlist', label: 'Watchlist' },
  { key: 'logs', label: 'Logs' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('dashboard')

  return (
    <div className="min-h-screen bg-gray-950">
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Trading Engine</h1>
        <EngineControl />
      </header>

      <nav className="bg-gray-900 border-b border-gray-800 px-3 sm:px-6 py-1">
        <div className="flex flex-wrap gap-1">
          {TABS.map(t => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={clsx(
                'px-3 py-1.5 text-xs sm:text-sm font-medium rounded-t transition-colors whitespace-nowrap',
                tab === t.key
                  ? 'bg-gray-800 text-white border-b-2 border-blue-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="p-6">
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'positions' && <PositionList />}
        {tab === 'trades' && <TradeHistory />}
        {tab === 'chart' && <StockChart />}
        {tab === 'strategies' && <StrategyPanel />}
        {tab === 'scanner' && <ScannerPanel />}
        {tab === 'watchlist' && <WatchlistPanel />}
        {tab === 'backtest' && <BacktestPanel />}
        {tab === 'optimize' && <OptimizePanel />}
        {tab === 'portfolio' && <PortfolioChart />}
        {tab === 'sectors' && <SectorHeatmap />}
        {tab === 'performance' && <StrategyPerformance />}
        {tab === 'etf' && <ETFPanel />}
        {tab === 'logs' && <LogPanel />}
      </main>
    </div>
  )
}
