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
import SignalPanel from './SignalPanel'
import NewsSentiment from './NewsSentiment'
import EventsCalendar from './EventsCalendar'
import MarketToggle from './MarketToggle'
import AccountSelector from './AccountSelector'
import { AccountProvider } from '../contexts/AccountContext'
import { useAccounts } from '../hooks/useApi'
import clsx from 'clsx'

type Tab = 'dashboard' | 'positions' | 'trades' | 'signals' | 'chart' | 'strategies' | 'scanner' | 'watchlist' | 'logs' | 'backtest' | 'optimize' | 'portfolio' | 'sectors' | 'performance' | 'etf' | 'sentiment'

const TABS: { key: Tab; label: string }[] = [
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'portfolio', label: 'Portfolio' },
  { key: 'positions', label: 'Positions' },
  { key: 'trades', label: 'Trades' },
  { key: 'signals', label: 'Signals' },
  { key: 'chart', label: 'Chart' },
  { key: 'strategies', label: 'Strategies' },
  { key: 'performance', label: 'Performance' },
  { key: 'backtest', label: 'Backtest' },
  { key: 'optimize', label: 'Optimize' },
  { key: 'scanner', label: 'Scanner' },
  { key: 'sectors', label: 'Sectors' },
  { key: 'sentiment', label: 'Sentiment' },
  { key: 'etf', label: 'ETF' },
  { key: 'watchlist', label: 'Watchlist' },
  { key: 'logs', label: 'Logs' },
]

/** Inner component that has access to useAccounts (requires QueryClientProvider). */
function AppShell() {
  const [tab, setTab] = useState<Tab>('dashboard')
  const { data: accounts = [] } = useAccounts()

  return (
    <AccountProvider accounts={accounts}>
      <AppContent tab={tab} setTab={setTab} />
    </AccountProvider>
  )
}

export default function App() {
  return <AppShell />
}

function AppContent({ tab, setTab }: { tab: Tab; setTab: (t: Tab) => void }) {
  return (
    <div className="min-h-screen bg-gray-950">
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Trading Engine</h1>
        <div className="flex items-center gap-3">
          <AccountSelector />
          <EngineControl />
        </div>
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

      <main className="p-3 sm:p-6">
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'positions' && <PositionList />}
        {tab === 'trades' && <TradeHistory />}
        {tab === 'signals' && <><div className="flex items-center justify-between mb-4"><h2 className="text-lg font-semibold">Strategy Signals</h2><MarketToggle /></div><SignalPanel /></>}
        {tab === 'chart' && <StockChart />}
        {tab === 'strategies' && <StrategyPanel />}
        {tab === 'scanner' && <ScannerPanel />}
        {tab === 'watchlist' && <WatchlistPanel />}
        {tab === 'backtest' && <BacktestPanel />}
        {tab === 'optimize' && <OptimizePanel />}
        {tab === 'portfolio' && <PortfolioChart />}
        {tab === 'sectors' && <SectorHeatmap />}
        {tab === 'performance' && <StrategyPerformance />}
        {tab === 'sentiment' && <><div className="flex items-center justify-between mb-4"><h2 className="text-lg font-semibold">Sentiment & Events</h2><MarketToggle /></div><NewsSentiment /><div className="mt-6"><EventsCalendar /></div></>}
        {tab === 'etf' && <ETFPanel />}
        {tab === 'logs' && <LogPanel />}
      </main>
    </div>
  )
}
