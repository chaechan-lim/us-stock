import { useState } from 'react'
import Dashboard from './Dashboard'
import StrategyPanel from './StrategyPanel'
import StrategyPerformance from './StrategyPerformance'
import EngineControl from './EngineControl'
import WatchlistPanel from './WatchlistPanel'
import ScannerPanel from './ScannerPanel'
import LogPanel from './LogPanel'
import TradeHistory from './TradeHistory'
import SignalPanel from './SignalPanel'
import StockChart from './StockChart'
import SectorHeatmap from './SectorHeatmap'
import ETFPanel from './ETFPanel'
import NewsSentiment from './NewsSentiment'
import EventsCalendar from './EventsCalendar'
import MarketToggle from './MarketToggle'
import AccountSelector from './AccountSelector'
import { AccountProvider } from '../contexts/AccountContext'
import { useAccounts } from '../hooks/useApi'
import clsx from 'clsx'

type Tab = 'dashboard' | 'trades' | 'chart' | 'scanner' | 'market' | 'strategies' | 'logs'

const TABS: { key: Tab; label: string }[] = [
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'trades', label: 'Trades' },
  { key: 'chart', label: 'Chart' },
  { key: 'scanner', label: 'Scanner' },
  { key: 'market', label: 'Market' },
  { key: 'strategies', label: 'Strategies' },
  { key: 'logs', label: 'Logs' },
]

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
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-lg font-bold text-gray-900">Trading Engine</h1>
        <div className="flex items-center gap-2">
          <AccountSelector />
          <EngineControl />
        </div>
      </header>

      {/* Nav */}
      <nav className="bg-white border-b border-gray-200 px-4 sm:px-6 py-1">
        <div className="flex gap-1 overflow-x-auto">
          {TABS.map(t => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={clsx(
                'px-3 py-2 text-xs sm:text-sm font-semibold rounded-lg transition whitespace-nowrap',
                tab === t.key
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-500 hover:bg-gray-100 hover:text-gray-700'
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main className="p-3 sm:p-6">
        {tab === 'dashboard' && <Dashboard />}

        {tab === 'trades' && (
          <div className="space-y-6">
            <TradeHistory />
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-gray-900">Strategy Signals</h2>
                <MarketToggle />
              </div>
              <SignalPanel />
            </div>
          </div>
        )}

        {tab === 'chart' && <StockChart />}

        {tab === 'scanner' && (
          <div className="space-y-6">
            <ScannerPanel />
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4">
              <h2 className="text-sm font-semibold text-gray-900 mb-3">Watchlist</h2>
              <WatchlistPanel />
            </div>
          </div>
        )}

        {tab === 'market' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">Market Overview</h2>
              <MarketToggle />
            </div>
            <SectorHeatmap />
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4">
              <h2 className="text-sm font-semibold text-gray-900 mb-3">ETF Engine</h2>
              <ETFPanel />
            </div>
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4">
              <h2 className="text-sm font-semibold text-gray-900 mb-3">Sentiment & Events</h2>
              <NewsSentiment />
              <div className="mt-4">
                <EventsCalendar />
              </div>
            </div>
          </div>
        )}

        {tab === 'strategies' && (
          <div className="space-y-6">
            <StrategyPanel />
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4">
              <h2 className="text-sm font-semibold text-gray-900 mb-3">Strategy Performance</h2>
              <StrategyPerformance />
            </div>
          </div>
        )}

        {tab === 'logs' && <LogPanel />}
      </main>
    </div>
  )
}
