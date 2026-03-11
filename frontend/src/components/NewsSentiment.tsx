import { useMemo, useState } from 'react'
import { useNewsSentiment } from '../hooks/useApi'
import type { SentimentSummary, SentimentSignal } from '../api/client'

function sentimentColor(v: number): string {
  if (v >= 0.3) return 'text-green-400'
  if (v > 0) return 'text-green-300/70'
  if (v <= -0.3) return 'text-red-400'
  if (v < 0) return 'text-red-300/70'
  return 'text-gray-400'
}

function sentimentBg(v: number): string {
  if (v >= 0.3) return 'bg-green-500'
  if (v > 0) return 'bg-green-500/50'
  if (v <= -0.3) return 'bg-red-500'
  if (v < 0) return 'bg-red-500/50'
  return 'bg-gray-600'
}

function impactBadge(impact: string) {
  const cls = impact === 'HIGH'
    ? 'bg-red-900/60 text-red-300'
    : impact === 'MEDIUM'
      ? 'bg-yellow-900/60 text-yellow-300'
      : 'bg-gray-800 text-gray-400'
  return <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${cls}`}>{impact}</span>
}

function signalBadge(signal: string) {
  const cls = signal === 'BULLISH'
    ? 'text-green-400'
    : signal === 'BEARISH'
      ? 'text-red-400'
      : 'text-gray-500'
  return <span className={`font-medium ${cls}`}>{signal}</span>
}

function SentimentBar({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-20 text-right">{label}</span>
      <div className="flex-1 h-3 bg-gray-800 rounded-full relative overflow-hidden">
        {/* Center line */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-600 z-10" />
        {/* Fill */}
        {value !== 0 && (
          <div
            className={`absolute top-0 bottom-0 ${sentimentBg(value)} rounded-full`}
            style={
              value > 0
                ? { left: '50%', width: `${(value / 1) * 50}%` }
                : { right: '50%', width: `${(Math.abs(value) / 1) * 50}%` }
            }
          />
        )}
      </div>
      <span className={`text-xs font-mono w-12 text-right ${sentimentColor(value)}`}>
        {value >= 0 ? '+' : ''}{value.toFixed(2)}
      </span>
    </div>
  )
}

function SentimentPanel({ summary, signals, symbolList, sectorList, updatedAt }: {
  summary: SentimentSummary
  signals: SentimentSignal[]
  symbolList: { symbol: string; score: number }[]
  sectorList: { sector: string; score: number }[]
  updatedAt: string
}) {
  return (
    <div className="space-y-6">
      {/* Stats row */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">{summary.analyzed_count} articles analyzed</span>
        <span className="text-xs text-gray-500">Updated: {updatedAt}</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="text-xs text-gray-400 uppercase mb-2">Market Sentiment</div>
          <div className={`text-3xl font-bold ${sentimentColor(summary.market_sentiment)}`}>
            {summary.market_sentiment >= 0 ? '+' : ''}{summary.market_sentiment.toFixed(2)}
          </div>
          <div className="mt-2">
            <div className="h-2 bg-gray-800 rounded-full relative overflow-hidden">
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-600 z-10" />
              {summary.market_sentiment !== 0 && (
                <div
                  className={`absolute top-0 bottom-0 ${sentimentBg(summary.market_sentiment)} rounded-full`}
                  style={
                    summary.market_sentiment > 0
                      ? { left: '50%', width: `${(summary.market_sentiment) * 50}%` }
                      : { right: '50%', width: `${Math.abs(summary.market_sentiment) * 50}%` }
                  }
                />
              )}
            </div>
          </div>
        </div>

        <div className="bg-gray-900 rounded-lg p-4">
          <div className="text-xs text-gray-400 uppercase mb-2">Symbol Coverage</div>
          <div className="text-3xl font-bold">{Object.keys(summary.symbol_sentiments).length}</div>
          <div className="text-xs text-gray-500 mt-1">stocks analyzed</div>
        </div>

        <div className="bg-gray-900 rounded-lg p-4">
          <div className="text-xs text-gray-400 uppercase mb-2">Actionable Signals</div>
          <div className={`text-3xl font-bold ${summary.actionable_count > 0 ? 'text-yellow-400' : 'text-gray-500'}`}>
            {summary.actionable_count}
          </div>
          <div className="text-xs text-gray-500 mt-1">high-conviction signals</div>
        </div>
      </div>

      {sectorList.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-semibold mb-3 text-gray-300">Sector Sentiment</h3>
          <div className="space-y-2">
            {sectorList.map(s => (
              <SentimentBar key={s.sector} label={s.sector} value={s.score} />
            ))}
          </div>
        </div>
      )}

      {symbolList.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-semibold mb-3 text-gray-300">Symbol Sentiment</h3>
          <div className="space-y-1.5">
            {symbolList.map(s => (
              <SentimentBar key={s.symbol} label={s.symbol} value={s.score} />
            ))}
          </div>
        </div>
      )}

      {signals.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-semibold mb-3 text-gray-300">Actionable Signals</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[600px]">
              <thead className="text-gray-400 border-b border-gray-800">
                <tr>
                  <th className="text-left py-2">Symbol</th>
                  <th className="text-right py-2">Sentiment</th>
                  <th className="text-center py-2">Signal</th>
                  <th className="text-center py-2">Impact</th>
                  <th className="text-left py-2">Event</th>
                  <th className="text-center py-2">Urgency</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((s, i) => (
                  <tr key={i} className="border-b border-gray-800/50">
                    <td className="py-2 font-medium">{s.symbol}</td>
                    <td className={`text-right py-2 font-mono ${sentimentColor(s.sentiment)}`}>
                      {s.sentiment >= 0 ? '+' : ''}{s.sentiment.toFixed(2)}
                    </td>
                    <td className="text-center py-2">{signalBadge(s.trading_signal)}</td>
                    <td className="text-center py-2">{impactBadge(s.impact)}</td>
                    <td className="py-2 text-gray-300 text-xs max-w-[250px] truncate">{s.key_event}</td>
                    <td className="text-center py-2 text-xs text-gray-500">{s.time_sensitivity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default function NewsSentiment() {
  const { data, isLoading } = useNewsSentiment()
  const [market, setMarket] = useState<'US' | 'KR'>('US')

  const activeData = useMemo(() => {
    if (!data) return null
    if (market === 'KR' && data.kr) {
      return { summary: data.kr.summary, signals: data.kr.signals, updated_at: data.kr.updated_at }
    }
    return { summary: data.summary, signals: data.signals, updated_at: data.updated_at }
  }, [data, market])

  const symbolList = useMemo(() => {
    if (!activeData?.summary.symbol_sentiments) return []
    return Object.entries(activeData.summary.symbol_sentiments)
      .map(([symbol, score]) => ({ symbol, score }))
      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
  }, [activeData])

  const sectorList = useMemo(() => {
    if (!activeData?.summary.sector_sentiments) return []
    return Object.entries(activeData.summary.sector_sentiments)
      .map(([sector, score]) => ({ sector, score }))
      .sort((a, b) => b.score - a.score)
  }, [activeData])

  if (isLoading) {
    return <div className="text-gray-500">Loading sentiment data...</div>
  }

  const hasUs = data?.updated_at
  const hasKr = data?.kr?.updated_at

  if (!data || (!hasUs && !hasKr)) {
    return (
      <div className="bg-gray-900 rounded-lg p-8 text-center">
        <p className="text-gray-500">No sentiment data available yet.</p>
        <p className="text-gray-600 text-xs mt-1">Sentiment analysis runs pre-market and every 30 min during trading hours.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header with market toggle */}
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold">News Sentiment</h2>
        <div className="flex rounded-md overflow-hidden border border-gray-700">
          <button
            onClick={() => setMarket('US')}
            className={`px-3 py-1 text-xs font-medium ${market === 'US' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
          >
            US {hasUs ? '' : '(-)'}
          </button>
          <button
            onClick={() => setMarket('KR')}
            className={`px-3 py-1 text-xs font-medium ${market === 'KR' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
          >
            KR {hasKr ? '' : '(-)'}
          </button>
        </div>
      </div>

      {activeData && activeData.updated_at ? (
        <SentimentPanel
          summary={activeData.summary}
          signals={activeData.signals}
          symbolList={symbolList}
          sectorList={sectorList}
          updatedAt={new Date(activeData.updated_at).toLocaleString()}
        />
      ) : (
        <div className="bg-gray-900 rounded-lg p-8 text-center">
          <p className="text-gray-500">No {market} sentiment data available yet.</p>
        </div>
      )}
    </div>
  )
}
