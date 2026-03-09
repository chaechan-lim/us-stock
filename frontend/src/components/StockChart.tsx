import { useEffect, useRef, useState } from 'react'
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData, Time } from 'lightweight-charts'
import { useChart } from '../hooks/useApi'
import { useMarket } from '../contexts/MarketContext'
import MarketToggle from './MarketToggle'

const TIMEFRAMES = ['1D', '1W', '1M'] as const

export default function StockChart() {
  const { market } = useMarket()
  const defaultSymbol = market === 'KR' ? '005930' : 'AAPL'
  const [symbol, setSymbol] = useState(defaultSymbol)
  const [input, setInput] = useState(defaultSymbol)
  const [timeframe, setTimeframe] = useState<string>('1D')

  // Reset symbol when market changes
  useEffect(() => {
    const s = market === 'KR' ? '005930' : 'AAPL'
    setSymbol(s)
    setInput(s)
  }, [market])

  const { data, isLoading, isError } = useChart(symbol, timeframe, market)

  const chartRef = useRef<HTMLDivElement>(null)
  const chartApi = useRef<IChartApi | null>(null)
  const candleSeries = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeries = useRef<ISeriesApi<'Histogram'> | null>(null)

  // Create chart once
  useEffect(() => {
    if (!chartRef.current) return

    const chart = createChart(chartRef.current, {
      layout: {
        background: { color: '#111827' },
        textColor: '#9CA3AF',
      },
      grid: {
        vertLines: { color: '#1F2937' },
        horzLines: { color: '#1F2937' },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: '#374151',
      },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
      },
      width: chartRef.current.clientWidth,
      height: 500,
    })

    const candle = chart.addCandlestickSeries({
      upColor: '#22C55E',
      downColor: '#EF4444',
      borderDownColor: '#EF4444',
      borderUpColor: '#22C55E',
      wickDownColor: '#EF4444',
      wickUpColor: '#22C55E',
    })

    const volume = chart.addHistogramSeries({
      color: '#3B82F6',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    })

    volume.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    })

    chartApi.current = chart
    candleSeries.current = candle
    volumeSeries.current = volume

    const handleResize = () => {
      if (chartRef.current) {
        chart.applyOptions({ width: chartRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  // Update data
  useEffect(() => {
    if (!data?.data?.length || !candleSeries.current || !volumeSeries.current) return

    const candles: CandlestickData<Time>[] = data.data.map(d => ({
      time: d.timestamp as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }))

    const volumes: HistogramData<Time>[] = data.data.map(d => ({
      time: d.timestamp as Time,
      value: d.volume,
      color: d.close >= d.open ? '#22C55E40' : '#EF444440',
    }))

    candleSeries.current.setData(candles)
    volumeSeries.current.setData(volumes)
    chartApi.current?.timeScale().fitContent()
  }, [data])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    const s = input.trim().toUpperCase()
    if (s) setSymbol(s)
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4">
        <MarketToggle />
        <form onSubmit={handleSearch} className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={market === 'KR' ? '종목코드' : 'Symbol'}
            className="bg-gray-800 text-white px-3 py-1.5 rounded text-sm w-28 border border-gray-700 focus:border-blue-500 focus:outline-none"
          />
          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded text-sm"
          >
            Go
          </button>
        </form>

        <div className="flex gap-1">
          {TIMEFRAMES.map(tf => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-3 py-1 rounded text-sm ${
                timeframe === tf
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>

        <span className="text-lg font-bold text-white ml-2">{symbol}</span>
        {isLoading && <span className="text-gray-500 text-sm">Loading...</span>}
        {isError && <span className="text-red-400 text-sm">Failed to load chart</span>}
      </div>

      {/* Chart */}
      <div className="bg-gray-900 rounded-lg p-1">
        <div ref={chartRef} />
      </div>
    </div>
  )
}
