import { useState, useEffect, useRef, useCallback } from 'react'
import clsx from 'clsx'
import axios from 'axios'

interface LogEntry {
  timestamp: string
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG'
  logger?: string
  message: string
}

const LEVEL_COLORS: Record<string, string> = {
  INFO: 'text-blue-400',
  WARN: 'text-yellow-400',
  ERROR: 'text-red-400',
  DEBUG: 'text-gray-500',
}

export default function LogPanel() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filter, setFilter] = useState<string>('ALL')
  const [connected, setConnected] = useState(false)
  const [mode, setMode] = useState<'ws' | 'http'>('ws')
  const [autoScroll, setAutoScroll] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // HTTP polling fallback
  const fetchLogs = useCallback(async () => {
    try {
      const { data } = await axios.get<LogEntry[]>('/api/v1/ws/logs', {
        params: { limit: 200 },
      })
      setLogs(data)
      setConnected(true)
    } catch {
      setConnected(false)
    }
  }, [])

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    fetchLogs()
    pollRef.current = setInterval(fetchLogs, 5000)
    setMode('http')
  }, [fetchLogs])

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  // WebSocket connection
  const connectWs = useCallback(() => {
    stopPolling()
    setMode('ws')

    let retryDelay = 2000
    let retryCount = 0
    let timer: ReturnType<typeof setTimeout> | null = null
    let stopped = false

    function connect() {
      if (stopped) return
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/api/v1/ws/logs`)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        retryDelay = 2000
        retryCount = 0
      }
      ws.onclose = () => {
        setConnected(false)
        if (!stopped) {
          retryCount++
          if (retryCount >= 3) {
            // Fall back to HTTP polling after 3 failed WS attempts
            startPolling()
            return
          }
          timer = setTimeout(connect, retryDelay)
          retryDelay = Math.min(retryDelay * 1.5, 15000)
        }
      }
      ws.onerror = () => ws.close()

      ws.onmessage = (event) => {
        try {
          const entry: LogEntry = JSON.parse(event.data)
          setLogs(prev => [...prev.slice(-499), entry])
        } catch {
          setLogs(prev => [...prev.slice(-499), {
            timestamp: new Date().toISOString(),
            level: 'INFO',
            message: event.data,
          }])
        }
      }
    }

    connect()
    return () => {
      stopped = true
      if (timer) clearTimeout(timer)
      wsRef.current?.close()
    }
  }, [startPolling, stopPolling])

  useEffect(() => {
    const cleanup = connectWs()
    return () => {
      cleanup()
      stopPolling()
    }
  }, [connectWs, stopPolling])

  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  const filtered = filter === 'ALL'
    ? logs
    : logs.filter(l => l.level === filter)

  const errorCount = logs.filter(l => l.level === 'ERROR').length
  const warnCount = logs.filter(l => l.level === 'WARN').length

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h2 className="text-lg font-semibold">System Log</h2>
        <div className="flex items-center gap-3 flex-wrap">
          {/* Connection status */}
          <div className="flex items-center gap-1.5">
            <div className={clsx(
              'w-2 h-2 rounded-full',
              connected ? 'bg-green-500' : 'bg-red-500'
            )} />
            <span className="text-xs text-gray-400">
              {connected ? (mode === 'ws' ? 'Live' : 'Polling') : 'Disconnected'}
            </span>
          </div>

          {/* Stats badges */}
          {errorCount > 0 && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-red-900/60 text-red-300">
              {errorCount} errors
            </span>
          )}
          {warnCount > 0 && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-900/60 text-yellow-300">
              {warnCount} warns
            </span>
          )}

          {/* Reconnect button */}
          {!connected && (
            <button
              onClick={() => { setLogs([]); connectWs() }}
              className="text-xs px-2 py-1 rounded bg-blue-800 hover:bg-blue-700 text-blue-200"
            >
              Reconnect
            </button>
          )}

          {/* Mode toggle */}
          {connected && mode === 'http' && (
            <button
              onClick={() => { setLogs([]); connectWs() }}
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              Try WebSocket
            </button>
          )}

          {/* Filter */}
          <select
            value={filter}
            onChange={e => setFilter(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
          >
            <option value="ALL">All</option>
            <option value="INFO">Info</option>
            <option value="WARN">Warn</option>
            <option value="ERROR">Error</option>
            <option value="DEBUG">Debug</option>
          </select>

          {/* Auto-scroll toggle */}
          <button
            onClick={() => setAutoScroll(v => !v)}
            className={clsx('text-xs', autoScroll ? 'text-green-400' : 'text-gray-500')}
          >
            {autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF'}
          </button>

          <button
            onClick={() => setLogs([])}
            className="text-xs text-gray-400 hover:text-white"
          >
            Clear
          </button>

          <span className="text-xs text-gray-600">{filtered.length} entries</span>
        </div>
      </div>

      <div className="bg-gray-950 border border-gray-800 rounded-lg p-3 h-[500px] overflow-y-auto font-mono text-xs">
        {filtered.length === 0 ? (
          <div className="text-gray-600 text-center mt-8">
            {connected ? 'Waiting for log messages...' : 'Click Reconnect to see logs.'}
          </div>
        ) : (
          filtered.map((entry, i) => (
            <div key={i} className="flex gap-2 py-0.5 hover:bg-gray-900/50">
              <span className="text-gray-600 shrink-0">
                {entry.timestamp.slice(11, 23)}
              </span>
              <span className={clsx('w-12 shrink-0 font-bold', LEVEL_COLORS[entry.level])}>
                {entry.level.padEnd(5)}
              </span>
              {entry.logger && (
                <span className="text-gray-600 shrink-0 max-w-[120px] truncate" title={entry.logger}>
                  {entry.logger.split('.').pop()}
                </span>
              )}
              <span className="text-gray-300 break-all">{entry.message}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
