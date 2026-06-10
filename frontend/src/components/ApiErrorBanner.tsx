import { useEffect, useState } from 'react'
import clsx from 'clsx'
import type { ApiErrorEvent } from '../api/client'

// HIGH-14 (2026-06-07): listens for `apiError` CustomEvents that
// client.ts dispatches on 401/403/429/5xx and shows a top-of-screen
// banner. Auto-dismisses after 8s (auth errors stay until clicked
// since they keep recurring on every poll).
export default function ApiErrorBanner() {
  const [err, setErr] = useState<ApiErrorEvent | null>(null)

  useEffect(() => {
    const handler = (ev: Event) => {
      const detail = (ev as CustomEvent<ApiErrorEvent>).detail
      if (!detail) return
      setErr(detail)
      if (detail.level !== 'auth') {
        const t = window.setTimeout(() => setErr(null), 8000)
        return () => window.clearTimeout(t)
      }
    }
    window.addEventListener('apiError', handler)
    return () => window.removeEventListener('apiError', handler)
  }, [])

  if (!err) return null

  const color =
    err.level === 'auth' ? 'bg-rose-600' :
    err.level === 'rate' ? 'bg-amber-500' :
    'bg-rose-500'

  return (
    <div
      className={clsx(
        'fixed top-0 left-0 right-0 z-50 px-4 py-2 text-sm text-white shadow',
        'flex items-center justify-between gap-4',
        color,
      )}
      role="alert"
    >
      <span>
        <strong className="mr-2">[{err.status}]</strong>
        {err.message}
        {err.url && <span className="ml-2 opacity-70 text-xs">({err.url})</span>}
      </span>
      <button
        onClick={() => setErr(null)}
        className="text-white/80 hover:text-white text-lg leading-none"
        aria-label="dismiss"
      >
        ×
      </button>
    </div>
  )
}
