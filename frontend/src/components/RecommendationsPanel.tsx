import { useState } from 'react'
import {
  useRecommendations,
  useAcceptRecommendation,
  useRejectRecommendation,
} from '../hooks/useApi'
import type { AgentRecommendation } from '../api/client'
import clsx from 'clsx'

const TABS = [
  { key: 'pending', label: '검토 대기' },
  { key: 'accepted', label: '승인됨' },
  { key: 'rejected', label: '거절됨' },
]

const RISK_TONE = {
  low: 'bg-emerald-100 text-emerald-700',
  medium: 'bg-amber-100 text-amber-700',
  high: 'bg-red-100 text-red-700',
} as const

const CONF_TONE = {
  low: 'bg-gray-100 text-gray-700',
  medium: 'bg-blue-100 text-blue-700',
  high: 'bg-blue-200 text-blue-900',
} as const

function formatValue(v: any): string {
  if (v === null || v === undefined) return '∅'
  if (Array.isArray(v)) return `[${v.join(', ')}]`
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function RecommendationRow({ r }: { r: AgentRecommendation }) {
  const [showReject, setShowReject] = useState(false)
  const [reason, setReason] = useState('')
  const accept = useAcceptRecommendation()
  const reject = useRejectRecommendation()

  const isPending = r.status === 'pending'

  return (
    <div className="border border-gray-200 rounded-lg p-3 bg-white">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded">
              #{r.id}
            </span>
            <span className="text-xs font-mono text-gray-700">{r.param_path}</span>
            {r.confidence && (
              <span className={clsx('text-[10px] px-1.5 py-0.5 rounded font-medium',
                CONF_TONE[r.confidence as keyof typeof CONF_TONE] ?? CONF_TONE.medium)}>
                conf: {r.confidence}
              </span>
            )}
            {r.risk && (
              <span className={clsx('text-[10px] px-1.5 py-0.5 rounded font-medium',
                RISK_TONE[r.risk as keyof typeof RISK_TONE] ?? RISK_TONE.medium)}>
                risk: {r.risk}
              </span>
            )}
            <span className="text-[10px] text-gray-500 ml-auto">
              {new Date(r.created_at).toLocaleString('ko-KR', {
                month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
              })}
            </span>
          </div>
          <div className="flex items-center gap-2 mb-2 text-sm">
            <span className="font-mono text-red-600 line-through">
              {formatValue(r.current_value)}
            </span>
            <span className="text-gray-400">→</span>
            <span className="font-mono text-emerald-700 font-semibold">
              {formatValue(r.proposed_value)}
            </span>
          </div>
          {r.rationale && (
            <div className="text-xs text-gray-700 mb-1">
              <span className="font-medium text-gray-900">근거: </span>
              {r.rationale}
            </div>
          )}
          {r.expected_effect && (
            <div className="text-xs text-gray-700 mb-1">
              <span className="font-medium text-gray-900">기대 효과: </span>
              {r.expected_effect}
            </div>
          )}
          {r.backtest_result && (
            <div className="text-[11px] text-gray-600 bg-gray-50 rounded p-1.5 mt-1 font-mono">
              backtest: {JSON.stringify(r.backtest_result)}
            </div>
          )}
          {r.status === 'rejected' && r.rejected_reason && (
            <div className="text-xs text-red-700 mt-1">
              거절 사유: {r.rejected_reason}
            </div>
          )}
          {r.status === 'accepted' && r.applied_at && (
            <div className="text-xs text-emerald-700 mt-1">
              승인됨 ({new Date(r.applied_at).toLocaleString('ko-KR')})
            </div>
          )}
        </div>

        {isPending && (
          <div className="flex flex-col gap-1.5 shrink-0">
            <button
              onClick={() => accept.mutate({ id: r.id })}
              disabled={accept.isPending}
              className="text-xs px-2.5 py-1 bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50"
            >
              승인
            </button>
            <button
              onClick={() => setShowReject(!showReject)}
              className="text-xs px-2.5 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
            >
              거절
            </button>
          </div>
        )}
      </div>
      {showReject && (
        <div className="mt-2 flex gap-1">
          <input
            type="text"
            value={reason}
            onChange={e => setReason(e.target.value)}
            placeholder="거절 사유 (선택)"
            className="flex-1 text-xs border rounded px-2 py-1"
          />
          <button
            onClick={() => {
              reject.mutate({ id: r.id, reason })
              setShowReject(false)
              setReason('')
            }}
            disabled={reject.isPending}
            className="text-xs px-2 py-1 bg-red-600 text-white rounded disabled:opacity-50"
          >
            확정
          </button>
        </div>
      )}
    </div>
  )
}

export default function RecommendationsPanel() {
  const [tab, setTab] = useState('pending')
  const { data, isLoading, error } = useRecommendations(tab)

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 border-b border-gray-200 pb-2">
        {TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={clsx(
              'px-2.5 py-1 rounded text-xs font-medium',
              tab === t.key
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200',
            )}
          >
            {t.label}
            {data && tab === t.key && data.length > 0 && (
              <span className="ml-1.5 text-[10px] opacity-80">({data.length})</span>
            )}
          </button>
        ))}
      </div>

      {isLoading && (
        <div className="text-sm text-gray-500">로딩 중...</div>
      )}
      {error && (
        <div className="text-sm text-red-600">로드 실패</div>
      )}
      {data && data.length === 0 && !isLoading && (
        <div className="text-sm text-gray-500 py-6 text-center">
          {tab === 'pending'
            ? '검토할 권고가 없습니다. trade_review 에이전트가 매일 after_hours에 분석합니다.'
            : '없음'}
        </div>
      )}
      {data && data.length > 0 && (
        <div className="space-y-2">
          {data.map(r => <RecommendationRow key={r.id} r={r} />)}
        </div>
      )}
    </div>
  )
}
