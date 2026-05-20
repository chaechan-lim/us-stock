import clsx from 'clsx'
import { useDailyAnalyses } from '../hooks/useApi'
import type { DailyAnalysisArtifact } from '../api/client'

const LEVEL_COLORS: Record<string, { bg: string; ring: string; text: string; label: string }> = {
  info: {
    bg: 'bg-emerald-50',
    ring: 'ring-emerald-200',
    text: 'text-emerald-700',
    label: '✓ 정상',
  },
  warning: {
    bg: 'bg-amber-50',
    ring: 'ring-amber-200',
    text: 'text-amber-700',
    label: '⚠️ 주의',
  },
  critical: {
    bg: 'bg-red-50',
    ring: 'ring-red-200',
    text: 'text-red-700',
    label: '⚠️ 위험',
  },
}

function bodyToHtml(body: string): string {
  // Markdown-lite — backticks → <code>, **x** → <strong>, line breaks
  let html = body
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 bg-gray-100 rounded text-[11px] text-gray-800">$1</code>')
  html = html.replace(/\n/g, '<br/>')
  return html
}

function HeatCell({ a }: { a: DailyAnalysisArtifact }) {
  const color = LEVEL_COLORS[a.level] || LEVEL_COLORS.info
  return (
    <div
      title={`${a.date}: PnL $${a.daily.pnl_usd.toFixed(0)}, cleanups ${a.daily.cleanups}`}
      className={clsx(
        'w-8 h-8 rounded text-[10px] flex flex-col items-center justify-center ring-1',
        color.bg, color.ring, color.text,
      )}
    >
      <div className="font-semibold">{a.date.slice(5)}</div>
    </div>
  )
}

export default function DailyAnalysisPanel() {
  const { data, isLoading, error } = useDailyAnalyses(7)

  if (isLoading) {
    return <div className="text-sm text-gray-500 p-4">일일 분석 로딩 중...</div>
  }
  if (error || !data) {
    return (
      <div className="text-sm text-red-600 p-4">분석 데이터 로드 실패. 백엔드 확인.</div>
    )
  }
  if (data.artifacts.length === 0) {
    return (
      <div className="text-xs text-gray-500 p-3">
        아직 분석 아티팩트가 없습니다. 매일 06:00 KST에 systemd timer가 생성합니다.
      </div>
    )
  }

  const latest = data.artifacts[0]
  const color = LEVEL_COLORS[latest.level] || LEVEL_COLORS.info

  return (
    <div className="space-y-3">
      {/* Header: latest verdict */}
      <div className={clsx('rounded-xl ring-1 p-4', color.bg, color.ring)}>
        <div className="flex items-center justify-between mb-2">
          <div className={clsx('text-sm font-semibold', color.text)}>
            {color.label} · {latest.date}
          </div>
          <div className="text-[11px] text-gray-500">
            {new Date(latest.generated_at).toLocaleString('ko-KR', { hour12: false })}
          </div>
        </div>
        <div
          className="text-xs leading-relaxed text-gray-800"
          dangerouslySetInnerHTML={{ __html: bodyToHtml(latest.body) }}
        />
      </div>

      {/* 7-day trend heatmap */}
      {data.artifacts.length > 1 && (
        <div>
          <div className="text-[11px] uppercase tracking-wide text-gray-500 mb-2">
            최근 {data.artifacts.length}일 추세
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {[...data.artifacts].reverse().map(a => (
              <HeatCell key={a.date} a={a} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
