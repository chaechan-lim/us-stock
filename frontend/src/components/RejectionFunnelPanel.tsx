import { useRejectionFunnel } from '../hooks/useApi'
import clsx from 'clsx'

const REASON_LABELS: Record<string, string> = {
  opening_avoidance: '개장 회피',
  daily_limit: '일일 매수 한도',
  daily_budget_conf_bar: '예산 confidence bar',
  pending_order: '주문 대기 중',
  already_held: '이미 보유',
  already_held_exchange: '거래소 보유 (이중방어)',
  sell_cooldown: '매도 쿨다운',
  whipsaw_block: 'Whipsaw 차단',
  same_signal_24h: '24h 동일 신호',
  event_calendar: '이벤트 (실적/FOMC)',
  sector_limit: '섹터 집중도',
  position_concentration: '단일종목 집중도',
  risk_agent_block: 'AI 리스크 차단',
}

function labelFor(reason: string): string {
  if (REASON_LABELS[reason]) return REASON_LABELS[reason]
  if (reason.startsWith('sizing_')) return `Sizing: ${reason.slice(7)}`
  return reason
}

export default function RejectionFunnelPanel() {
  const { data, isLoading, error } = useRejectionFunnel()

  if (isLoading) {
    return <div className="text-sm text-gray-500 p-4">Funnel 로딩 중...</div>
  }
  if (error || !data) {
    return (
      <div className="text-sm text-red-600 p-4">
        Funnel 로드 실패. 백엔드 확인.
      </div>
    )
  }

  const markets = (['US', 'KR'] as const).filter(m => data[m])
  if (markets.length === 0) {
    return (
      <div className="text-xs text-gray-500 p-3">
        오늘 BUY 신호가 아직 없습니다. (엔진 미시작 또는 시장 외 시간)
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      {markets.map(market => {
        const m = data[market]!
        const total = m.buy_signals_total
        const placed = m.buys_placed
        const rejectedTotal = m.rejected_total
        const fillPct = m.fill_rate != null ? (m.fill_rate * 100).toFixed(0) : '—'
        const rows = Object.entries(m.rejections)
        const maxCount = rows.length > 0 ? Math.max(...rows.map(([, v]) => v)) : 1

        return (
          <div
            key={market}
            className="rounded-xl border border-gray-200 bg-white p-3 space-y-2"
          >
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-gray-700">
                {market} 매수 깔때기
              </div>
              <div className="text-[11px] text-gray-500">
                {m.daily_buy_date || '—'} · 한도 {m.daily_buy_count}/{m.daily_buy_limit}
              </div>
            </div>

            {/* Funnel summary */}
            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-md bg-gray-50 px-2 py-1.5">
                <div className="text-[10px] uppercase text-gray-500">신호</div>
                <div className="text-base font-semibold text-gray-900">{total}</div>
              </div>
              <div className="rounded-md bg-red-50 px-2 py-1.5">
                <div className="text-[10px] uppercase text-red-600">거절</div>
                <div className="text-base font-semibold text-red-700">{rejectedTotal}</div>
              </div>
              <div className="rounded-md bg-emerald-50 px-2 py-1.5">
                <div className="text-[10px] uppercase text-emerald-600">체결</div>
                <div className="text-base font-semibold text-emerald-700">
                  {placed} <span className="text-[11px] font-normal text-emerald-600">({fillPct}%)</span>
                </div>
              </div>
            </div>

            {/* Rejection breakdown */}
            {rows.length === 0 ? (
              <div className="text-[11px] text-gray-500 px-1 py-2">거절 없음</div>
            ) : (
              <div className="space-y-1">
                {rows.map(([reason, count]) => {
                  const widthPct = (count / maxCount) * 100
                  const sharePct = total > 0 ? ((count / total) * 100).toFixed(0) : '0'
                  return (
                    <div key={reason} className="flex items-center gap-2">
                      <div className="text-[11px] text-gray-700 w-32 shrink-0 truncate">
                        {labelFor(reason)}
                      </div>
                      <div className="flex-1 bg-gray-100 rounded h-4 relative overflow-hidden">
                        <div
                          className={clsx(
                            'h-full',
                            count >= total * 0.3
                              ? 'bg-red-400'
                              : count >= total * 0.1
                                ? 'bg-amber-300'
                                : 'bg-gray-300',
                          )}
                          style={{ width: `${widthPct}%` }}
                        />
                      </div>
                      <div className="text-[11px] text-gray-700 w-16 shrink-0 text-right tabular-nums">
                        {count} <span className="text-gray-400">({sharePct}%)</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
