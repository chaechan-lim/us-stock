import { useState } from 'react'
import { usePerformanceMetrics } from '../hooks/useApi'
import clsx from 'clsx'

const WINDOWS = [
  { days: 7, label: '7d' },
  { days: 30, label: '30d' },
  { days: 90, label: '90d' },
]

const MARKETS = [
  { key: '', label: 'ALL' },
  { key: 'KR', label: 'KR' },
  { key: 'US', label: 'US' },
]

function fmtPct(v: number | null | undefined, signed = true) {
  if (v == null) return '—'
  const s = signed && v > 0 ? '+' : ''
  return `${s}${v.toFixed(1)}%`
}
function fmtNum(v: number | null | undefined, decimals = 2) {
  if (v == null) return '—'
  return v.toFixed(decimals)
}
function fmtMoney(v: number | null | undefined) {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : '-'}${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

function tone(v: number | null | undefined, threshold = 0): 'pos' | 'neg' | 'neu' {
  if (v == null) return 'neu'
  if (v > threshold) return 'pos'
  if (v < threshold) return 'neg'
  return 'neu'
}
const toneClass = {
  pos: 'text-emerald-600',
  neg: 'text-red-600',
  neu: 'text-gray-700',
}

interface KpiTileProps {
  label: string
  value: string
  sub?: string
  toneVal?: 'pos' | 'neg' | 'neu'
  highlight?: boolean
}
function KpiTile({ label, value, sub, toneVal = 'neu', highlight }: KpiTileProps) {
  return (
    <div
      className={clsx(
        'rounded-xl border p-3 bg-white',
        highlight ? 'border-blue-200 bg-blue-50/50' : 'border-gray-100',
      )}
    >
      <div className="text-[10px] uppercase tracking-wide text-gray-500">
        {label}
      </div>
      <div className={clsx('text-lg font-semibold mt-1', toneClass[toneVal])}>
        {value}
      </div>
      {sub && <div className="text-[11px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  )
}

export default function PerformanceDashboard() {
  const [days, setDays] = useState(30)
  const [market, setMarket] = useState<string>('')
  const { data, isLoading, error } = usePerformanceMetrics(days, market || undefined)

  if (isLoading) {
    return (
      <div className="text-sm text-gray-500 p-4">성과 지표 로딩 중...</div>
    )
  }
  if (error || !data) {
    return (
      <div className="text-sm text-red-600 p-4">
        성과 지표 로드 실패. 백엔드가 실행 중인지 확인하세요.
      </div>
    )
  }
  const { equity: e, trades: t, benchmark: bm, target: tg } = data

  const insufficient = !e.sufficient_samples
  const sampleHint = insufficient
    ? `데이터 ${e.sample_days}/7일`
    : null

  return (
    <div className="space-y-3">
      {/* Window + Market selectors */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">기간:</span>
          {WINDOWS.map(w => (
            <button
              key={w.days}
              onClick={() => setDays(w.days)}
              className={clsx(
                'px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
                days === w.days
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200',
              )}
            >
              {w.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">시장:</span>
          {MARKETS.map(m => (
            <button
              key={m.key}
              onClick={() => setMarket(m.key)}
              className={clsx(
                'px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
                market === m.key
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200',
              )}
            >
              {m.label}
            </button>
          ))}
        </div>
        {sampleHint && (
          <div className="text-[11px] text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded">
            ⚠️ {sampleHint} — 일부 지표 신뢰도 낮음
          </div>
        )}
      </div>

      {/* 1순위: Equity-based primary metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <KpiTile
          label="Net Equity"
          value={`₩${(e.end_equity / 10000).toFixed(0)}만`}
          sub={`시작 ₩${(e.start_equity / 10000).toFixed(0)}만`}
          highlight
        />
        <KpiTile
          label="Net 수익률"
          value={fmtPct(e.net_return_pct)}
          sub={
            insufficient
              ? `연환산: 데이터 ${e.sample_days}/7일`
              : `연환산 ${fmtPct(e.annualized_return_pct)}`
          }
          toneVal={tone(e.net_return_pct)}
          highlight
        />
        <KpiTile
          label={`vs ${bm.label} (Adj.)`}
          value={bm.adjusted_alpha_pct != null ? fmtPct(bm.adjusted_alpha_pct) : '—'}
          sub={
            bm.return_pct != null
              ? `${bm.label} ${fmtPct(bm.return_pct)} → exposure-adj ${bm.adjusted_return_pct != null ? fmtPct(bm.adjusted_return_pct) : '—'}`
              : '벤치마크 가져오기 실패'
          }
          toneVal={tone(bm.adjusted_alpha_pct)}
          highlight
        />
        <KpiTile
          label="MDD"
          value={fmtPct(e.max_drawdown_pct, false)}
          sub={
            e.intraday_sample_count > 0 && e.intraday_max_drawdown_pct < e.max_drawdown_pct
              ? `장중 ${fmtPct(e.intraday_max_drawdown_pct, false)} · 복구 ${e.max_dd_recovery_days || 0}일`
              : e.max_dd_recovery_days > 0
                ? `복구 ${e.max_dd_recovery_days}일`
                : '낙폭 없음'
          }
          toneVal={tone(e.max_drawdown_pct)}
          highlight
        />
      </div>

      {/* 2순위: Risk-adjusted */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <KpiTile
          label="Calmar"
          value={insufficient ? '—' : fmtNum(e.calmar_ratio)}
          sub={
            insufficient
              ? `데이터 ${e.sample_days}/7일 필요`
              : '연수익률 / |MDD|'
          }
          toneVal={
            insufficient
              ? 'neu'
              : (e.calmar_ratio ?? 0) > 1 ? 'pos' : (e.calmar_ratio ?? 0) > 0 ? 'neu' : 'neg'
          }
        />
        <KpiTile
          label="Sharpe"
          value={insufficient ? '—' : fmtNum(e.sharpe_ratio)}
          sub={
            insufficient
              ? `데이터 ${e.sample_days}/7일 필요`
              : '일간 수익률 기반 (×√252)'
          }
          toneVal={insufficient ? 'neu' : tone(e.sharpe_ratio)}
        />
        <KpiTile
          label="Sortino"
          value={insufficient ? '—' : fmtNum(e.sortino_ratio)}
          sub={
            insufficient
              ? `데이터 ${e.sample_days}/7일 필요`
              : '하락 변동성만'
          }
          toneVal={insufficient ? 'neu' : tone(e.sortino_ratio)}
        />
        <KpiTile
          label="Exposure"
          value={`${e.exposure_pct.toFixed(0)}%`}
          sub={
            tg.target_exposure_pct != null
              ? `목표 ${tg.target_exposure_pct.toFixed(0)}% · Gap ${tg.exposure_gap_pct != null && tg.exposure_gap_pct > 0 ? '+' : ''}${tg.exposure_gap_pct?.toFixed(0) ?? 0}%`
              : '평균 투자비율'
          }
          toneVal={
            tg.exposure_gap_pct != null
              ? (tg.exposure_gap_pct > 20 ? 'neg' : tg.exposure_gap_pct > 5 ? 'neu' : 'pos')
              : 'neu'
          }
        />
      </div>

      {/* 3순위: Trade-level */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <KpiTile
          label="Net PnL"
          value={fmtMoney(t.net_profit)}
          sub={`수수료 ${fmtMoney(-t.estimated_fees)} / 슬리피지 ${fmtMoney(-t.estimated_slippage)}`}
          toneVal={tone(t.net_profit)}
        />
        <KpiTile
          label="Net PF"
          value={t.net_pf == null ? '∞' : fmtNum(t.net_pf, 2)}
          sub={`Gross ${t.gross_pf == null ? '∞' : fmtNum(t.gross_pf, 2)}`}
          toneVal={
            t.net_pf == null ? 'pos' : t.net_pf > 1 ? 'pos' : t.net_pf > 0 ? 'neu' : 'neg'
          }
        />
        <KpiTile
          label="Expectancy"
          value={fmtMoney(t.expectancy)}
          sub={`평균 W ${fmtNum(t.avg_win, 0)} / L ${fmtNum(t.avg_loss, 0)}`}
          toneVal={tone(t.expectancy)}
        />
        <KpiTile
          label="라운드트립 WR"
          value={t.round_trips > 0 ? `${(t.round_trip_win_rate * 100).toFixed(0)}%` : '—'}
          sub={
            t.round_trips > 0
              ? `${t.round_trip_wins}승/${t.round_trip_losses}패 · Partial WR ${(t.win_rate * 100).toFixed(0)}%`
              : '진행 중 (open positions)'
          }
          toneVal={t.round_trips > 0 && t.round_trip_win_rate >= 0.5 ? 'pos' : 'neu'}
        />
      </div>

      <div className="text-[11px] text-gray-500 px-2">
        💡 Net Equity Curve 우상향 + MDD 감당 가능 + Net PF &gt; 1.2 + Expectancy 양수가
        건강한 봇의 4가지 조건. 수수료/슬리피지 추정 (KR 0.10%, US 0.05% per side).
      </div>
    </div>
  )
}
