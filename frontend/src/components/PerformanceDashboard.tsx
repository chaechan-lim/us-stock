import { useState } from 'react'
import { usePerformanceMetrics, useEquityHistory } from '../hooks/useApi'
import clsx from 'clsx'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

const WINDOWS = [
  { days: 7, label: '7d' },
  { days: 30, label: '30d' },
  { days: 90, label: '90d' },
]

// Market filter removed (2026-05-23): on a 통합증거금 account,
// per-market equity has cross-leakage between US/KR (collateral
// sharing, FX conversion of holdings) so KR-only / US-only equity
// metrics are misleading. ALL view via integrated_total_krw is the
// only portfolio-level metric that makes sense. The series builds
// up since 2026-05-06 when KIS CTRP6548R wiring landed; before that
// the API has no integrated total to serve.

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
  // Always ALL — single-market view removed (see top-of-file note).
  const { data, isLoading, error } = usePerformanceMetrics(days, undefined)

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
    ? `데이터 ${e.sample_days}/30일`
    : null

  return (
    <div className="space-y-3">
      {/* Window selector — market filter removed (통합증거금) */}
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
        <span className="text-[10px] text-gray-400">
          통합 (US + KR) · CTRP6548R 통합총자산 기반
        </span>
        {sampleHint && (
          <div className="text-[11px] text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded">
            ⚠️ {sampleHint} — 통합총자산 추적은 2026-05-06부터 시작, 일부 지표 신뢰도 낮음
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
              ? `연환산: 데이터 ${e.sample_days}/30일`
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
              ? `데이터 ${e.sample_days}/30일 필요`
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
              ? `데이터 ${e.sample_days}/30일 필요`
              : '일간 수익률 기반 (×√252)'
          }
          toneVal={insufficient ? 'neu' : tone(e.sharpe_ratio)}
        />
        <KpiTile
          label="Sortino"
          value={insufficient ? '—' : fmtNum(e.sortino_ratio)}
          sub={
            insufficient
              ? `데이터 ${e.sample_days}/30일 필요`
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

      {/* Health checklist */}
      <HealthChecklist
        netReturnPct={e.net_return_pct}
        maxDrawdownPct={e.max_drawdown_pct}
        netPf={t.net_pf}
        expectancy={t.expectancy}
        sharpe={insufficient ? null : e.sharpe_ratio}
        exposureGapPct={tg.exposure_gap_pct}
      />

      {/* Net equity curve — uses combined 통합 총자산 series */}
      <EquityCurve days={days} />

      <div className="text-[11px] text-gray-500 px-2">
        💡 Net Equity Curve 우상향 + MDD 감당 가능 + Net PF &gt; 1.2 + Expectancy 양수가
        건강한 봇의 4가지 조건. 수수료/슬리피지 추정 (KR 0.10%, US 0.05% per side).
      </div>
    </div>
  )
}


interface HealthRow {
  label: string
  status: 'pass' | 'warn' | 'fail' | 'na'
  detail: string
}

function HealthChecklist({
  netReturnPct, maxDrawdownPct, netPf, expectancy, sharpe, exposureGapPct,
}: {
  netReturnPct: number
  maxDrawdownPct: number
  netPf: number | null
  expectancy: number
  sharpe: number | null
  exposureGapPct: number | null | undefined
}) {
  const rows: HealthRow[] = [
    {
      label: 'Net Equity Curve 우상향',
      status: netReturnPct > 2 ? 'pass' : netReturnPct >= 0 ? 'warn' : 'fail',
      detail: `Net 수익률 ${netReturnPct >= 0 ? '+' : ''}${netReturnPct.toFixed(2)}%`,
    },
    {
      label: 'MDD 감당 가능 (< 20%)',
      status:
        maxDrawdownPct >= -10
          ? 'pass'
          : maxDrawdownPct >= -20
          ? 'warn'
          : 'fail',
      detail: `최대 낙폭 ${maxDrawdownPct.toFixed(2)}%`,
    },
    {
      label: 'Net PF > 1.2',
      status:
        netPf == null
          ? 'pass'  // ∞ — only winning trades, treat as healthy
          : netPf >= 1.2
          ? 'pass'
          : netPf > 1
          ? 'warn'
          : 'fail',
      detail: `Net Profit Factor ${netPf == null ? '∞' : netPf.toFixed(2)}`,
    },
    {
      label: 'Expectancy 양수',
      status:
        expectancy > 0 ? 'pass' : expectancy === 0 ? 'warn' : 'fail',
      detail: `Expectancy ${expectancy >= 0 ? '+' : ''}${expectancy.toFixed(2)}`,
    },
    {
      label: 'Sharpe > 1.0',
      status:
        sharpe == null
          ? 'na'
          : sharpe >= 1.0
          ? 'pass'
          : sharpe >= 0.5
          ? 'warn'
          : 'fail',
      detail: sharpe == null ? '데이터 부족' : `Sharpe ${sharpe.toFixed(2)}`,
    },
    {
      label: 'Exposure Gap < 20pp',
      status:
        exposureGapPct == null
          ? 'na'
          : Math.abs(exposureGapPct) <= 10
          ? 'pass'
          : Math.abs(exposureGapPct) <= 20
          ? 'warn'
          : 'fail',
      detail:
        exposureGapPct == null
          ? '목표 미설정'
          : `Gap ${exposureGapPct > 0 ? '+' : ''}${exposureGapPct.toFixed(0)}pp`,
    },
  ]

  const counts = {
    pass: rows.filter(r => r.status === 'pass').length,
    warn: rows.filter(r => r.status === 'warn').length,
    fail: rows.filter(r => r.status === 'fail').length,
    na: rows.filter(r => r.status === 'na').length,
  }
  const total = rows.length - counts.na
  const score = total > 0 ? counts.pass + counts.warn * 0.5 : 0

  return (
    <div className="rounded-xl border border-gray-100 bg-white p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[11px] uppercase tracking-wide text-gray-500">
          🏥 건강한 봇 조건
        </div>
        <div className="text-[11px] text-gray-500">
          충족 {counts.pass}/{total}{counts.warn ? ` · 주의 ${counts.warn}` : ''}
          {counts.fail ? ` · 실패 ${counts.fail}` : ''}
        </div>
      </div>
      {/* Score bar */}
      <div className="w-full h-1.5 bg-gray-100 rounded-full mb-3 overflow-hidden">
        <div
          className={clsx(
            'h-full transition-all',
            score / total >= 0.8
              ? 'bg-emerald-500'
              : score / total >= 0.5
              ? 'bg-amber-500'
              : 'bg-red-500'
          )}
          style={{ width: total > 0 ? `${(score / total) * 100}%` : '0%' }}
        />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
        {rows.map(r => (
          <HealthLine key={r.label} {...r} />
        ))}
      </div>
    </div>
  )
}

function HealthLine({ label, status, detail }: HealthRow) {
  const icon =
    status === 'pass' ? '✓' : status === 'warn' ? '!' : status === 'fail' ? '✗' : '—'
  const iconClass =
    status === 'pass'
      ? 'bg-emerald-100 text-emerald-700'
      : status === 'warn'
      ? 'bg-amber-100 text-amber-700'
      : status === 'fail'
      ? 'bg-red-100 text-red-700'
      : 'bg-gray-100 text-gray-500'
  const textClass = status === 'na' ? 'text-gray-400' : 'text-gray-800'
  return (
    <div className="flex items-center gap-2 text-xs">
      <span
        className={clsx(
          'w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-bold flex-shrink-0',
          iconClass,
        )}
      >
        {icon}
      </span>
      <div className="flex-1 min-w-0">
        <div className={clsx('font-medium', textClass)}>{label}</div>
        <div className="text-[10px] text-gray-500">{detail}</div>
      </div>
    </div>
  )
}


function EquityCurve({ days }: { days: number }) {
  // 'combined' triggers kr_pm.get_combined_equity_history → returns the
  // CTRP6548R integrated total (₩, since 2026-05-06 onwards).
  const { data, isLoading, error } = useEquityHistory(days, 'combined')

  if (isLoading) {
    return (
      <div className="rounded-xl border border-gray-100 bg-white p-3 text-xs text-gray-500">
        Equity curve 로딩 중...
      </div>
    )
  }
  if (error || !data || data.length === 0) {
    return (
      <div className="rounded-xl border border-gray-100 bg-white p-3 text-xs text-gray-500">
        Equity curve 데이터 없음.
      </div>
    )
  }

  // Dedupe by date — keep the last value per day. The combined endpoint
  // surfaces the integrated total in `total_value_krw` (not _usd).
  const byDate = new Map<string, number>()
  for (const p of data) {
    const d = (p.date ?? '').slice(0, 10)
    if (!d) continue
    const v = (p as { total_value_krw?: number; total_value_usd?: number })
      .total_value_krw
      ?? (p as { total_value_usd?: number }).total_value_usd
      ?? 0
    if (v) byDate.set(d, v)
  }
  const series = Array.from(byDate.entries()).map(([date, value]) => ({ date, value }))
  if (series.length === 0) return null

  const startValue = series[0].value
  const yFormatter = (v: number) => `₩${(v / 1_000_000).toFixed(1)}M`
  const tooltipFormatter = (v: number) => [`₩${v.toLocaleString()}`, '통합총자산']

  return (
    <div className="rounded-xl border border-gray-100 bg-white p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[11px] uppercase tracking-wide text-gray-500">
          📈 Net Equity Curve · 통합 · {days}d
        </div>
        <div className="text-[11px] text-gray-500">
          {series.length} points
        </div>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={series} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="eq-curve" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#6b7280' }} minTickGap={30} />
          <YAxis
            tick={{ fontSize: 10, fill: '#6b7280' }}
            tickFormatter={yFormatter}
            domain={['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}
            formatter={tooltipFormatter}
            labelStyle={{ color: '#374151', fontSize: 11 }}
          />
          <ReferenceLine
            y={startValue}
            stroke="#9ca3af"
            strokeDasharray="3 3"
            label={{ value: 'start', fontSize: 10, fill: '#6b7280', position: 'left' }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke="#10b981"
            strokeWidth={2}
            fill="url(#eq-curve)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
