import { useEngineStatus, useEngineControl } from '../hooks/useApi'
import { useMutation } from '@tanstack/react-query'
import { runEvaluation } from '../api/client'
import clsx from 'clsx'

const phaseBadge: Record<string, string> = {
  regular: 'bg-green-600 text-green-100',
  pre_market: 'bg-blue-600 text-blue-100',
  after_hours: 'bg-orange-600 text-orange-100',
  closed: 'bg-gray-600 text-gray-300',
}

export default function EngineControl() {
  const { data: status } = useEngineStatus()
  const { start, stop } = useEngineControl()
  const evaluate = useMutation({ mutationFn: runEvaluation })

  const running = status?.running ?? false
  const phase = status?.market_phase ?? ''

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <div
          className={clsx(
            'w-2.5 h-2.5 rounded-full',
            running ? 'bg-green-500 animate-pulse' : 'bg-gray-600'
          )}
        />
        <span className="text-sm text-gray-300">
          {running ? 'Running' : 'Stopped'}
        </span>
      </div>

      {running ? (
        <button
          onClick={() => stop.mutate()}
          disabled={stop.isPending}
          className="px-3 py-1 text-xs font-medium bg-red-600 hover:bg-red-700 rounded transition-colors disabled:opacity-50"
        >
          Stop
        </button>
      ) : (
        <button
          onClick={() => start.mutate()}
          disabled={start.isPending}
          className="px-3 py-1 text-xs font-medium bg-green-600 hover:bg-green-700 rounded transition-colors disabled:opacity-50"
        >
          Start
        </button>
      )}

      <button
        onClick={() => evaluate.mutate()}
        disabled={evaluate.isPending}
        className="px-3 py-1 text-xs font-medium bg-amber-600 hover:bg-amber-700 rounded transition-colors disabled:opacity-50"
      >
        {evaluate.isPending ? 'Evaluating...' : 'Run Evaluate'}
      </button>

      {phase && (
        <span
          className={clsx(
            'px-2 py-0.5 text-xs font-medium rounded-full uppercase',
            phaseBadge[phase] ?? 'bg-gray-600 text-gray-300'
          )}
        >
          {phase.replace('_', ' ')}
        </span>
      )}
    </div>
  )
}
