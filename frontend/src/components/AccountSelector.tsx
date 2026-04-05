import { useState, useRef, useEffect } from 'react'
import { useAccount } from '../contexts/AccountContext'

/**
 * AccountSelector — header dropdown for selecting a trading account.
 *
 * Renders a compact button showing the active account (or "전체").
 * On click, opens a dropdown listing all accounts plus the "전체" option.
 * Account colors distinguish entries visually. Paper accounts show a badge.
 */
export default function AccountSelector() {
  const { accounts, selectedAccountId, setSelectedAccountId, selectedAccount, accountColor } =
    useAccount()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [])

  if (accounts.length === 0) return null

  const activeColor = selectedAccountId ? accountColor(selectedAccountId) : '#6b7280'
  const activeLabel = selectedAccount ? selectedAccount.name : '전체'

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(prev => !prev)}
        className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded-md text-sm font-medium transition-colors border border-gray-700"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        {/* Color indicator dot */}
        <span
          className="w-2 h-2 rounded-full flex-shrink-0"
          style={{ backgroundColor: activeColor }}
        />
        <span className="text-white max-w-[140px] truncate">{activeLabel}</span>
        {selectedAccount?.is_paper && (
          <span className="text-[10px] px-1 py-0.5 rounded bg-amber-900/50 text-amber-300 flex-shrink-0">
            Paper
          </span>
        )}
        {/* Chevron */}
        <svg
          className={`w-3 h-3 text-gray-400 transition-transform flex-shrink-0 ${open ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div
          className="absolute right-0 mt-1 w-64 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 py-1"
          role="listbox"
        >
          {/* "전체" option */}
          <button
            role="option"
            aria-selected={selectedAccountId === null}
            onClick={() => {
              setSelectedAccountId(null)
              setOpen(false)
            }}
            className={`w-full flex items-center gap-3 px-3 py-2 text-sm text-left hover:bg-gray-700 transition-colors ${
              selectedAccountId === null ? 'bg-gray-700/60' : ''
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-gray-500 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="font-medium text-white">전체</div>
              <div className="text-xs text-gray-400">모든 계좌 통합 뷰</div>
            </div>
            {selectedAccountId === null && (
              <svg
                className="w-4 h-4 text-blue-400 flex-shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            )}
          </button>

          {accounts.length > 0 && (
            <div className="border-t border-gray-700 my-1" />
          )}

          {/* Per-account options */}
          {accounts.map((account) => {
            const color = accountColor(account.account_id)
            const isSelected = selectedAccountId === account.account_id
            return (
              <button
                key={account.account_id}
                role="option"
                aria-selected={isSelected}
                onClick={() => {
                  setSelectedAccountId(account.account_id)
                  setOpen(false)
                }}
                className={`w-full flex items-center gap-3 px-3 py-2 text-sm text-left hover:bg-gray-700 transition-colors ${
                  isSelected ? 'bg-gray-700/60' : ''
                }`}
              >
                <span
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ backgroundColor: color }}
                />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-white truncate">{account.name}</div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    {account.markets.map(m => (
                      <span
                        key={m}
                        className={`text-[10px] px-1 py-0.5 rounded ${
                          m === 'KR'
                            ? 'bg-purple-900/40 text-purple-300'
                            : 'bg-blue-900/40 text-blue-300'
                        }`}
                      >
                        {m}
                      </span>
                    ))}
                    {account.is_paper && (
                      <span className="text-[10px] px-1 py-0.5 rounded bg-amber-900/50 text-amber-300">
                        Paper
                      </span>
                    )}
                    <span className="text-[10px] text-gray-500">{account.account_id}</span>
                  </div>
                </div>
                {isSelected && (
                  <svg
                    className="w-4 h-4 text-blue-400 flex-shrink-0"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
