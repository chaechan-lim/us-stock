import { createContext, useContext, useState, type ReactNode } from 'react'
import type { AccountInfo } from '../types'

/** Predefined color palette for up to 8 accounts. */
export const ACCOUNT_COLORS = [
  '#3b82f6', // blue-500
  '#8b5cf6', // violet-500
  '#10b981', // emerald-500
  '#f59e0b', // amber-500
  '#ef4444', // red-500
  '#06b6d4', // cyan-500
  '#f97316', // orange-500
  '#6366f1', // indigo-500
]

export function getAccountColor(index: number): string {
  return ACCOUNT_COLORS[index % ACCOUNT_COLORS.length]
}

interface AccountContextType {
  /** All configured accounts from /accounts/ API. */
  accounts: AccountInfo[]
  /** Currently selected account ID, or null for "전체" (all accounts). */
  selectedAccountId: string | null
  setSelectedAccountId: (id: string | null) => void
  /** Convenience: resolved AccountInfo for selectedAccountId (undefined when "전체"). */
  selectedAccount: AccountInfo | undefined
  /** Color assigned to an account by its position in the list. */
  accountColor: (accountId: string) => string
}

const AccountContext = createContext<AccountContextType>({
  accounts: [],
  selectedAccountId: null,
  setSelectedAccountId: () => {},
  selectedAccount: undefined,
  accountColor: () => ACCOUNT_COLORS[0],
})

interface AccountProviderProps {
  children: ReactNode
  accounts: AccountInfo[]
}

export function AccountProvider({ children, accounts }: AccountProviderProps) {
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null)

  const selectedAccount = accounts.find(a => a.account_id === selectedAccountId)

  const accountColor = (accountId: string): string => {
    const idx = accounts.findIndex(a => a.account_id === accountId)
    return getAccountColor(idx >= 0 ? idx : 0)
  }

  return (
    <AccountContext.Provider
      value={{ accounts, selectedAccountId, setSelectedAccountId, selectedAccount, accountColor }}
    >
      {children}
    </AccountContext.Provider>
  )
}

export function useAccount(): AccountContextType {
  return useContext(AccountContext)
}
