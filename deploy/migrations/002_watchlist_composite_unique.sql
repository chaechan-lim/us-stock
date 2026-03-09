-- Migration 002: Change watchlist unique constraint from symbol-only to (market, symbol)
-- This allows the same symbol code to exist in both US and KR markets

-- Drop old unique constraint on symbol
ALTER TABLE watchlist DROP CONSTRAINT IF EXISTS watchlist_symbol_key;
DROP INDEX IF EXISTS watchlist_symbol_key;

-- Add composite unique constraint
ALTER TABLE watchlist ADD CONSTRAINT uq_watchlist_market_symbol UNIQUE (market, symbol);
