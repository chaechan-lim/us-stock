-- Migration: Add 'market' column to support KR stock trading
-- Run: psql -U usstock -d us_stock_trading -f deploy/migrations/001_add_market_column.sql

-- Orders
ALTER TABLE orders ADD COLUMN IF NOT EXISTS market VARCHAR(2) NOT NULL DEFAULT 'US';

-- Positions
ALTER TABLE positions ADD COLUMN IF NOT EXISTS market VARCHAR(2) NOT NULL DEFAULT 'US';

-- Portfolio snapshots
ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS market VARCHAR(2) NOT NULL DEFAULT 'US';

-- Watchlist
ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS market VARCHAR(2) NOT NULL DEFAULT 'US';
