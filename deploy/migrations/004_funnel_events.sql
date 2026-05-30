-- Hermes Phase 3 (counterfactual replay) data collection layer.
-- Every BUY signal evaluation writes one row here. 30-day retention
-- enforced by a daily cleanup task in the scheduler.
CREATE TABLE IF NOT EXISTS funnel_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    market VARCHAR(2) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strategy_name VARCHAR(50),
    signal_confidence DOUBLE PRECISION,
    decision VARCHAR(10) NOT NULL,
    reject_reason VARCHAR(50),
    price DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_funnel_ts ON funnel_events(ts);
CREATE INDEX IF NOT EXISTS idx_funnel_market_ts ON funnel_events(market, ts);
CREATE INDEX IF NOT EXISTS idx_funnel_decision_reason
    ON funnel_events(decision, reject_reason);
