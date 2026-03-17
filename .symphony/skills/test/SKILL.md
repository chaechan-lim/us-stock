---
name: test
description: Run and validate the us-stock test suite (1400+ tests, 90% coverage).
---

# Test

## Commands

```bash
# Full test suite (from project root)
venv/bin/python -m pytest backend/tests/ -x -q

# With coverage
venv/bin/python -m pytest backend/tests/ --cov=backend --cov-report=term-missing --cov-fail-under=90

# Single test file
venv/bin/python -m pytest backend/tests/test_strategies/test_bollinger_rsi.py -x -q

# Scenario tests
venv/bin/python -m pytest backend/tests/scenarios/ -x -q
```

## Rules

- Every code change MUST have corresponding tests
- Run the FULL suite, not just new tests
- If tests fail, fix the code (not the tests) unless the test is wrong
- All external dependencies must be mocked (KIS, yfinance, Claude, Gemini, Finnhub, Naver)
- Test count must never decrease (currently **1400+**)
- Coverage must remain **90%+**
- DB tests use in-memory SQLite (aiosqlite), never real PostgreSQL
- Use pytest-asyncio (auto mode) for async tests
- Dual-market: test both US and KR paths when touching shared code
