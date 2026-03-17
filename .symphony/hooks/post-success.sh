#!/bin/bash
# Post-success hook: runs after successful agent completion
set -e

echo "[symphony] Post-success validation for us-stock"

# Run full test suite as final gate
echo "[symphony] Running full test suite..."
venv/bin/python -m pytest backend/tests/ -x -q

# Check test count didn't decrease
TEST_COUNT=$(venv/bin/python -m pytest backend/tests/ --co -q 2>/dev/null | tail -1 | grep -oP '\d+')
if [ "$TEST_COUNT" -lt 1400 ]; then
    echo "[symphony] ERROR: Test count decreased to $TEST_COUNT (minimum: 1400)"
    exit 1
fi

# Run lint
if command -v venv/bin/ruff &> /dev/null; then
    echo "[symphony] Running lint..."
    venv/bin/ruff check backend/
fi

echo "[symphony] Post-success validation passed ($TEST_COUNT tests)"
