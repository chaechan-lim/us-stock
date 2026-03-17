#!/bin/bash
# Pre-run hook: runs before the agent starts in a worktree
set -e

echo "[symphony] Pre-run hook for us-stock"

# Create symlink to original venv if not present
if [ ! -d "venv" ] && [ ! -L "venv" ]; then
    echo "[symphony] Creating venv symlink..."
    ln -s /home/chans/us-stock/venv venv
fi

# Verify venv works
if [ -x "venv/bin/python" ]; then
    echo "[symphony] Python: $(venv/bin/python --version)"
else
    echo "[symphony] ERROR: venv/bin/python not found"
    exit 1
fi

# Verify test infrastructure
echo "[symphony] Test count check..."
TEST_COUNT=$(venv/bin/python -m pytest backend/tests/ --co -q 2>/dev/null | tail -1 | grep -oP '\d+')
echo "[symphony] Collected $TEST_COUNT tests"

# Install frontend deps if missing
if [ -d "frontend" ] && [ ! -d "frontend/node_modules" ]; then
    echo "[symphony] Installing frontend dependencies..."
    cd frontend && npm install --silent && cd ..
fi

echo "[symphony] Pre-run complete"
