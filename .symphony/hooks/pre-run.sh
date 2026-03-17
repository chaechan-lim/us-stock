#!/bin/bash
# Pre-run hook: runs before the agent starts in a worktree
set -e

echo "[symphony] Pre-run hook for us-stock"

# --- venv setup ---
# Resolve the source repo's venv via git worktree metadata.
# Worktrees store the main repo path in .git (a file, not a directory).
resolve_source_venv() {
    # 1. Try git worktree list to find the base worktree (main repo)
    local base_dir
    base_dir=$(git worktree list --porcelain 2>/dev/null | head -1 | sed 's/^worktree //')
    if [ -n "$base_dir" ] && [ -d "$base_dir/venv" ]; then
        echo "$base_dir/venv"
        return
    fi

    # 2. Fallback: hardcoded source path
    if [ -d "/home/chans/us-stock/venv" ]; then
        echo "/home/chans/us-stock/venv"
        return
    fi

    return 1
}

# Create symlink to original venv if not present
if [ ! -d "venv" ] && [ ! -L "venv" ]; then
    SOURCE_VENV=$(resolve_source_venv) || {
        echo "[symphony] ERROR: Cannot find source venv — tried git worktree base and /home/chans/us-stock/venv"
        exit 1
    }
    echo "[symphony] Creating venv symlink -> $SOURCE_VENV"
    ln -s "$SOURCE_VENV" venv
fi

# Verify venv works (even if symlink existed, it could be dangling)
if [ -x "venv/bin/python" ]; then
    echo "[symphony] Python: $(venv/bin/python --version)"
else
    echo "[symphony] ERROR: venv/bin/python not executable (symlink target missing?)"
    echo "[symphony]   symlink: $(readlink -f venv 2>/dev/null || echo 'not a symlink')"
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
