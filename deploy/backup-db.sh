#!/bin/bash
# PostgreSQL backup script for us-stock + coin databases.
# Runs pg_dump inside the Docker container, compresses with gzip.
#
# Local: daily backup, 7-day retention
# Remote: weekly push to GitHub private repo
#
# Usage:
#   ./backup-db.sh              # backup both DBs (local)
#   ./backup-db.sh us_stock     # backup us_stock only
#   ./backup-db.sh coin         # backup coin only
#   ./backup-db.sh --push       # push latest backups to GitHub
#   ./backup-db.sh --restore <file>  # restore from backup
#   ./backup-db.sh --list       # list available backups

set -euo pipefail

CONTAINER="coin-postgres-1"
BACKUP_DIR="/home/chans/backups/db"
REMOTE_REPO_DIR="/home/chans/backups/db-backups-repo"
RETENTION_DAYS=7
DATE=$(date +%Y%m%d_%H%M%S)

# DB configs: name|user
DATABASES=(
    "us_stock_trading|usstock"
    "coin_trading|coin"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

backup_db() {
    local db_name=$1
    local db_user=$2
    local backup_file="${BACKUP_DIR}/${db_name}_${DATE}.sql.gz"

    log "Backing up ${db_name}..."

    if ! docker exec "${CONTAINER}" pg_dump -U "${db_user}" -d "${db_name}" --no-owner --no-acl \
        | gzip > "${backup_file}"; then
        log "ERROR: Backup failed for ${db_name}"
        rm -f "${backup_file}"
        return 1
    fi

    local size
    size=$(du -h "${backup_file}" | cut -f1)
    log "OK: ${backup_file} (${size})"
}

cleanup() {
    log "Cleaning up backups older than ${RETENTION_DAYS} days..."
    local count
    count=$(find "${BACKUP_DIR}" -name "*.sql.gz" -mtime "+${RETENTION_DAYS}" | wc -l)
    find "${BACKUP_DIR}" -name "*.sql.gz" -mtime "+${RETENTION_DAYS}" -delete
    log "Removed ${count} old backup(s)"
}

push_to_remote() {
    if [[ ! -d "${REMOTE_REPO_DIR}/.git" ]]; then
        log "ERROR: Remote repo not initialized at ${REMOTE_REPO_DIR}"
        log "Run: git clone git@github-usstock:chaechan-lim/db-backups.git ${REMOTE_REPO_DIR}"
        return 1
    fi

    log "Pushing backups to GitHub..."

    # Copy latest backup for each DB to repo
    for entry in "${DATABASES[@]}"; do
        IFS='|' read -r db_name _ <<< "${entry}"
        local latest
        latest=$(ls -t "${BACKUP_DIR}/${db_name}_"*.sql.gz 2>/dev/null | head -1)
        if [[ -n "${latest}" ]]; then
            cp "${latest}" "${REMOTE_REPO_DIR}/"
            log "Copied $(basename "${latest}")"
        fi
    done

    # Keep only last 4 weeks of backups in repo
    cd "${REMOTE_REPO_DIR}"
    for entry in "${DATABASES[@]}"; do
        IFS='|' read -r db_name _ <<< "${entry}"
        local file_count
        file_count=$(ls -1 "${db_name}_"*.sql.gz 2>/dev/null | wc -l)
        if (( file_count > 4 )); then
            ls -t "${db_name}_"*.sql.gz | tail -n +"5" | xargs rm -f
            log "Trimmed ${db_name} backups to 4 in repo"
        fi
    done

    git add -A
    if git diff --cached --quiet; then
        log "No changes to push"
        return 0
    fi

    git commit -m "backup: $(date '+%Y-%m-%d %H:%M')"
    git push
    log "OK: Pushed to GitHub"
}

restore_db() {
    local backup_file=$1

    if [[ ! -f "${backup_file}" ]]; then
        log "ERROR: File not found: ${backup_file}"
        exit 1
    fi

    local filename
    filename=$(basename "${backup_file}")
    local db_name
    local db_user

    if [[ "${filename}" == us_stock_trading_* ]]; then
        db_name="us_stock_trading"
        db_user="usstock"
    elif [[ "${filename}" == coin_trading_* ]]; then
        db_name="coin_trading"
        db_user="coin"
    else
        log "ERROR: Cannot determine DB from filename: ${filename}"
        log "Expected format: <db_name>_YYYYMMDD_HHMMSS.sql.gz"
        exit 1
    fi

    log "WARNING: This will DROP and recreate all tables in ${db_name}!"
    read -p "Continue? (yes/no): " confirm
    if [[ "${confirm}" != "yes" ]]; then
        log "Aborted."
        exit 0
    fi

    log "Restoring ${db_name} from ${backup_file}..."
    gunzip -c "${backup_file}" | docker exec -i "${CONTAINER}" psql -U "${db_user}" -d "${db_name}"
    log "OK: Restore complete"
}

list_backups() {
    log "=== Local backups ==="
    ls -lht "${BACKUP_DIR}"/*.sql.gz 2>/dev/null || log "(none)"
    if [[ -d "${REMOTE_REPO_DIR}/.git" ]]; then
        log "=== Remote backups ==="
        ls -lht "${REMOTE_REPO_DIR}"/*.sql.gz 2>/dev/null || log "(none)"
    fi
}

# --- Main ---

mkdir -p "${BACKUP_DIR}"

case "${1:-all}" in
    --restore)
        restore_db "${2:?Usage: $0 --restore <file>}"
        ;;
    --list)
        list_backups
        ;;
    --push)
        push_to_remote
        ;;
    us_stock)
        backup_db "us_stock_trading" "usstock"
        cleanup
        ;;
    coin)
        backup_db "coin_trading" "coin"
        cleanup
        ;;
    all)
        for entry in "${DATABASES[@]}"; do
            IFS='|' read -r db_name db_user <<< "${entry}"
            backup_db "${db_name}" "${db_user}" || true
        done
        cleanup
        ;;
    *)
        echo "Usage: $0 [all|us_stock|coin|--list|--push|--restore <file>]"
        exit 1
        ;;
esac

log "Done."
