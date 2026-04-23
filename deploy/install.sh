#!/bin/bash
# Install us-stock systemd services and nginx config.
# Run: sudo bash deploy/install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Installing us-stock systemd services..."

# Copy service files
sudo cp "$SCRIPT_DIR/usstock-backend.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/usstock-frontend.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/signal-quality-snapshot.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/signal-quality-snapshot.timer" /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable usstock-backend.service
sudo systemctl enable usstock-frontend.service
# Weekly SignalQualityTracker snapshot (Mon 07:00 KST) — feeds backtests
# so they mirror the live tracker's gating + Kelly inputs.
sudo systemctl enable --now signal-quality-snapshot.timer

# Install nginx config (if nginx is installed)
if command -v nginx &> /dev/null; then
    echo ""
    echo "Installing nginx config..."
    sudo cp "$SCRIPT_DIR/nginx/us-stock" /etc/nginx/sites-available/us-stock
    sudo ln -sf /etc/nginx/sites-available/us-stock /etc/nginx/sites-enabled/us-stock
    if sudo nginx -t; then
        sudo systemctl reload nginx
        echo "  nginx config installed and reloaded."
    else
        echo "  WARNING: nginx config test failed. Config copied but NOT reloaded."
    fi
else
    echo ""
    echo "nginx not found — skipping nginx config install."
    echo "  To install manually:"
    echo "    sudo cp deploy/nginx/us-stock /etc/nginx/sites-available/us-stock"
    echo "    sudo ln -sf /etc/nginx/sites-available/us-stock /etc/nginx/sites-enabled/us-stock"
fi

echo ""
echo "Services installed. To start:"
echo "  sudo systemctl start usstock-backend"
echo "  sudo systemctl start usstock-frontend"
echo ""
echo "To check status:"
echo "  sudo systemctl status usstock-backend"
echo "  sudo systemctl status usstock-frontend"
echo ""
echo "Ports:"
echo "  Backend:  http://localhost:8001"
echo "  Frontend: http://localhost:3001"
echo "  HTTPS:    https://$(hostname):8443"
