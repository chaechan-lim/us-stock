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

# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable usstock-backend.service
sudo systemctl enable usstock-frontend.service

# Install nginx config (if nginx is installed)
if command -v nginx &> /dev/null; then
    echo ""
    echo "Installing nginx config..."
    sudo cp "$SCRIPT_DIR/nginx/us-stock" /etc/nginx/sites-enabled/us-stock
    if sudo nginx -t 2>/dev/null; then
        sudo systemctl reload nginx
        echo "  nginx config installed and reloaded."
    else
        echo "  WARNING: nginx config test failed. Check: sudo nginx -t"
    fi
else
    echo ""
    echo "nginx not found — skipping nginx config install."
    echo "  To install manually: sudo cp deploy/nginx/us-stock /etc/nginx/sites-enabled/us-stock"
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
echo "  HTTPS:    https://rpi-server:8443"
