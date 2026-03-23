"""Tests for nginx reverse proxy configuration.

Validates that the deploy/nginx/us-stock config contains required directives
for HTTPS, HTTP→HTTPS redirect, and reverse proxy routing.
"""

import re
from pathlib import Path

import pytest

# Path to the nginx config relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
NGINX_CONFIG_PATH = PROJECT_ROOT / "deploy" / "nginx" / "us-stock"


@pytest.fixture
def nginx_config() -> str:
    """Read the nginx config file."""
    assert NGINX_CONFIG_PATH.exists(), f"nginx config not found at {NGINX_CONFIG_PATH}"
    return NGINX_CONFIG_PATH.read_text()


class TestNginxConfig:
    """Tests for deploy/nginx/us-stock configuration."""

    def test_config_file_exists(self) -> None:
        """Nginx config file must exist in deploy/nginx/."""
        assert NGINX_CONFIG_PATH.exists()

    def test_listens_on_8443_ssl(self, nginx_config: str) -> None:
        """Must listen on port 8443 with SSL."""
        assert "listen 8443 ssl" in nginx_config

    def test_ssl_certificate_directives(self, nginx_config: str) -> None:
        """Must have SSL certificate and key directives."""
        assert "ssl_certificate " in nginx_config
        assert "ssl_certificate_key " in nginx_config

    def test_error_page_497_redirect(self, nginx_config: str) -> None:
        """Must have error_page 497 for HTTP→HTTPS redirect.

        When a plain HTTP request hits an SSL-only port, nginx returns
        status 497. This directive redirects to HTTPS automatically.
        This is the core fix for STOCK-44.

        The directive must be exactly `error_page 497 =301 https://...`
        — no extra status codes between 497 and =301, otherwise nginx
        would also intercept legitimate backend 301 responses.
        """
        # Regex validates: only 497 before =301, no extra codes
        assert re.search(
            r"error_page\s+497\s+=301\s+https://\$host:\$server_port\$request_uri",
            nginx_config,
        ), (
            "error_page directive must be: error_page 497 =301 https://$host:$server_port$request_uri"
        )

    def test_frontend_proxy(self, nginx_config: str) -> None:
        """Must proxy root location to frontend (port 3001)."""
        assert "proxy_pass http://127.0.0.1:3001" in nginx_config

    def test_backend_api_proxy(self, nginx_config: str) -> None:
        """Must proxy /api/ to backend (port 8001)."""
        assert "location /api/" in nginx_config
        assert "proxy_pass http://127.0.0.1:8001" in nginx_config

    def test_websocket_proxy(self, nginx_config: str) -> None:
        """Must have WebSocket proxy support for /ws/."""
        assert "location /ws/" in nginx_config
        assert "proxy_http_version 1.1" in nginx_config
        assert "Upgrade $http_upgrade" in nginx_config

    def test_health_endpoint(self, nginx_config: str) -> None:
        """Must proxy /health to backend."""
        assert "location /health" in nginx_config

    def test_forwarded_headers(self, nginx_config: str) -> None:
        """Must set X-Forwarded-* headers for proper request tracking."""
        assert "X-Real-IP" in nginx_config
        assert "X-Forwarded-For" in nginx_config
        assert "X-Forwarded-Proto" in nginx_config

    def test_no_hardcoded_server_name(self, nginx_config: str) -> None:
        """Server name should be wildcard, not hardcoded hostname."""
        assert "server_name _" in nginx_config
