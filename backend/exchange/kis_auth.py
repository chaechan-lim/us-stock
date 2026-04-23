"""KIS API OAuth2 token manager.

Handles:
- Token issuance (1/day limit)
- Redis caching (survives restarts)
- Auto-refresh before expiry
- WebSocket approval key
"""

import time
import json
import logging

import aiohttp

logger = logging.getLogger(__name__)

TOKEN_VALIDITY_SEC = 86400  # 24 hours
REFRESH_BEFORE_SEC = 3600   # refresh 1 hour before expiry
REDIS_TOKEN_KEY = "kis:access_token"
REDIS_APPROVAL_KEY = "kis:approval_key"


def is_token_error(resp_data: dict) -> bool:
    """Detect KIS server-side token invalidation responses.

    Covers cases `_should_refresh()`'s local clock-only check misses
    (server expired the token early / cross-device auth rotation).
    만료 and 유효하지 않은 phrases are the two visible msg1 forms.
    """
    msg = resp_data.get("msg1", "")
    if "만료된 token" in msg or "유효하지 않은 token" in msg:
        return True
    msg_cd = resp_data.get("msg_cd", "")
    return msg_cd in ("EGW00121", "EGW00122", "EGW00123", "EGW00124", "EGW00125")


class KISAuth:
    def __init__(
        self,
        app_key: str,
        app_secret: str,
        base_url: str,
        redis_client=None,
    ):
        self._app_key = app_key
        self._app_secret = app_secret
        self._base_url = base_url
        self._redis = redis_client

        self._access_token: str | None = None
        self._token_expires_at: float = 0
        self._approval_key: str | None = None
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        if self._session and not self._session.closed:
            # Already initialized — skip to avoid session leak
            return
        self._session = aiohttp.ClientSession()
        # Try to restore token from Redis first (avoid 1/day limit)
        if self._redis:
            await self._restore_from_redis()
        if not self._access_token or self._is_token_expired():
            await self._issue_token()

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    @property
    def access_token(self) -> str:
        if not self._access_token:
            raise RuntimeError("KIS auth not initialized. Call initialize() first.")
        return self._access_token

    @property
    def app_key(self) -> str:
        return self._app_key

    @property
    def app_secret(self) -> str:
        return self._app_secret

    @property
    def approval_key(self) -> str | None:
        return self._approval_key

    def get_auth_headers(self, tr_id: str, hashkey: str = "") -> dict[str, str]:
        """Build common request headers for KIS API calls."""
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        if hashkey:
            headers["hashkey"] = hashkey
        return headers

    async def ensure_valid_token(self) -> None:
        """Check and refresh token if needed. Called before each API request."""
        if self._should_refresh():
            logger.info("KIS token approaching expiry, refreshing...")
            await self._issue_token()

    async def force_refresh(self) -> None:
        """Force re-issuance regardless of local expiry clock.

        Called by adapters when the server returns "expired/invalid token"
        even though our local clock still thinks the token is valid — the
        KIS server can invalidate tokens early (e.g. cross-device auth,
        server-side expiry mismatch). Without this, `_should_refresh()`
        would never trigger and the backend would keep failing silently
        until a restart.
        """
        logger.info("KIS token server-rejected, forcing re-issue...")
        await self._issue_token()

    async def get_approval_key(self) -> str:
        """Get WebSocket approval key."""
        if self._approval_key:
            return self._approval_key

        url = f"{self._base_url}/oauth2/Approval"
        body = {
            "grant_type": "client_credentials",
            "appkey": self._app_key,
            "secretkey": self._app_secret,
        }

        async with self._session.post(url, json=body) as resp:
            data = await resp.json()
            key = data.get("approval_key")
            if not key:
                logger.error("KIS approval key missing from response: %s", data)
                raise RuntimeError("Failed to obtain KIS WebSocket approval key")
            self._approval_key = key

        if self._redis:
            await self._redis.set(
                REDIS_APPROVAL_KEY, self._approval_key, ex=TOKEN_VALIDITY_SEC
            )

        logger.info("KIS WebSocket approval key obtained")
        return self._approval_key

    async def get_hashkey(self, body: dict) -> str:
        """Generate hashkey for order requests (POST body integrity)."""
        url = f"{self._base_url}/uapi/hashkey"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
        }

        async with self._session.post(url, headers=headers, json=body) as resp:
            data = await resp.json()
            return data.get("HASH", "")

    # -- Private --

    async def _issue_token(self) -> None:
        url = f"{self._base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
        }

        async with self._session.post(url, json=body) as resp:
            data = await resp.json()
            if "access_token" not in data:
                raise RuntimeError(f"KIS token issuance failed: {data}")

            self._access_token = data["access_token"]
            self._token_expires_at = time.time() + TOKEN_VALIDITY_SEC

        if self._redis:
            token_data = json.dumps({
                "access_token": self._access_token,
                "expires_at": self._token_expires_at,
            })
            await self._redis.set(
                REDIS_TOKEN_KEY, token_data, ex=TOKEN_VALIDITY_SEC - 60
            )

        logger.info("KIS access token issued (valid 24h)")

    async def _restore_from_redis(self) -> None:
        raw = await self._redis.get(REDIS_TOKEN_KEY)
        if raw:
            data = json.loads(raw)
            self._access_token = data["access_token"]
            self._token_expires_at = data["expires_at"]
            logger.info("KIS token restored from Redis cache")

        raw_approval = await self._redis.get(REDIS_APPROVAL_KEY)
        if raw_approval:
            self._approval_key = raw_approval if isinstance(raw_approval, str) else raw_approval.decode()

    def _is_token_expired(self) -> bool:
        return time.time() >= self._token_expires_at

    def _should_refresh(self) -> bool:
        return time.time() >= (self._token_expires_at - REFRESH_BEFORE_SEC)
