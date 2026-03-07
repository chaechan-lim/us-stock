#!/usr/bin/env python3
"""KIS API connection test script.

Tests authentication, market data, and account endpoints.
Run: python -m scripts.test_kis_connection
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig
from exchange.kis_auth import KISAuth
from exchange.kis_adapter import KISAdapter


async def main():
    config = AppConfig()

    if not config.kis.app_key or config.kis.app_key == "your_app_key_here":
        print("ERROR: KIS API credentials not configured.")
        print("Set KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO in .env")
        sys.exit(1)

    is_paper = "vts" in config.kis.base_url
    mode = "PAPER" if is_paper else "LIVE"
    print(f"=== KIS API Connection Test ({mode}) ===")
    print(f"Base URL: {config.kis.base_url}")
    print(f"Account: {config.kis.account_no}")
    print()

    # 1. Auth test
    print("[1/5] Testing authentication...")
    auth = KISAuth(
        app_key=config.kis.app_key,
        app_secret=config.kis.app_secret,
        base_url=config.kis.base_url,
    )
    try:
        await auth.initialize()
        token = auth.access_token
        print(f"  OK - Token obtained: {token[:20]}...")
    except Exception as e:
        print(f"  FAIL - Auth error: {e}")
        await auth.close()
        sys.exit(1)

    # 2. KIS Adapter test
    adapter = KISAdapter(config.kis, auth)
    await adapter.initialize()

    # 3. Market data test
    print("[2/5] Testing market data (AAPL ticker)...")
    try:
        ticker = await adapter.fetch_ticker("AAPL")
        print(f"  OK - AAPL: ${ticker.price:,.2f}, vol={ticker.volume:,}")
    except Exception as e:
        print(f"  FAIL - {e}")

    # 4. OHLCV test
    print("[3/5] Testing OHLCV data (AAPL daily, 5 bars)...")
    try:
        candles = await adapter.fetch_ohlcv("AAPL", timeframe="1D", limit=5)
        print(f"  OK - Got {len(candles)} candles")
        if candles:
            c = candles[-1]
            print(f"  Latest: O={c.open:.2f} H={c.high:.2f} L={c.low:.2f} C={c.close:.2f} V={c.volume:,}")
    except Exception as e:
        print(f"  FAIL - {e}")

    # 5. Balance test
    print("[4/5] Testing account balance...")
    try:
        balance = await adapter.fetch_balance()
        print(f"  OK - Total: ${balance.total:,.2f} | Available: ${balance.available:,.2f} | Locked: ${balance.locked:,.2f}")
    except Exception as e:
        print(f"  FAIL - {e}")

    # 6. Positions test
    print("[5/5] Testing positions...")
    try:
        positions = await adapter.fetch_positions()
        print(f"  OK - {len(positions)} position(s)")
        for p in positions:
            print(f"    {p.symbol}: {p.quantity} shares @ ${p.avg_price:,.2f} | "
                  f"Current: ${p.current_price:,.2f} | PnL: ${p.unrealized_pnl:,.2f}")
    except Exception as e:
        print(f"  FAIL - {e}")

    # Cleanup
    await adapter.close()
    await auth.close()

    print()
    print("=== Connection test complete ===")


if __name__ == "__main__":
    asyncio.run(main())
