"""DART corp_code map fetcher — one-time setup for KR insider service.

DART distinguishes companies by 8-digit corp_code, not the 6-digit
stock_code. To use the insider/disclosure APIs we need a stock→corp
map. This is published as a zipped XML at /api/corpCode.xml.

Run once after registering for a DART API key:

  DART_API_KEY=xxx python scripts/fetch_dart_corp_codes.py

Outputs: data/dart_corp_map.json — {stock_code: corp_code} for all
listed (stock_code non-empty) companies. ~3,000 entries.

Re-run periodically (~quarterly) to capture new IPOs / delistings.
"""

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import requests


DART_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"


def main() -> int:
    api_key = os.environ.get("DART_API_KEY", "").strip()
    if not api_key:
        print("ERROR: DART_API_KEY env not set.", file=sys.stderr)
        print("Register at https://opendart.fss.or.kr → API 인증키 신청", file=sys.stderr)
        return 1

    print(f"Fetching DART corp code map ...")
    r = requests.get(DART_CORP_CODE_URL, params={"crtfc_key": api_key}, timeout=60)
    r.raise_for_status()

    # Response is a zip file containing CORPCODE.xml
    try:
        zf = zipfile.ZipFile(io.BytesIO(r.content))
    except zipfile.BadZipFile:
        print("ERROR: response not a zip file. Body sample:", file=sys.stderr)
        print(r.content[:500], file=sys.stderr)
        return 2

    xml_bytes = zf.read("CORPCODE.xml")
    root = ET.fromstring(xml_bytes)

    out: dict[str, str] = {}
    for item in root.findall("list"):
        corp_code = (item.findtext("corp_code") or "").strip()
        stock_code = (item.findtext("stock_code") or "").strip()
        if corp_code and stock_code:
            out[stock_code] = corp_code

    out_path = Path("data/dart_corp_map.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"Wrote {len(out)} stock→corp mappings to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
