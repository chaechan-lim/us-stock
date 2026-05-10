"""F1: GET /engine/rejection-funnel response contract."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.engine_api import router as engine_router


def _mock_loop(rejects: dict[str, int], signals: int, placed: int, **extras):
    base = {
        "_reject_counters": dict(rejects),
        "_buy_flow_counters": {
            "buy_signals_total": signals,
            "buys_placed": placed,
        },
        "_daily_buy_count": placed,
        "_daily_buy_limit": 5,
        "_daily_buy_date": "2026-05-10",
    }
    base.update(extras)
    return SimpleNamespace(**base)


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(engine_router, prefix="/api/v1")
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestRejectionFunnelEndpoint:
    def test_returns_both_markets(self, app, client):
        app.state.evaluation_loop = _mock_loop(
            {"opening_avoidance": 3, "daily_limit": 1}, signals=10, placed=4
        )
        app.state.kr_evaluation_loop = _mock_loop(
            {"sector_limit": 2}, signals=5, placed=1
        )

        resp = client.get("/api/v1/engine/rejection-funnel")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"US", "KR"}

        us = data["US"]
        assert us["buy_signals_total"] == 10
        assert us["buys_placed"] == 4
        assert us["rejected_total"] == 4
        assert us["fill_rate"] == 0.4
        # rejections sorted by count desc
        assert list(us["rejections"].keys()) == ["opening_avoidance", "daily_limit"]
        assert us["daily_buy_limit"] == 5
        assert us["daily_buy_date"] == "2026-05-10"

    def test_omits_market_when_loop_missing(self, app, client):
        app.state.evaluation_loop = _mock_loop({}, signals=2, placed=2)
        # kr_evaluation_loop intentionally not set

        resp = client.get("/api/v1/engine/rejection-funnel")
        assert resp.status_code == 200
        assert "US" in resp.json()
        assert "KR" not in resp.json()

    def test_fill_rate_none_when_no_signals(self, app, client):
        app.state.evaluation_loop = _mock_loop({}, signals=0, placed=0)
        resp = client.get("/api/v1/engine/rejection-funnel")
        assert resp.json()["US"]["fill_rate"] is None

    def test_empty_when_no_loops(self, app, client):
        resp = client.get("/api/v1/engine/rejection-funnel")
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_handles_loop_without_funnel_attrs(self, app, client):
        # Defensive: an EvaluationLoop instance without F1 attrs should not 500
        app.state.evaluation_loop = SimpleNamespace()  # no _reject_counters
        resp = client.get("/api/v1/engine/rejection-funnel")
        assert resp.status_code == 200
        us = resp.json()["US"]
        assert us["buy_signals_total"] == 0
        assert us["buys_placed"] == 0
        assert us["rejections"] == {}
