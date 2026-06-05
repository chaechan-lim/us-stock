"""Unit tests for scripts/generate_recommendations.py pure helpers (#60).

Only the parsing / filtering / merging logic is unit-tested here. The
CLI invocations and DB write paths are exercised by the live timer in
production (and by an end-to-end --dry-run smoke if needed).
"""

from __future__ import annotations

import os
import sys

import pytest

# Make scripts/ importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import generate_recommendations as gr  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_llm_output
# ---------------------------------------------------------------------------


class TestParseLLMOutput:
    def test_clean_json(self):
        raw = '{"recommendations": [{"param_path": "markets.US.evaluation_loop.daily_buy_limit", "proposed_value": 10}]}'
        out = gr._parse_llm_output(raw)
        assert len(out) == 1
        assert out[0]["param_path"] == "markets.US.evaluation_loop.daily_buy_limit"

    def test_with_markdown_fence(self):
        raw = '```json\n{"recommendations": [{"param_path": "x", "proposed_value": 1}]}\n```'
        out = gr._parse_llm_output(raw)
        assert len(out) == 1

    def test_with_preamble_prose(self):
        raw = 'Here are my recommendations:\n\n{"recommendations": [{"param_path": "p", "proposed_value": 1}]}\n\nThanks!'
        out = gr._parse_llm_output(raw)
        assert len(out) == 1

    def test_empty_list(self):
        assert gr._parse_llm_output('{"recommendations": []}') == []

    def test_missing_required_field(self):
        # Missing proposed_value → row dropped
        raw = '{"recommendations": [{"param_path": "p"}]}'
        assert gr._parse_llm_output(raw) == []

    def test_no_json_at_all(self):
        assert gr._parse_llm_output("I refuse to comply.") == []

    def test_none_input(self):
        assert gr._parse_llm_output(None) == []

    def test_invalid_json(self):
        assert gr._parse_llm_output("{this is not json") == []

    def test_codex_preamble_then_trailing_meta(self):
        """Repro of codex exec output: workdir info, then JSON, then
        `tokens used` summary. Our balanced-brace scan picks the JSON."""
        raw = (
            "workdir: /home/chans/us-stock\n"
            "model: gpt-5.4\n"
            "codex\n"
            '{"recommendations": [{"param_path": "markets.US.evaluation_loop.daily_buy_limit", "proposed_value": 12}]}\n'
            "tokens used\n"
            "51,288\n"
        )
        out = gr._parse_llm_output(raw)
        assert len(out) == 1
        assert out[0]["proposed_value"] == 12

    def test_two_json_blocks_first_wins(self):
        """Some CLIs echo the JSON twice. The first valid match returns."""
        raw = (
            '{"recommendations": [{"param_path": "p1", "proposed_value": 1}]}\n'
            'extra prose\n'
            '{"recommendations": [{"param_path": "p2", "proposed_value": 2}]}\n'
        )
        out = gr._parse_llm_output(raw)
        assert len(out) == 1
        assert out[0]["param_path"] == "p1"

    def test_string_with_braces_does_not_break_scanner(self):
        raw = '{"recommendations": [{"param_path": "weird {nested}", "proposed_value": 1}]}'
        out = gr._parse_llm_output(raw)
        assert len(out) == 1


class TestExtractJsonBlocks:
    def test_finds_single_block(self):
        assert gr._extract_json_blocks('hi {"a": 1} bye') == ['{"a": 1}']

    def test_finds_nested_block_as_one(self):
        assert gr._extract_json_blocks('{"a": {"b": 1}}') == ['{"a": {"b": 1}}']

    def test_finds_two_blocks_sequentially(self):
        out = gr._extract_json_blocks('{"a":1}\n{"b":2}')
        assert out == ['{"a":1}', '{"b":2}']

    def test_skips_braces_inside_strings(self):
        raw = '{"k": "value with } brace"}'
        out = gr._extract_json_blocks(raw)
        assert out == [raw]

    def test_unmatched_closing_brace_does_not_break(self):
        out = gr._extract_json_blocks('}}{"ok": true}')
        assert out == ['{"ok": true}']


# ---------------------------------------------------------------------------
# _filter_whitelist
# ---------------------------------------------------------------------------


class TestFilterWhitelist:
    def test_keeps_allowed_paths(self):
        recs = [
            {"param_path": "markets.US.evaluation_loop.daily_buy_limit", "proposed_value": 10},
            {"param_path": "markets.KR.risk.max_positions", "proposed_value": 20},
        ]
        out = gr._filter_whitelist(recs)
        assert len(out) == 2

    def test_drops_non_whitelisted(self):
        recs = [
            {"param_path": "global.min_confidence", "proposed_value": 0.6},  # not allowed
            {"param_path": "markets.US.cash_parking.threshold", "proposed_value": 0.4},  # allowed
        ]
        out = gr._filter_whitelist(recs)
        assert len(out) == 1
        assert out[0]["param_path"] == "markets.US.cash_parking.threshold"

    def test_drops_path_outside_market_block(self):
        # markets.JP doesn't exist on the whitelist
        recs = [{"param_path": "markets.JP.evaluation_loop.x", "proposed_value": 1}]
        assert gr._filter_whitelist(recs) == []


# ---------------------------------------------------------------------------
# _filter_against_pending
# ---------------------------------------------------------------------------


class TestFilterAgainstPending:
    def test_drops_duplicate_path(self):
        recs = [{"param_path": "markets.US.evaluation_loop.daily_buy_limit", "proposed_value": 10}]
        out = gr._filter_against_pending(recs, {"markets.US.evaluation_loop.daily_buy_limit"})
        assert out == []

    def test_keeps_new_paths(self):
        recs = [
            {"param_path": "markets.KR.evaluation_loop.daily_buy_limit", "proposed_value": 10},
            {"param_path": "markets.US.evaluation_loop.daily_buy_limit", "proposed_value": 12},
        ]
        out = gr._filter_against_pending(
            recs, {"markets.KR.evaluation_loop.sector_boost_weight"},
        )
        assert len(out) == 2


# ---------------------------------------------------------------------------
# _merge_sources
# ---------------------------------------------------------------------------


class TestMergeSources:
    def test_same_path_same_value_collapses_to_one(self):
        claude = [{"param_path": "p", "proposed_value": 10, "rationale": "A"}]
        codex = [{"param_path": "p", "proposed_value": 10, "rationale": "B"}]
        merged = gr._merge_sources(claude, codex)
        assert len(merged) == 1
        assert merged[0]["_source"] == "both"
        assert "Codex rationale: B" in merged[0]["_notes_extra"]

    def test_same_path_diff_value_keeps_both(self):
        claude = [{"param_path": "p", "proposed_value": 10}]
        codex = [{"param_path": "p", "proposed_value": 12}]
        merged = gr._merge_sources(claude, codex)
        assert len(merged) == 2
        sources = {m["_source"] for m in merged}
        assert sources == {"claude", "codex"}

    def test_disjoint_paths_pass_through(self):
        claude = [{"param_path": "p1", "proposed_value": 1}]
        codex = [{"param_path": "p2", "proposed_value": 2}]
        merged = gr._merge_sources(claude, codex)
        assert len(merged) == 2
        sources = {m["_source"] for m in merged}
        assert sources == {"claude", "codex"}

    def test_one_side_empty(self):
        claude = [{"param_path": "p", "proposed_value": 1}]
        merged = gr._merge_sources(claude, [])
        assert len(merged) == 1
        assert merged[0]["_source"] == "claude"

    def test_both_empty(self):
        assert gr._merge_sources([], []) == []


class TestFormatExposureSection:
    """Hermes Phase 2 — prompt enrichment with exposure snapshot."""

    def test_no_exposure_returns_empty_string(self):
        assert gr._format_exposure_section(None) == ""
        assert gr._format_exposure_section({}) == ""

    def test_renders_markets_and_funnel(self):
        exposure = {
            "date": "2026-05-29",
            "markets": {
                "KR": {
                    "deployed_pct": 0.226,
                    "slot_fill_ratio": 0.323,
                    "placeholder_count": 9,
                    "position_count": 14,
                    "cash_idle_days": 1,
                    "funnel": {
                        "total_signals": 31,
                        "buys_placed": 1,
                        "top_reasons": [
                            ["sell_cooldown", 20, 0.645],
                            ["same_signal_24h", 8, 0.258],
                        ],
                    },
                },
            },
            "flags": [],
        }
        section = gr._format_exposure_section(exposure)
        assert "Exposure snapshot" in section
        assert "KR: deployed 22.6%" in section
        assert "placeholders 9/14" in section
        assert "sell_cooldown 20 (64%)" in section
        assert "same_signal_24h 8 (26%)" in section

    def test_flags_emit_priority_directive(self):
        exposure = {
            "markets": {
                "KR": {
                    "deployed_pct": 0.22,
                    "slot_fill_ratio": 0.32,
                    "placeholder_count": 9, "position_count": 14,
                    "cash_idle_days": 5,
                    "funnel": {
                        "total_signals": 31, "buys_placed": 1,
                        "top_reasons": [["sell_cooldown", 20, 0.645]],
                    },
                },
            },
            "flags": [
                {
                    "market": "KR", "flag": "chronic_under_deployment",
                    "severity": "warning",
                    "detail": "deployed 22% for 5d",
                },
            ],
        }
        section = gr._format_exposure_section(exposure)
        assert "Sentinel flags raised today" in section
        assert "chronic_under_deployment" in section
        assert "prioritize" in section


class TestFilterPhantomPaths:
    """A: drop LLM-hallucinated paths that look plausible but don't
    exist in the live yaml (e.g. sell_cooldown_hours when the real
    key is sell_cooldown_days)."""

    def test_phantom_paths_dropped_real_paths_kept(self, tmp_path, monkeypatch):
        import yaml as _yaml

        fake_yaml = tmp_path / "strategies.yaml"
        fake_yaml.write_text(_yaml.safe_dump({
            "markets": {
                "KR": {
                    "evaluation_loop": {"sell_cooldown_days": 3},
                },
            },
        }), encoding="utf-8")
        monkeypatch.setattr(gr, "YAML_PATH", fake_yaml)

        recs = [
            {"param_path": "markets.KR.evaluation_loop.sell_cooldown_days",
             "proposed_value": 2},
            {"param_path": "markets.KR.evaluation_loop.sell_cooldown_hours",
             "proposed_value": 12},  # phantom: real key is _days
            {"param_path": "markets.US.evaluation_loop.same_signal_dedup_hours",
             "proposed_value": 12},  # phantom: no such key anywhere
        ]
        kept = gr._filter_phantom_paths(recs)
        assert len(kept) == 1
        assert kept[0]["param_path"] == "markets.KR.evaluation_loop.sell_cooldown_days"

    def test_empty_input_returns_empty(self, monkeypatch, tmp_path):
        fake_yaml = tmp_path / "strategies.yaml"
        fake_yaml.write_text("markets: {}\n", encoding="utf-8")
        monkeypatch.setattr(gr, "YAML_PATH", fake_yaml)
        assert gr._filter_phantom_paths([]) == []


# ---------------------------------------------------------------------------
# _validate_and_promote — pre-submission backtest gate (2026-06-05)
# ---------------------------------------------------------------------------


class TestValidateAndPromote:
    """Verify the synchronous backtest gate flips status correctly.

    The gate must:
    - promote rows whose backtest passes the floor (passes_floor=True)
    - auto-reject rows that fail the floor (passes_floor=False)
    - leave rows in operator queue when backtest data is unusable
      (skip / error / replay-only) — defer to human judgement
    """

    @pytest.mark.asyncio
    async def test_promotes_when_passes_floor(self, monkeypatch):
        from sqlalchemy.ext.asyncio import (
            AsyncSession, async_sessionmaker, create_async_engine,
        )
        from sqlalchemy.pool import StaticPool

        from core.models import AgentRecommendation, Base

        engine = create_async_engine(
            "sqlite+aiosqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        monkeypatch.setattr(gr, "get_session_factory", lambda: factory)

        # Stub the validator: just stamp passes_floor=True
        async def stub_validate(rec_id, sf):
            async with sf() as s:
                rec = await s.get(AgentRecommendation, rec_id)
                rec.backtest_result = {
                    "baseline": {"ret": 10.0, "sharpe": 1.0, "mdd": -10.0, "pf": 1.3},
                    "proposed": {"ret": 12.0, "sharpe": 1.1, "mdd": -10.0, "pf": 1.35},
                    "delta": {"ret": 2.0, "sharpe": 0.1, "mdd": 0.0, "pf": 0.05},
                    "passes_floor": True,
                }
                await s.commit()

        import services.recommendation_validator as rv
        monkeypatch.setattr(rv, "validate_recommendation", stub_validate)

        async with factory() as s:
            rec = AgentRecommendation(
                agent_type="llm_claude_daily",
                param_path="markets.KR.risk.max_positions",
                current_value=18, proposed_value=22,
                status="pending_validation",
            )
            s.add(rec)
            await s.commit()
            rid = rec.id

        promoted, rejected = await gr._validate_and_promote([rid])
        assert promoted == [rid]
        assert rejected == []

        async with factory() as s:
            rec = await s.get(AgentRecommendation, rid)
            assert rec.status == "pending"

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_auto_rejects_when_fails_floor(self, monkeypatch):
        from sqlalchemy.ext.asyncio import (
            AsyncSession, async_sessionmaker, create_async_engine,
        )
        from sqlalchemy.pool import StaticPool

        from core.models import AgentRecommendation, Base

        engine = create_async_engine(
            "sqlite+aiosqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        monkeypatch.setattr(gr, "get_session_factory", lambda: factory)

        async def stub_validate(rec_id, sf):
            async with sf() as s:
                rec = await s.get(AgentRecommendation, rec_id)
                rec.backtest_result = {
                    "baseline": {"ret": 10.0, "sharpe": 1.0, "mdd": -10.0, "pf": 1.3},
                    "proposed": {"ret": 5.0, "sharpe": 0.3, "mdd": -20.0, "pf": 0.9},
                    "delta": {"ret": -5.0, "sharpe": -0.7, "mdd": -10.0, "pf": -0.4},
                    "passes_floor": False,
                }
                await s.commit()

        import services.recommendation_validator as rv
        monkeypatch.setattr(rv, "validate_recommendation", stub_validate)

        async with factory() as s:
            rec = AgentRecommendation(
                agent_type="llm_codex_daily",
                param_path="markets.KR.evaluation_loop.sell_cooldown_days",
                current_value=3, proposed_value=0,
                status="pending_validation",
            )
            s.add(rec)
            await s.commit()
            rid = rec.id

        promoted, rejected = await gr._validate_and_promote([rid])
        assert promoted == []
        assert rejected == [rid]

        async with factory() as s:
            rec = await s.get(AgentRecommendation, rid)
            assert rec.status == "rejected_by_backtest"
            assert "auto-reject" in (rec.rejected_reason or "")
            assert "ret=-5" in (rec.rejected_reason or "")

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_skip_result_promotes_to_pending(self, monkeypatch):
        """When backtest is unavailable (skip), defer to operator."""
        from sqlalchemy.ext.asyncio import (
            AsyncSession, async_sessionmaker, create_async_engine,
        )
        from sqlalchemy.pool import StaticPool

        from core.models import AgentRecommendation, Base

        engine = create_async_engine(
            "sqlite+aiosqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        monkeypatch.setattr(gr, "get_session_factory", lambda: factory)

        async def stub_validate(rec_id, sf):
            async with sf() as s:
                rec = await s.get(AgentRecommendation, rec_id)
                rec.backtest_result = {"skip": "path not in map"}
                await s.commit()

        import services.recommendation_validator as rv
        monkeypatch.setattr(rv, "validate_recommendation", stub_validate)

        async with factory() as s:
            rec = AgentRecommendation(
                agent_type="llm_claude_daily",
                param_path="markets.KR.evaluation_loop.opening_avoidance_minutes",
                current_value=30, proposed_value=15,
                status="pending_validation",
            )
            s.add(rec)
            await s.commit()
            rid = rec.id

        promoted, rejected = await gr._validate_and_promote([rid])
        assert promoted == [rid]
        assert rejected == []

        async with factory() as s:
            rec = await s.get(AgentRecommendation, rid)
            assert rec.status == "pending"

        await engine.dispose()
