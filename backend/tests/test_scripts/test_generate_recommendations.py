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
