import json

import pytest

from traceburn.pricing import ModelPrice, PricingTable


def table():
    return PricingTable(
        prices={
            "openai/gpt-test": ModelPrice(2.0, 8.0, cached_input_per_mtok=0.5),
            "openai/gpt-test-mini": ModelPrice(0.4, 1.6),
            "anthropic/claude-test": ModelPrice(
                3.0, 15.0, cached_input_per_mtok=0.3, cache_write_per_mtok=3.75
            ),
        },
        as_of="2026-01-01",
    )


def test_exact_match_with_provider():
    price = table().lookup("openai", "gpt-test")
    assert price.input_per_mtok == 2.0


def test_dated_model_id_prefix_match():
    price = table().lookup("anthropic", "claude-test-20260115")
    assert price.output_per_mtok == 15.0


def test_mini_never_falls_through_to_base():
    price = table().lookup("openai", "gpt-test-mini")
    assert price.input_per_mtok == 0.4
    dated = table().lookup("openai", "gpt-test-mini-20260101")
    assert dated.input_per_mtok == 0.4


def test_non_date_suffix_does_not_match():
    assert table().lookup("openai", "gpt-test-turbo") is None


def test_unknown_model_returns_none():
    assert table().lookup("openai", "unlisted") is None
    assert table().lookup(None, None) is None


def test_cost_math_all_buckets():
    cost = table().cost(
        "anthropic",
        "claude-test",
        input_tokens=1_000_000,
        output_tokens=200_000,
        cached_input_tokens=2_000_000,
        cache_write_tokens=100_000,
    )
    expected = 3.0 + (0.2 * 15.0) + (2 * 0.3) + (0.1 * 3.75)
    assert cost == pytest.approx(expected)


def test_missing_cache_price_bills_full_rate():
    cost = table().cost(
        "openai", "gpt-test-mini", input_tokens=0, cached_input_tokens=1_000_000
    )
    assert cost == pytest.approx(0.4)


def test_cost_for_attributes():
    cost = table().cost_for_attributes(
        {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-test",
            "gen_ai.usage.input_tokens": 500_000,
            "gen_ai.usage.output_tokens": 125_000,
        }
    )
    assert cost == pytest.approx(1.0 + 1.0)


def test_response_model_preferred_over_request_model():
    cost = table().cost_for_attributes(
        {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-test",
            "gen_ai.response.model": "gpt-test-mini",
            "gen_ai.usage.input_tokens": 1_000_000,
        }
    )
    assert cost == pytest.approx(0.4)


def test_bundled_table_loads():
    bundled = PricingTable.load()
    assert bundled.as_of
    assert bundled.lookup("openai", "gpt-4o") is not None


def test_env_override_merges(tmp_path, monkeypatch):
    override = {
        "as_of": "2026-07-01",
        "prices": {
            "openai/gpt-4o": {"input_per_mtok": 99.0, "output_per_mtok": 99.0},
            "custom/my-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
        },
    }
    path = tmp_path / "override.json"
    path.write_text(json.dumps(override))
    monkeypatch.setenv("TRACEBURN_PRICING", str(path))

    merged = PricingTable.load()
    assert merged.lookup("openai", "gpt-4o").input_per_mtok == 99.0
    assert merged.lookup("custom", "my-model").output_per_mtok == 2.0
    assert merged.lookup("openai", "gpt-4o-mini") is not None
    assert merged.as_of == "2026-07-01"


def test_runtime_override():
    t = table()
    t.override("openai/gpt-test", ModelPrice(1.0, 1.0))
    assert t.lookup("openai", "gpt-test").input_per_mtok == 1.0


def test_per_model_provenance_survives_overlay(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"as_of": "2026-10-01", "prices": {
        "custom/model": {"input_per_mtok": 1, "output_per_mtok": 2, "source": "contract"}}}))
    t = PricingTable.load(path)
    assert t.lookup("custom", "model").as_of == "2026-10-01"
    assert t.lookup("custom", "model").source == "contract"
    assert t.lookup("anthropic", "claude-haiku-4-5-20251001").as_of == "2026-10-11"
    assert t.lookup("openai", "gpt-4o").as_of == "2026-07-04"


def test_recorder_keeps_estimation_provenance(store):
    from traceburn.recorder import Recorder
    from traceburn.schema import Span
    recorder = Recorder(store=store, pricing=PricingTable.load())
    span = Span(span_id="provenance", trace_id="t", name="call", kind="llm", start_ns=0,
                attributes={"gen_ai.system": "anthropic", "gen_ai.request.model": "claude-haiku-4-5",
                            "gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 10})
    recorder._finalize_and_write(span)
    assert span.attributes["cost_basis"] == "estimated"
    assert span.attributes["pricing_as_of"] == "2026-10-11"
    assert span.attributes["pricing_source"].startswith("https://platform.claude.com/")
    # An explicitly supplied cost and its provenance must not be overwritten.
    span.attributes.update(cost_usd=7, cost_basis="invoiced", pricing_as_of="contract date")
    span.span_id = "explicit"
    recorder._finalize_and_write(span)
    assert span.attributes["cost_usd"] == 7
    assert span.attributes["cost_basis"] == "invoiced"
    assert span.attributes["pricing_as_of"] == "contract date"


def test_version_suffix_never_matches_shorter_key():
    t = PricingTable(
        prices={"anthropic/claude-test-4": ModelPrice(15.0, 75.0)},
        as_of="2026-01-01",
    )
    assert t.lookup("anthropic", "claude-test-4-5") is None
    assert t.lookup("anthropic", "claude-test-4-20260101") is not None


def test_dashed_date_suffix_matches():
    price = table().lookup("openai", "gpt-test-2024-08-06")
    assert price.input_per_mtok == 2.0
    assert table().lookup("openai", "gpt-test-2025-04") is not None


def test_absent_usage_is_unknown_cost_not_zero():
    assert table().cost_for_attributes({
        "gen_ai.system": "openai", "gen_ai.request.model": "gpt-test",
    }) is None
    assert table().cost_for_attributes({
        "gen_ai.system": "openai", "gen_ai.request.model": "gpt-test",
        "gen_ai.usage.input_tokens": 0, "gen_ai.usage.output_tokens": 0,
    }) == 0
