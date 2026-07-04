import pytest

from traceburn.pricing import ModelPrice, PricingTable
from traceburn.recorder import Recorder
from traceburn.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "traces.db")
    yield s
    s.close()


@pytest.fixture
def pricing():
    return PricingTable(
        prices={
            "openai/gpt-test": ModelPrice(
                input_per_mtok=2.0,
                output_per_mtok=8.0,
                cached_input_per_mtok=0.5,
            ),
            "anthropic/claude-test": ModelPrice(
                input_per_mtok=3.0,
                output_per_mtok=15.0,
                cached_input_per_mtok=0.3,
                cache_write_per_mtok=3.75,
            ),
        },
        as_of="2026-01-01",
    )


@pytest.fixture
def recorder(store, pricing):
    return Recorder(store=store, pricing=pricing)
