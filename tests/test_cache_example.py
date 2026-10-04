"""The paid example must bound model calls before executing another round."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from traceburn import recorder as recorder_module


@pytest.fixture
def example(monkeypatch, recorder):
    monkeypatch.setattr(recorder_module, "_default_recorder", recorder)
    path = Path(__file__).parents[1] / "examples" / "cache_before_after.py"
    spec = importlib.util.spec_from_file_location("cache_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tool_response(name="lookup_account"):
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[SimpleNamespace(type="tool_use", id="t1", name=name,
                                 input={"email": "test@example.com"})],
    )


class FakeClient:
    def __init__(self, responses):
        self.messages = self
        self.responses = iter(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return next(self.responses)


def test_tool_loop_stops_before_another_paid_call(example):
    client = FakeClient([tool_response()] * 10)
    with pytest.raises(RuntimeError, match="exceeded 2 tool rounds"):
        example.triage("t1", "test@example.com", "Help", False, client=client)
    assert client.calls == 3


def test_normal_tool_result_is_returned(example):
    completed = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text='{"category":"question"}')],
    )
    client = FakeClient([tool_response(), completed])
    assert example.triage("t1", "test@example.com", "Help", True, client=client) == {
        "category": "question"
    }
    assert client.calls == 2


def test_unrecognized_tool_does_not_trigger_another_model_call(example):
    client = FakeClient([tool_response("unknown")])
    with pytest.raises(RuntimeError, match="unsupported tool"):
        example.triage("t1", "test@example.com", "Help", False, client=client)
    assert client.calls == 1


def test_invalid_variant_fails_before_client_creation(example):
    with pytest.raises(SystemExit) as exc:
        example.main(["cachedd"])
    assert exc.value.code == 2
