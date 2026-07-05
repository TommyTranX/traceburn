import importlib.util
from pathlib import Path

HOOK = Path(__file__).parent.parent / "hooks" / "traceburn_autoinstall.py"


def _load(monkeypatch, env_value, recorder):
    if env_value is None:
        monkeypatch.delenv("TRACEBURN", raising=False)
    else:
        monkeypatch.setenv("TRACEBURN", env_value)
    import traceburn

    monkeypatch.setattr(traceburn, "install", lambda: recorder.append(True))
    spec = importlib.util.spec_from_file_location("traceburn_autoinstall_test", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_activates_when_env_set(monkeypatch):
    calls = []
    _load(monkeypatch, "1", calls)
    assert calls == [True]


def test_inert_without_env(monkeypatch):
    calls = []
    _load(monkeypatch, None, calls)
    assert calls == []


def test_inert_on_other_values(monkeypatch):
    calls = []
    _load(monkeypatch, "0", calls)
    assert calls == []
