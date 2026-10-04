import socket

import pytest

from traceburn.cli import main
from traceburn.demo import record_demo
from traceburn.report import render_report, report_data, write_report
from traceburn.schema import Span, Trace
from traceburn.store import Store


@pytest.fixture
def private_store(tmp_path):
    store = Store(tmp_path / 'private.db')
    secret = 'SECRET_PAYLOAD_abc123_</script><img src=x onerror=alert(1)>'
    trace_id = secret + '-trace'
    store.insert_trace(Trace(trace_id, secret, session_id=secret, start_ns=0, end_ns=1000))
    for i in range(3):
        store.insert_span(Span(
            secret + str(i), trace_id, secret, kind='llm',
            start_ns=i * 100, end_ns=i * 100 + 50, error=secret,
            attributes={
                'gen_ai.system': secret, 'gen_ai.request.model': secret,
                'gen_ai.usage.input_tokens': 200, 'gen_ai.usage.output_tokens': 20,
                'cost_usd': 0.01, 'request_hash': secret,
                'request': {'messages': [{'role': 'user', 'content': secret}]},
                'response': {'text': secret}, 'response_raw': secret,
                'path': secret, 'custom': secret,
            },
        ))
    yield store, trace_id, secret
    store.close()


def test_export_omits_secrets_in_every_string_field(private_store):
    store, trace_id, secret = private_store
    data = report_data(store, trace_id)
    html = render_report(store, trace_id)
    assert data['findings']
    assert secret not in repr(data)
    assert 'SECRET_PAYLOAD' not in html
    assert '<script' not in html
    assert '<img' not in html
    assert 'src=' not in html
    assert 'href="http' not in html
    assert 'Repeated model requests' in html
    assert 'Call 1' in html and '#call-1' in html
    assert 'Repeated sampling can be intentional' in html
    assert '$0.0300' in html
    assert 'Synthetic demo' not in html
    # Export never changes or deletes the underlying evidence.
    assert store.get_spans(trace_id)[0].attributes['request']['messages'][0]['content'] == secret


def test_export_drops_unrecognized_kind_status_and_dynamic_findings(private_store, monkeypatch):
    store, trace_id, secret = private_store
    store.insert_span(Span(secret+'-tool', trace_id, secret, kind=secret, status=secret))
    import traceburn.report as module
    monkeypatch.setattr(module.waste, 'report', lambda *a: {'findings': [
        {'rule_id': secret},
        {'rule_id': 'loops', 'span_ids': [secret], 'confidence': secret,
         'avoidable_usd': secret, 'summary': secret, 'explanation': secret},
    ]})
    html = render_report(store, trace_id)
    assert 'SECRET_PAYLOAD' not in html
    assert 'unknown confidence' in html


@pytest.mark.parametrize('cost', [None, '0.10', -1, float('inf'), True])
def test_export_does_not_treat_missing_or_invalid_cost_as_zero(tmp_path, cost):
    store = Store(tmp_path / 'db')
    store.insert_trace(Trace('trace', 'test'))
    store.insert_span(Span('a', 'trace', 'model', kind='llm', attributes={'cost_usd': cost}))
    html = render_report(store, 'trace')
    assert 'Known-cost subtotal' in html
    assert '1 model call(s) have unknown cost' in html
    assert '<td>unknown</td>' in html
    store.close()


def test_demo_is_offline_and_does_not_touch_user_database(tmp_path, monkeypatch, capsys):
    def no_network(*a, **kw):
        raise AssertionError('unexpected network access')
    monkeypatch.setattr(socket, 'create_connection', no_network)
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setenv('TRACEBURN_DB', str(tmp_path / 'user.db'))
    output = tmp_path / 'demo.html'
    assert main(['demo', '--no-browser', '-o', str(output)]) == 0
    assert not (tmp_path / 'user.db').exists()
    html = output.read_text()
    assert 'Synthetic demo' in html
    assert 'invented tokens' in html
    assert 'Repeated model requests' in html
    assert '$0.0360' in html
    assert str(output) in capsys.readouterr().out


def test_demo_opens_the_report(tmp_path, monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, 'open', opened.append)
    output = tmp_path / 'demo.html'
    assert main(['demo', '-o', str(output)]) == 0
    assert opened == [output.as_uri()]


def test_report_refuses_overwrite_and_database_even_with_force(private_store, tmp_path):
    store, trace_id, _ = private_store
    output = tmp_path / 'report.html'
    output.write_text('original')
    with pytest.raises(FileExistsError):
        write_report(store, trace_id, output)
    assert output.read_text() == 'original'
    write_report(store, trace_id, output, force=True)
    assert output.read_text().startswith('<!doctype html>')
    with pytest.raises(ValueError, match='database'):
        write_report(store, trace_id, store.path, force=True)
    assert store.get_trace(trace_id)


def test_cli_latest_and_report(tmp_path, capsys):
    db = tmp_path / 'db'
    store = Store(db)
    trace_id = record_demo(store)
    store.close()
    for command in ('show', 'waste', 'fix'):
        assert main(['--db', str(db), command]) == 0
        capsys.readouterr()
        assert main(['--db', str(db), command, 'latest']) == 0
        capsys.readouterr()
    output = tmp_path / 'report.html'
    assert main(['--db', str(db), 'report', '-o', str(output)]) == 0
    assert 'Repeated model requests' in output.read_text()
    with pytest.raises(SystemExit, match='cannot write report'):
        main(['--db', str(db), 'report', trace_id, '-o', str(output)])


def test_empty_latest(tmp_path):
    db = tmp_path / 'db'
    Store(db).close()
    with pytest.raises(SystemExit, match='no traces recorded'):
        main(['--db', str(db), 'report'])
