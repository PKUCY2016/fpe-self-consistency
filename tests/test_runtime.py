"""Resource/lifecycle invariants independent of numerical solver code."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('fpe_bounded_run', Path(__file__).parents[1] / 'deploy/bounded_run.py')
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def request(job='probe-001', **kwargs):
    return dict(job_id=job, campaign='test-campaign', kind='smoke', device='cpu', timeout_seconds=1,
                command=[sys.executable, '-c', 'print("ok")'], **kwargs)


def reserved(tmp_path, req):
    d = tmp_path / 'jobs' / req['job_id']
    d.mkdir(parents=True)
    runtime.atomic_json(d / 'request.json', req)
    return d


def test_requests_reject_unbounded_limits_and_paths():
    for change in ({'timeout_seconds': 1801}, {'timeout_seconds': float('nan')},
                   {'kind': 'arbitrary'}, {'device': 'any'}, {'job_id': '../escape'},
                   {'command': ['python', '-c', 'pass']}):
        req = request()
        req.update(change)
        with pytest.raises(ValueError):
            runtime.validate_request(req)


def test_unknown_is_not_success_or_permission_to_resubmit(tmp_path):
    reserved(tmp_path, request())
    assert runtime.query(tmp_path, 'probe-001')['status'] == 'UNKNOWN'
    assert runtime.submit(tmp_path, request())['submission'] == 'EXISTING_NOT_RESUBMITTED'
    changed = request()
    changed['timeout_seconds'] = 2
    with pytest.raises(ValueError, match='collision'):
        runtime.submit(tmp_path, changed)


def test_campaign_keeps_full_reservation_until_terminal(tmp_path):
    d = reserved(tmp_path, request())
    assert runtime.campaign_reserved(tmp_path, 'test-campaign') == 1
    runtime.atomic_json(d / 'result.json', {'elapsed_seconds': 0.25})
    assert runtime.campaign_reserved(tmp_path, 'test-campaign') == 0.25


def test_symlink_control_files_fail_closed(tmp_path):
    d = reserved(tmp_path, request())
    (d / 'process.json').symlink_to(tmp_path / 'unrelated')
    with pytest.raises(ValueError, match='symlink'):
        runtime.query(tmp_path, 'probe-001')


def test_identity_missing_or_mismatched_is_not_alive(monkeypatch):
    monkeypatch.setattr(runtime, 'identity', lambda pid: '12')
    assert not runtime.alive({'pid': 1, 'start_ticks': None})
    assert not runtime.alive({'pid': 1, 'start_ticks': '13'})
    assert runtime.alive({'pid': 1, 'start_ticks': '12'})


def test_verified_pidfd_cancel(monkeypatch, tmp_path):
    ref = {'pid': 123, 'start_ticks': '12'}
    monkeypatch.setattr(runtime, 'query', lambda *a: {'status': 'RUNNING', 'process': ref})
    monkeypatch.setattr(runtime, 'alive', lambda r: True)
    sent = []
    monkeypatch.setattr(runtime.os, 'pidfd_open', lambda pid: 99, raising=False)
    monkeypatch.setattr(runtime.signal, 'pidfd_send_signal', lambda fd, sig: sent.append(fd), raising=False)
    monkeypatch.setattr(runtime.os, 'close', lambda fd: None)
    assert runtime.cancel(tmp_path, 'probe-001')['status'] == 'CANCEL_REQUESTED'
    assert sent == [99]


def test_supervisor_success_preserves_limits_and_environment(tmp_path):
    req = request()
    req['command'] = [sys.executable, '-c', ('import os; assert os.environ["JAX_PLATFORMS"]=="cpu"; '
                      'assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]=="false"')]
    d = reserved(tmp_path, req)
    runtime.supervise(tmp_path, req['job_id'])
    result = json.loads((d / 'result.json').read_text())
    assert result['status'] == 'SUCCEEDED'
    assert result['request_sha256'] == runtime.digest(req)
    assert result['monitor_limit_bytes'] == 16 * 1024**3
    assert not result['monitor_is_hard_quota']


@pytest.mark.skipif(os.name != 'posix', reason='owned POSIX process groups')
def test_supervisor_timeout_retains_terminal_receipt(tmp_path):
    req = request()
    req.update(timeout_seconds=0.05, command=[sys.executable, '-c', 'import time; time.sleep(10)'])
    d = reserved(tmp_path, req)
    runtime.supervise(tmp_path, req['job_id'])
    result = json.loads((d / 'result.json').read_text())
    assert result['status'] == 'TIMEOUT'
    assert result['elapsed_seconds'] < 5


def test_busy_gpu_never_starts_child(monkeypatch, tmp_path):
    req = request()
    req['device'] = 'gpu'
    d = reserved(tmp_path, req)
    monkeypatch.setattr(runtime, 'GPU_LOCK', tmp_path / 'missing-owner-lock')
    runtime.supervise(tmp_path, req['job_id'])
    assert json.loads((d / 'result.json').read_text())['status'] == 'RESOURCE_BUSY'
    assert not (d / 'child.json').exists()


def test_rss_limit_stops_only_owned_child(monkeypatch, tmp_path):
    req = request()
    req['command'] = [sys.executable, '-c', 'import time; time.sleep(10)']
    d = reserved(tmp_path, req)
    monkeypatch.setattr(runtime, 'rss_bytes', lambda pids: 17 * runtime.GIB)
    runtime.supervise(tmp_path, req['job_id'])
    assert json.loads((d / 'result.json').read_text())['status'] == 'MEMORY_LIMIT'


def test_submission_snapshots_source_and_is_launched_once(monkeypatch, tmp_path):
    (tmp_path / "src/fpe_solver").mkdir(parents=True)
    (tmp_path / "src/fpe_solver/core.py").write_text("VALUE = 1\n")
    launches = []

    class Child:
        pid = 12345

    def launch(*args, **kwargs):
        launches.append(args)
        return Child()

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    assert runtime.submit(tmp_path, request())["status"] == "SUBMITTED"
    assert runtime.submit(tmp_path, request())["submission"] == "EXISTING_NOT_RESUBMITTED"
    assert len(launches) == 1
    provenance = json.loads((tmp_path / "jobs/probe-001/provenance.json").read_text())
    assert "src/fpe_solver/core.py" in provenance["source_sha256"]


def test_campaign_budget_rejects_before_reservation(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "campaign_reserved", lambda *args: runtime.CAMPAIGN_SECONDS)
    with pytest.raises(ValueError, match="campaign budget"):
        runtime.submit(tmp_path, request())
    assert not (tmp_path / "jobs/probe-001").exists()


def test_cancel_rechecks_identity_after_pidfd_open(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "query", lambda *args: {
        "status": "RUNNING", "process": {"pid": 1, "start_ticks": "12"}})
    monkeypatch.setattr(runtime.os, "pidfd_open", lambda pid: 99, raising=False)
    monkeypatch.setattr(runtime.os, "close", lambda fd: None)
    monkeypatch.setattr(runtime, "alive", lambda ref: False)
    assert runtime.cancel(tmp_path, "probe-001")["status"] == "UNKNOWN"


def _grandchild_request(tmp_path, *, parent_exits=False):
    """A finite owned process tree; the grandchild deliberately ignores SIGTERM."""
    import textwrap

    marker = tmp_path / "owned-grandchild.json"
    grandchild = textwrap.dedent(f"""
        import json, os, signal, subprocess, time
        from pathlib import Path
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signature = subprocess.check_output(
            ['ps', '-p', str(os.getpid()), '-o', 'lstart=', '-o', 'command='], text=True).strip()
        Path({str(marker)!r}).write_text(json.dumps({{'pid': os.getpid(), 'signature': signature}}))
        time.sleep(30)
    """)
    leader = textwrap.dedent(f"""
        import subprocess, sys, time
        from pathlib import Path
        subprocess.Popen([sys.executable, '-c', {grandchild!r}])
        marker = Path({str(marker)!r})
        while not marker.exists():
            time.sleep(0.01)
        {'pass' if parent_exits else 'time.sleep(30)'}
    """)
    req = request()
    req.update(timeout_seconds=1.0, command=[sys.executable, '-c', leader])
    return req, marker


def _owned_grandchild_alive(marker):
    """Only identify the PID and full start/command identity written by this test."""
    import subprocess

    if not marker.exists():
        return False
    ref = json.loads(marker.read_text())
    signature = subprocess.run(
        ['ps', '-p', str(ref['pid']), '-o', 'lstart=', '-o', 'command='],
        capture_output=True, text=True, check=False).stdout.strip()
    if signature != ref['signature']:
        return False
    state = subprocess.run(['ps', '-p', str(ref['pid']), '-o', 'stat='],
                           capture_output=True, text=True, check=False).stdout.strip()
    return bool(state) and not state.startswith('Z')


def _cleanup_owned_grandchild(marker):
    import signal

    if _owned_grandchild_alive(marker):
        # Never signal the process group or a PID whose recorded identity changed.
        os.kill(json.loads(marker.read_text())['pid'], signal.SIGKILL)


@pytest.mark.skipif(os.name != 'posix', reason='owned POSIX process groups')
def test_parent_exit_does_not_leave_sigterm_ignoring_descendant(tmp_path):
    req, marker = _grandchild_request(tmp_path, parent_exits=True)
    d = reserved(tmp_path, req)
    try:
        runtime.supervise(tmp_path, req['job_id'])
        assert marker.exists(), 'fixture grandchild never started'
        assert json.loads((d / 'result.json').read_text())['status'] in runtime.TERMINAL
        assert not _owned_grandchild_alive(marker), 'terminal receipt left a live owned descendant'
    finally:
        _cleanup_owned_grandchild(marker)


@pytest.mark.skipif(os.name != 'posix', reason='owned POSIX process groups')
@pytest.mark.parametrize('cancellation', [False, True], ids=['timeout', 'cancel'])
def test_timeout_or_cancel_reaps_sigterm_ignoring_descendant(tmp_path, monkeypatch, cancellation):
    import signal

    req, marker = _grandchild_request(tmp_path)
    d = reserved(tmp_path, req)
    if cancellation:
        # Deliver cancellation only once the real descendant has installed its handler.
        real_rss = runtime.rss_bytes
        sent = False

        def cancel_once_ready(pids):
            nonlocal sent
            if marker.exists() and not sent:
                sent = True
                os.kill(os.getpid(), signal.SIGTERM)
            return real_rss(pids)

        monkeypatch.setattr(runtime, 'rss_bytes', cancel_once_ready)
    try:
        runtime.supervise(tmp_path, req['job_id'])
        assert marker.exists(), 'fixture grandchild never started'
        expected = 'CANCELLED' if cancellation else 'TIMEOUT'
        assert json.loads((d / 'result.json').read_text())['status'] == expected
        assert not _owned_grandchild_alive(marker), 'terminal receipt left a live owned descendant'
    finally:
        _cleanup_owned_grandchild(marker)


def test_gpu_lock_retained_through_monitor_failure_cleanup_and_receipt(tmp_path, monkeypatch):
    from contextlib import contextmanager

    req = request()
    req.update(device='gpu', command=[sys.executable, '-c', 'import time; time.sleep(30)'])
    d = reserved(tmp_path, req)
    events = []
    held = False
    calls = 0
    real_stop = runtime.stop_child

    @contextmanager
    def observed_lock(*args, **kwargs):
        nonlocal held
        held = True
        events.append('lock_acquired')
        try:
            yield 123
        finally:
            assert (d / 'result.json').exists(), 'GPU lock released before terminal receipt'
            receipt = json.loads((d / 'result.json').read_text())
            assert receipt['owned_process_group_cleanup_confirmed'] is True
            events.append('lock_released_after_receipt')
            held = False

    def failing_monitor():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {}
        raise RuntimeError('injected GPU inventory failure')

    def observed_cleanup(child):
        assert held, 'GPU lock was released before cleanup'
        events.append('cleanup_started')
        clean = real_stop(child)
        assert clean
        events.append('cleanup_finished')
        return clean

    monkeypatch.setattr(runtime, 'locked', observed_lock)
    monkeypatch.setattr(runtime, 'gpu_usage', failing_monitor)
    monkeypatch.setattr(runtime, 'stop_child', observed_cleanup)
    runtime.supervise(tmp_path, req['job_id'])
    receipt = json.loads((d / 'result.json').read_text())
    assert receipt['status'] == 'FAILED'
    assert 'injected GPU inventory failure' in receipt['reason']
    assert receipt['owned_process_group_cleanup_confirmed'] is True
    assert not held
    assert events == ['lock_acquired', 'cleanup_started', 'cleanup_finished', 'lock_released_after_receipt']


def test_unconfirmed_cleanup_cannot_be_reported_as_success(tmp_path, monkeypatch):
    req = request()
    d = reserved(tmp_path, req)
    real_stop = runtime.stop_child

    def cleanup_without_confirmation(child):
        # Reap the actual test process, then inject uncertainty only into its evidence.
        assert real_stop(child)
        child._fpe_cleanup_error = 'injected observation failure'
        return False

    monkeypatch.setattr(runtime, 'stop_child', cleanup_without_confirmation)
    runtime.supervise(tmp_path, req['job_id'])
    receipt = json.loads((d / 'result.json').read_text())
    assert receipt['returncode'] == 0
    assert receipt['status'] == 'UNCLEAN'
    assert receipt['owned_process_group_cleanup_confirmed'] is False
    assert receipt['cleanup_error'] == 'injected observation failure'


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux /proc start identity and pidfd')
def test_observed_detached_descendant_forces_unclean_without_killing_it(tmp_path):
    import signal
    import textwrap

    marker = tmp_path / 'detached-grandchild.json'
    grandchild = textwrap.dedent(f"""
        import json, os, signal, time
        from pathlib import Path
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pid = os.getpid()
        fields = Path(f'/proc/{{pid}}/stat').read_text().rsplit(') ', 1)[1].split()
        Path({str(marker)!r}).write_text(json.dumps({{'pid': pid, 'start_ticks': fields[19]}}))
        time.sleep(30)
    """)
    leader = textwrap.dedent(f"""
        import subprocess, sys, time
        from pathlib import Path
        subprocess.Popen([sys.executable, '-c', {grandchild!r}], start_new_session=True)
        while not Path({str(marker)!r}).exists():
            time.sleep(0.01)
        time.sleep(0.6)
    """)
    req = request()
    req.update(timeout_seconds=2, command=[sys.executable, '-c', leader])
    d = reserved(tmp_path, req)
    try:
        runtime.supervise(tmp_path, req['job_id'])
        assert marker.exists(), 'fixture grandchild never started'
        ref = json.loads(marker.read_text())
        receipt = json.loads((d / 'result.json').read_text())
        assert receipt['status'] == 'UNCLEAN'
        assert ref in receipt['observed_surviving_descendants']
        assert runtime.alive(ref), 'group backend must not signal a detached process'
    finally:
        if marker.exists():
            ref = json.loads(marker.read_text())
            if runtime.alive(ref):
                try:
                    fd = os.pidfd_open(ref['pid'])
                except ProcessLookupError:
                    pass
                else:
                    try:
                        if runtime.alive(ref):
                            signal.pidfd_send_signal(fd, signal.SIGKILL)
                    finally:
                        os.close(fd)
