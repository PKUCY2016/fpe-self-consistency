"""Lifecycle budgets apply to the known single worker, including bounded reaping."""
import os
import select
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from fpe_solver import campaign
from fpe_solver.storage import read_json, write_json


@pytest.fixture
def subject():
    return campaign


class FakeClock:
    def __init__(self):
        self.value = 100.

    def monotonic(self):
        return self.value


class FakeProcess:
    def __init__(self, clock, outcomes):
        self.clock = clock
        self.outcomes = list(outcomes)
        self.returncode = None
        self.timeouts = []
        self.signals = []
        self.pid = 12345

    def wait(self, timeout):
        assert timeout is not None and timeout >= 0
        self.timeouts.append(timeout)
        outcome = self.outcomes.pop(0)
        if outcome == "timeout":
            self.clock.value += timeout
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        code, elapsed = outcome
        assert elapsed <= timeout
        self.clock.value += elapsed
        self.returncode = code
        return code

    def terminate(self):
        self.signals.append("TERM")

    def kill(self):
        self.signals.append("KILL")


@pytest.mark.parametrize("code,status", [(0, "EXITED"), (3, "FAILED")])
def test_normal_exit_keeps_result_and_never_signals(subject, monkeypatch, code, status):
    clock = FakeClock()
    process = FakeProcess(clock, [(code, 2.)])
    monkeypatch.setattr(subject, "time", clock)
    assert subject.wait_worker_until(process, 7300.) == (code, status)
    assert process.timeouts == [7189.]
    assert process.signals == []


def test_termination_grace_ends_early_after_confirmed_reap(subject, monkeypatch):
    clock = FakeClock()
    process = FakeProcess(clock, ["timeout", (-15, .25)])
    monkeypatch.setattr(subject, "time", clock)
    assert subject.wait_worker_until(process, 7300.) == (-15, "TIMEOUT")
    assert process.timeouts == [7189., 10.]
    assert process.signals == ["TERM"]
    assert clock.value < 7300.


@pytest.mark.parametrize("budget,waits", [(7200., [7189., 10., 1.]),
                                         (8., [0., 7., 1.]),
                                         (.25, [0., 0., .25]),
                                         (0., [0., 0., 0.])])
def test_unreaped_worker_never_waits_beyond_total_budget(subject, monkeypatch, budget, waits):
    clock = FakeClock()
    process = FakeProcess(clock, ["timeout"]*3)
    monkeypatch.setattr(subject, "time", clock)
    assert subject.wait_worker_until(process, 100.+budget) == (None, "TIMEOUT_UNREAPED")
    assert process.timeouts == waits
    assert process.signals == ["TERM", "KILL"]
    assert clock.value <= 100.+budget


def test_kill_has_at_most_one_second_to_confirm_reap(subject, monkeypatch):
    clock = FakeClock()
    process = FakeProcess(clock, ["timeout", "timeout", (-9, .1)])
    monkeypatch.setattr(subject, "time", clock)
    assert subject.wait_worker_until(process, 7300.) == (-9, "TIMEOUT")
    assert process.timeouts == [7189., 10., 1.]
    assert process.signals == ["TERM", "KILL"]
    assert clock.value == pytest.approx(7299.1)


def test_signal_race_still_reaps_without_retargeting_pid(subject, monkeypatch):
    clock = FakeClock()
    process = FakeProcess(clock, ["timeout", (-15, .1)])
    monkeypatch.setattr(subject, "time", clock)
    def already_exited():
        raise ProcessLookupError
    monkeypatch.setattr(process, "terminate", already_exited)
    assert subject.wait_worker_until(process, 7300.) == (-15, "TIMEOUT")
    assert process.signals == []


def test_unreaped_receipt_blocks_restart_before_any_spawn(subject, monkeypatch, tmp_path):
    write_json(tmp_path/"specification.json", subject.specification())
    write_json(tmp_path/"prior.receipt.json", {"status": "TIMEOUT_UNREAPED", "pid": 12345, "wall_seconds": 7200.})
    def forbidden_spawn(*args, **kwargs):
        pytest.fail("Unresolved worker must block a new submission")
    monkeypatch.setattr(subject, "subprocess", SimpleNamespace(Popen=forbidden_spawn))
    with pytest.raises(RuntimeError, match="Unresolved campaign jobs"):
        subject._main(["--out", str(tmp_path), "--phase", "stress"])
    result = subject.summarize(tmp_path)
    assert not result["engineering_acceptance_passed"]
    assert result["results"]["prior"]["status"] == "TIMEOUT_UNREAPED"


def test_unreaped_worker_stops_current_campaign_and_preserves_receipt(subject, monkeypatch, tmp_path):
    write_json(tmp_path/"specification.json", subject.specification())
    monkeypatch.setattr(subject, "tasks", lambda _: iter([("first", {}), ("second", {})]))
    spawned = []
    def spawn(*args, **kwargs):
        spawned.append(1)
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(subject, "subprocess", SimpleNamespace(Popen=spawn, STDOUT=subprocess.STDOUT))
    monkeypatch.setattr(subject, "wait_worker_until", lambda process, deadline: (None, "TIMEOUT_UNREAPED"))
    monkeypatch.setattr(subject, "summarize", lambda root: {})
    with pytest.raises(RuntimeError, match="not confirmed reaped"):
        subject._main(["--out", str(tmp_path), "--phase", "stress"])
    assert len(spawned) == 1
    assert not (tmp_path/"second.request.json").exists()
    receipt = read_json(tmp_path/"first.receipt.json")
    assert receipt["status"] == "TIMEOUT_UNREAPED"
    assert receipt["exit_code"] is None


@pytest.mark.parametrize("already_spent,expected_budget", [(0., 7200.), (28792., 8.)])
def test_deadline_includes_spawn_and_receipt_overhead(subject, monkeypatch, tmp_path, already_spent, expected_budget):
    write_json(tmp_path/"specification.json", subject.specification())
    if already_spent:
        write_json(tmp_path/"earlier.receipt.json", {"status": "EXITED", "wall_seconds": already_spent})
    clock, deadlines = FakeClock(), []
    monkeypatch.setattr(subject, "time", clock)
    monkeypatch.setattr(subject, "tasks", lambda _: iter([("one", {})]))
    def spawn(*args, **kwargs):
        clock.value += 2.
        return SimpleNamespace(pid=12345)
    def wait(process, deadline):
        deadlines.append(deadline)
        return 0, "EXITED"
    monkeypatch.setattr(subject, "subprocess", SimpleNamespace(Popen=spawn, STDOUT=subprocess.STDOUT))
    monkeypatch.setattr(subject, "wait_worker_until", wait)
    monkeypatch.setattr(subject, "summarize", lambda root: {})
    subject._main(["--out", str(tmp_path), "--phase", "stress"])
    assert deadlines == [100.+expected_budget]


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM regression")
def test_real_single_worker_ignoring_term_is_killed_and_reaped(subject):
    code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-u", "-I", "-c", code], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        ready, _, _ = select.select([process.stdout], [], [], 5.)
        assert ready and process.stdout.readline().strip() == "ready"
        started = time.monotonic()
        code, status = subject.wait_worker_until(process, started+.5)
        assert status == "TIMEOUT"
        assert code is not None and code != 0
        assert process.poll() is not None
        assert time.monotonic()-started < 2.
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2.)
        process.stdout.close()
        process.stderr.close()
