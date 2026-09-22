"""Bounded Linux job supervisor; receipts are evidence, not a durable scheduler.

Run with the standard library Python. The child uses the project's own environment.
One immutable job id identifies one submission, including uncertain SSH responses.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/workspace/fpe-self-consistency')
GPU_LOCK = Path('/mnt/workspace/materials-workbench/jobs/gpu.lock')
GIB = 1024**3
MAX_SECONDS = {'smoke': 1800, 'instance': 7200}
CAMPAIGN_SECONDS = 28800
TERMINAL = {'SUCCEEDED', 'FAILED', 'TIMEOUT', 'MEMORY_LIMIT', 'CANCELLED', 'RESOURCE_BUSY', 'UNCLEAN'}


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x') as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def digest(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_id(value: str) -> str:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', value):
        raise ValueError('id must contain only letters, digits, underscores and hyphens')
    return value


def identity(pid: int) -> str | None:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def alive(ref: dict | None) -> bool:
    return bool(ref and ref.get('start_ticks') is not None and identity(ref['pid']) == ref['start_ticks'])


def safe_root(root: Path) -> Path:
    root = Path(root).absolute()
    if any(p.is_symlink() for p in [root, *root.parents]):
        raise ValueError('symlink in runtime root')
    return root


def job_directory(root: Path, job_id: str) -> Path:
    root = safe_root(root)
    d = root / 'jobs' / safe_id(job_id)
    if d.parent.is_symlink() or d.is_symlink():
        raise ValueError('symlink job directory')
    if d.exists() and any(p.is_symlink() for p in d.iterdir()):
        raise ValueError('symlink in job control files')
    return d


@contextlib.contextmanager
def locked(path: Path, *, create: bool = True, nonblock: bool = False):
    flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
    fd = os.open(path, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblock else 0))
        yield fd
    finally:
        os.close(fd)


def validate_request(request: dict) -> None:
    safe_id(request['job_id'])
    safe_id(request['campaign'])
    limit = MAX_SECONDS.get(request['kind'])
    timeout = request['timeout_seconds']
    if limit is None or not isinstance(timeout, (float, int)) or not 0 < timeout <= limit:
        raise ValueError('timeout exceeds the approved smoke/instance bound')
    if not math.isfinite(timeout):
        raise ValueError('timeout must be finite')
    if request['device'] not in ('cpu', 'gpu'):
        raise ValueError('device must be cpu or gpu')
    if not request['command'] or not all(isinstance(x, str) and '\x00' not in x for x in request['command']):
        raise ValueError('command must be a nonempty argv list')
    if not Path(request['command'][0]).is_absolute():
        raise ValueError('use an absolute executable path')


def campaign_reserved(root: Path, campaign: str) -> float:
    total = 0.0
    for d in (root / 'jobs').iterdir():
        if not d.is_dir() or d.is_symlink():
            continue
        req_file = d / 'request.json'
        if not req_file.exists():
            continue
        req = json.loads(req_file.read_text())
        if req['campaign'] != campaign:
            continue
        result = d / 'result.json'
        total += json.loads(result.read_text())['elapsed_seconds'] if result.exists() else req['timeout_seconds']
    return total


def query(root: Path, job_id: str, expected_digest: str | None = None) -> dict:
    d = job_directory(root, job_id)
    request_file = d / 'request.json'
    if not request_file.exists():
        return {'status': 'UNKNOWN', 'job_id': job_id, 'reason': 'no complete reservation; never retry implicitly'}
    req = json.loads(request_file.read_text())
    if expected_digest and digest(req) != expected_digest:
        raise ValueError('job request digest mismatch')
    if (d / 'result.json').exists():
        result = json.loads((d / 'result.json').read_text())
        if result['request_sha256'] != digest(req):
            raise ValueError('terminal receipt request mismatch')
        return result
    process = json.loads((d / 'process.json').read_text()) if (d / 'process.json').exists() else None
    return {'status': 'RUNNING' if alive(process) else 'UNKNOWN', 'job_id': job_id,
            'request_sha256': digest(req), 'process': process,
            'reason': 'matching live supervisor' if alive(process) else 'no terminal evidence or matching process; do not resubmit'}


def submit(root: Path, request: dict) -> dict:
    validate_request(request)
    root = safe_root(root)
    d = job_directory(root, request['job_id'])
    d.parent.mkdir(parents=True, exist_ok=True)
    with locked(root / 'campaign.lock'):
        if d.exists():
            existing = d / 'request.json'
            if not existing.exists() or digest(json.loads(existing.read_text())) != digest(request):
                raise ValueError('job id collision or incomplete reservation; no process launched')
            return {**query(root, request['job_id']), 'submission': 'EXISTING_NOT_RESUBMITTED'}
        if campaign_reserved(root, request['campaign']) + request['timeout_seconds'] > CAMPAIGN_SECONDS:
            raise ValueError('8 hour campaign budget exhausted or reserved by existing jobs')
        d.mkdir()
        atomic_json(d / 'request.json', request)
        source_files = [*root.glob('src/fpe_solver/*.py'), *root.glob('deploy/*.py'),
                        root / 'pyproject.toml', root / 'uv.lock']
        atomic_json(d / 'provenance.json', {
            'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source_files if p.is_file()},
            'environment_receipt': json.loads((root / 'receipts/runtime.json').read_text())
                                   if (root / 'receipts/runtime.json').exists() else None})
        with (d / 'supervisor.log').open('x') as log:
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                                  'supervise', request['job_id']], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
        atomic_json(d / 'launch.json', {'supervisor_pid': p.pid, 'submitted_unix': time.time(),
                                      'request_sha256': digest(request)})
    return {'status': 'SUBMITTED', 'job_id': request['job_id'], 'request_sha256': digest(request)}


def descendants(pid: int) -> set[int]:
    result = {pid}
    changed = True
    parents = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(') ', 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            pass
    while changed:
        old = len(result)
        result.update(p for p, parent in parents.items() if parent in result)
        changed = len(result) != old
    return result


def rss_bytes(pids: set[int]) -> int:
    total = 0
    for pid in pids:
        try:
            resident = int(Path(f'/proc/{pid}/statm').read_text().split()[1])
            total += resident * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            pass
    return total


def gpu_usage() -> dict[int, int]:
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=10, check=True)
    result = {}
    for row in p.stdout.splitlines():
        pid, memory = row.split(',')
        result[int(pid)] = result.get(int(pid), 0) + int(memory.strip()) * 1024**2
    return result


def exited_without_reaping(child: subprocess.Popen) -> bool:
    """Keep the original PID reserved until its owned process group is clean."""
    if child.returncode is not None:
        raise RuntimeError('child was reaped before process-group cleanup')
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    if hasattr(os, 'waitid'):
        return os.waitid(os.P_PID, child.pid, flags) is not None
    if sys.platform == 'darwin':
        # Some standalone CPython builds omit os.waitid despite macOS supporting it.
        # Darwin siginfo_t starts with si_signo (int); use aligned, oversized storage.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        info = (ctypes.c_long * 32)()
        libc.waitid.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
        libc.waitid.restype = ctypes.c_int
        if libc.waitid(os.P_PID, child.pid, ctypes.byref(info), flags) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return ctypes.cast(info, ctypes.POINTER(ctypes.c_int))[0] != 0
    raise RuntimeError('non-reaping waitid is required for safe group ownership')


def process_group_members(group: int) -> set[int]:
    """Live members of a group whose unreaped leader was started by this supervisor."""
    result = set()
    if sys.platform.startswith('linux'):
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = path.read_text().rsplit(') ', 1)[1].split()
                if int(fields[2]) == group and fields[0] != 'Z':
                    result.add(int(path.parent.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
    else:
        observed = subprocess.run(['ps', '-A', '-o', 'pid=', '-o', 'pgid=', '-o', 'stat='],
                                  capture_output=True, text=True, timeout=5, check=True)
        for line in observed.stdout.splitlines():
            pid, pgid, state = line.split()
            if int(pgid) == group and not state.startswith('Z'):
                result.add(int(pid))
    return result


def group_is_anchored(child: subprocess.Popen) -> bool:
    if child.returncode is not None:
        return False
    # Darwin may hide a waitable zombie from getpgid; WNOWAIT still reserves its PID.
    if exited_without_reaping(child):
        return True
    try:
        return os.getpgid(child.pid) == child.pid
    except ProcessLookupError:
        return exited_without_reaping(child)


def stop_child(child: subprocess.Popen) -> bool:
    """Clean the owned group before reaping its leader or releasing the GPU lock.

    WNOWAIT keeps the leader PID reserved, including after an early normal exit.
    We never signal a process group after releasing that ownership anchor.
    Detached sessions are outside this bounded group backend's containment scope.
    """
    if getattr(child, '_fpe_cleanup_confirmed', False):
        return True
    if child.returncode is not None:
        return False
    try:
        if not group_is_anchored(child):
            return False
        if process_group_members(child.pid):
            os.killpg(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while process_group_members(child.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if process_group_members(child.pid):
            os.killpg(child.pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while process_group_members(child.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        clean = not process_group_members(child.pid)
        child.wait(timeout=5)
        child._fpe_cleanup_confirmed = clean
        return clean
    except (OSError, ValueError, IndexError, RuntimeError, AttributeError, subprocess.SubprocessError) as exc:
        child._fpe_cleanup_error = f'{type(exc).__name__}: {exc}'
        # The unreaped leader still anchors the group; stop only that owned group.
        # Failed observation is never converted into a successful cleanup claim.
        try:
            if group_is_anchored(child):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        except (OSError, RuntimeError, AttributeError, subprocess.SubprocessError):
            pass
        return False


def supervise(root: Path, job_id: str) -> None:
    d = job_directory(root, job_id)
    request = json.loads((d / 'request.json').read_text())
    validate_request(request)
    started = time.monotonic()
    supervisor_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    child = None
    cancel = False
    status, reason, code = 'FAILED', 'supervisor not completed', None
    peak_rss = peak_gpu = 0
    cleanup_confirmed = None
    observed_descendants = {}
    observed_survivors = []
    stack = contextlib.ExitStack()
    atomic_json(d / 'process.json', {'pid': os.getpid(), 'start_ticks': identity(os.getpid())})

    def cancelled(_sig, _frame):
        nonlocal cancel
        cancel = True

    old_term, old_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        if request['device'] == 'gpu':
            try:
                stack.enter_context(locked(GPU_LOCK, create=False, nonblock=True))
            except (BlockingIOError, FileNotFoundError):
                status, reason = 'RESOURCE_BUSY', 'shared GPU lock unavailable; use CPU or wait'
                return
            if gpu_usage():
                status, reason = 'RESOURCE_BUSY', 'GPU has an existing compute process'
                return
        env = os.environ.copy()
        env.update({'JAX_ENABLE_X64': 'true', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                    'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform',
                    'JAX_PLATFORMS': 'cpu' if request['device'] == 'cpu' else 'cuda',
                    'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        with (d / 'output.log').open('x') as log:
            child = subprocess.Popen(request['command'], cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
            atomic_json(d / 'child.json', {'pid': child.pid, 'start_ticks': identity(child.pid)})
            while not exited_without_reaping(child):
                pids = descendants(child.pid) | process_group_members(child.pid)
                for pid in pids - {child.pid}:
                    start_ticks = identity(pid)
                    if start_ticks is not None:
                        observed_descendants[pid] = {'pid': pid, 'start_ticks': start_ticks}
                rss = rss_bytes(pids)
                gpu = sum(v for p, v in gpu_usage().items() if p in pids) if request['device'] == 'gpu' else 0
                peak_rss, peak_gpu = max(peak_rss, rss), max(peak_gpu, gpu)
                if cancel:
                    status, reason = 'CANCELLED', 'explicit cancellation'
                    break
                if rss > 16 * GIB or gpu > 16 * GIB:
                    status, reason = 'MEMORY_LIMIT', 'owned process tree exceeded 16 GiB RSS or GPU monitor limit'
                    break
                if time.monotonic() - started >= request['timeout_seconds']:
                    status, reason = 'TIMEOUT', 'declared wall time exhausted'
                    break
                time.sleep(0.2)
            else:
                status, reason = 'CHILD_EXITED', 'child exited; scientific outcome is in its own report'
    except Exception as exc:  # noqa: BLE001 - all supervisor errors require a terminal receipt
        status, reason = 'FAILED', f'{type(exc).__name__}: {exc}'
    finally:
        # Lock ownership extends through normal, exception, timeout and cancellation cleanup.
        try:
            if child:
                cleanup_confirmed = stop_child(child)
                # Known descendants that detached are outside group termination, but
                # must not be silently reported as cleaned. Do not signal them.
                observed_survivors = [ref for ref in observed_descendants.values() if alive(ref)]
                code = child.returncode
                if observed_survivors:
                    status, reason = 'UNCLEAN', 'previously observed owned descendants remain alive outside the cleaned group'
                elif not cleanup_confirmed:
                    status, reason = 'UNCLEAN', 'owned process-group cleanup could not be confirmed; inspect before further work'
                elif status == 'CHILD_EXITED':
                    status = 'SUCCEEDED' if code == 0 else 'FAILED'
            atomic_json(d / 'result.json', {'status': status, 'reason': reason, 'job_id': job_id,
                                          'request_sha256': digest(request), 'returncode': code,
                                          'elapsed_seconds': time.monotonic() - started,
                                          'peak_tree_rss_bytes': peak_rss, 'peak_tree_gpu_bytes': peak_gpu,
                                          'monitor_limit_bytes': 16 * GIB, 'monitor_interval_seconds': 0.2,
                                          'monitor_is_hard_quota': False,
                                          'owned_process_group_cleanup_confirmed': cleanup_confirmed,
                                          'observed_surviving_descendants': observed_survivors,
                                          'cleanup_error': getattr(child, '_fpe_cleanup_error', None),
                                          'finished_unix': time.time(), 'supervisor_sha256': supervisor_hash})
        finally:
            stack.close()
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)


def cancel(root: Path, job_id: str) -> dict:
    result = query(root, job_id)
    if result['status'] != 'RUNNING':
        return result
    ref = result['process']
    fd = os.pidfd_open(ref['pid'])
    try:
        if not alive(ref):
            return {'status': 'UNKNOWN', 'reason': 'process identity changed; not signalled'}
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    return {'status': 'CANCEL_REQUESTED', 'job_id': job_id}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    sub = p.add_subparsers(dest='operation', required=True)
    s = sub.add_parser('submit')
    s.add_argument('request', type=Path)
    for name in ('query', 'cancel', 'supervise'):
        sub.add_parser(name).add_argument('job_id')
    args = p.parse_args()
    if args.operation == 'submit':
        result = submit(args.root, json.loads(args.request.read_text()))
    elif args.operation == 'supervise':
        supervise(args.root, args.job_id)
        return
    else:
        result = globals()[args.operation](args.root, args.job_id)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
