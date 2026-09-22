# Bounded DSW execution

Target: the existing `dsw-lnt0d9qe8zf1vqk6bi`. Use the operator's already registered SSH profile; never copy private keys, suppress host-key checks, resize/restart the instance, or edit another project's environment. Deployment and jobs stay inside `/mnt/workspace/fpe-self-consistency`.

1. Recheck GPU/process inventory and `/mnt/workspace` using the existing entrypoint. NAS free space is not a quota guarantee.
2. Transfer only this package's source, lockfile and deployment scripts. `python deploy/ssh_deploy.py --ssh-config REGISTERED_CONFIG` checks the exact instance and strict host-key policy, and sends compressed text without stdin streaming. Do not deploy over an active solver run. Run `sh deploy/install_environment.sh /mnt/workspace/fpe-self-consistency cpu` (or `gpu`). This makes a separate Python 3.12 environment. CPU versions and wheel hashes come from `uv.lock` exported to `receipts/cpu-requirements.lock`; set `FPE_PACKAGE_INDEX=https://mirrors.aliyun.com/pypi/simple` if upstream downloads are slow (the hashes remain required). Optional CUDA wheels use `deploy/locks/cuda-requirements.lock` when present; a first bootstrap generates that hash lock in `receipts/`. Installation retains the CPU constraints and fails on incompatible locks. Inventory remains in `receipts/`.
3. Submit an immutable request file with `python3 deploy/bounded_run.py submit request.json` once. After SSH disconnect, use `query JOB_ID`; never pick another id just to retry an uncertain submission.
4. Read `jobs/JOB_ID/result.json`, `output.log`, and the solver's own scientific report. Supervisor success only means the command exited with code zero.

Example request (save under this project's remote directory):

```json
{
  "job_id": "fpe-doctor-001",
  "campaign": "acceptance-001",
  "kind": "smoke",
  "device": "cpu",
  "timeout_seconds": 1800,
  "command": ["/mnt/workspace/fpe-self-consistency/.venv/bin/fpe", "doctor"]
}
```

Smoke jobs are capped at 30 minutes, instance jobs at 2 hours. A campaign has 8 hours total recorded elapsed/reserved time. Unknown or running jobs retain their full reservation until a terminal receipt exists; no implicit retry. Each process has a PID/start-tick identity. Cancellation uses a pidfd, validates the identity, and reports `CANCEL_REQUESTED` until the terminal receipt appears.

GPU jobs nonblockingly lock the **existing inode** `/mnt/workspace/materials-workbench/jobs/gpu.lock`, then require an empty compute-process inventory. The lock remains held until the owned process group has been cleaned and a terminal receipt has been written, including error, timeout and cancellation paths. Missing/busy locks or pre-existing compute users yield `RESOURCE_BUSY`; CPU jobs do not touch that lock. All participating projects must honor it; this is cooperative exclusion, not a scheduler or isolation boundary.

Owned process-tree RSS and GPU use are monitored against 16 GiB each. Exceeding a limit stops only this job's process group. This is a polling monitor, **not a hard quota**: short allocation spikes between observations may be missed; CUDA inventory adds up to 10 seconds to a failed monitor probe. JAX preallocation is disabled, float64 is enabled, and CPU jobs explicitly select CPU. Limits apply to supervised execution, not download/install preparation.

The runner preserves its unreaped child leader while cleaning the owned process group, so an early parent exit cannot release the GPU lock while a TERM-resistant descendant is still running. Cleanup escalates TERM to KILL only for that anchored group. An unconfirmed cleanup yields `UNCLEAN`, never `SUCCEEDED`. This backend contains an ordinary process group; deliberately detached sessions are outside its termination scope. If a previously observed descendant with matching PID/start identity survives, the receipt lists it and marks `UNCLEAN` without signalling it. Unobserved detachment cannot be certified by this backend.

The runner is a bounded Linux process backend. Machine failure or forced supervisor death can leave `UNKNOWN`; that state needs inspection and is never interpreted as success. No remote lifecycle operation or global scheduler configuration is performed.


Validation on 2026-09-22: the dedicated Python 3.12.13 environment used JAX/JAXlib/CUDA plugin 0.11.2. GPU float64 JIT ran on `cuda:0`; all 19 runtime tests passed on Linux after the lifecycle repair (18 passed and one Linux-specific skip on macOS). See `evidence/runtime-lifecycle-review.md` and `evidence/receipts__closeout-final.json` for lifecycle regression and final idle evidence, and `evidence/` for actual receipts and `locks/cuda-requirements.lock` for hash-pinned GPU wheels. NCCL is explicitly pinned to compatible 2.31.2 because 2.32.3 was absent from the observed public mirror. Setup time/download failures are distinct from solver computation.

For the authorized image stress test, `fetch_cifar_parallel.py --output PATH` fetches exact 1 MiB HTTP ranges with up to eight concurrent workers from the original CIFAR URL and requires the original binary archive checksum before publishing PATH. Run it as a supervised CPU job. It reuses verified-length prefixes from prior ranges and retains partial ranges for inspection/retry and never extracts untrusted archive paths. A downloaded dataset or a successful GPU probe is not evidence of image-generation quality.
