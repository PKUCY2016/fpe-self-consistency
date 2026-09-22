# FPE implementation ledger

Actual workspace: `/Users/yuchen/3. Project/Low priority/随机分析证明/fpe_solver`.
The pre-existing proof project is unchanged. This package supplies engineering evidence, not a completed proof.

## Frozen scope and order

1. Package/environment: Python 3.12, uv.lock, float64 JAX; isolated DSW deployment and receipts.
2. Numerical core: periodic Fourier/Bernstein velocity, exact prolongation, joint RK4 state and score/Jacobian checks.
3. Fixed solver and independent NumPy/SciPy evaluation, analytic heat / conservative positive finite volume reference.
4. Finite-action evolution, candidate isolation, fresh paired validation, history and resume.
5. All seeds 0,1,2; fixed-capacity control; heldout audit; cost comparison; installed-wheel validation.
6. User-added image experiment: separate 32x32 RGB Euclidean OU feasibility branch, no low-dimensional certification claim.

## Early development evidence (historical snapshot)

- Core: 9 tests passed; evaluation: 12; controller/storage: 7; regression review: 10; DSW runtime: 10.
- Core and evaluation were implemented/reviewed by separate agents.
- Reviewed defects fixed: new-direction-only expansion probes, absolute per-sample integration differences, unqualified initial isolation, nonfinite JSON, empty-checkpoint recovery. Sampling at intermediate time was subsequently corrected and covered by regression tests.
- `runs/evolution-seed0` preserves a JSON numpy-bool failure, not an acceptance result.
- `runs/evolution-seed0-review1`: K=2 -> K=3, then three rejected candidates. Residual 0.14194 -> 0.015279; PLATEAU. Independent 21-time max KL 0.0009499, TV 0.018494, mass error 2.21e-14. Single development seed only.
- At this early snapshot, release tests, multiseed campaign, final heldout audit and image experiment had not run. See later records below.

## Resource and recovery rules

Local run state/checkpoints are atomic and hash verified, with one writer. Resume of terminal runs is a no-op. Source/dependency drift on resume is recorded and rejected. Total round cap includes initial training; each candidate has at most 500 updates.

DSW uses `/mnt/workspace/fpe-self-consistency`, the existing verified Materials Workbench SSH profile and GPU lock; dedicated Python environment. Runtime monitor: 16 GiB RSS/GPU, 30 min debug, 2 h instance, 8 h campaign; process identity and terminal receipts. No duplicate uncertain submissions. See deploy/README.md.

All thresholds remain as requested; failed seeds/candidates and resource failures stay in reports. No offline reference values are supplied to the controller. Image samples, training-neighbor matches and heldout-neighbor matches must be reported separately.

## Reviewed version 2

- Full tests: 97 passed, one upstream Optax deprecation warning (`reports/full-tests-v2.txt`). All source/tests/deploy lint checks pass.
- Added rejected-candidate optimization feedback after an actual internal-residual regression. Only qualified, significantly worsened paired residuals prioritize the existing learning-rate-halving action. Numerical/sampling priorities and single-setting edits remain intact. 18 independent feedback tests passed.
- `acceptance-v1` is preserved development evidence and marked SUPERSEDED. Its active parent/worker were identity-checked and interrupted; all checkpoints remain, no resubmission. No heldout audit was run or inspected.
- `acceptance-v2` reruns all adaptive, controlled, heldout and stress cases with frozen new source. Fixed baseline reuse is independently justified in `reports/baseline-reuse.json`: old training source reconstructed to its recorded hash; optimizer/JIT step/residual AST unchanged; representative coefficients, PRNG and optimizer states bitwise equal.
- Installed-wheel v2: PASS (archived as `reports/delivery-v2.json`), hash `2b17a615781ee78fe9928f4d7a9516019590de34491603440dd29bb85a9ba4cb`.
- Actual direction-gradient error 5.24e-13 and score consistency 1.32e-8 recorded in `reports/mathematical-checks.json`; high-order evidence is only a two-point endpoint diagnostic.
- DSW isolated GPU environment and lock-protected float64 JIT check passed. Dataset download and image run use bounded, identity-tracked jobs. The image results were subsequently completed and independently reviewed; see below.

## Completed scientific evidence (2026-09-22)

- Frozen v2 adaptive development benchmarks: 15/15 passed. Heldout audit: 5/6 passed; `audit-1-s1` stopped at PLATEAU with max KL 0.00314443 and TV 0.03076995. Complete engineering acceptance is therefore **not passed**. No algorithm, threshold or seed selection was changed after inspecting the heldout audit.
- Controlled expansion: all three seeds improved, final KL median decreased 96.242% against fixed K=2. Both arms had the same 120-second and 12-block budget caps; actual times differ and are reported, with compilation/search/validation included.
- Real CIFAR image trials: rank 32 and 128, one seed, 500 updates each. Both are visual generation failures. Capacity increase reduced per-coordinate residual 33.507%, with training times 106.47/239.44 seconds; this is a cold-start capacity comparison, not warm evolution. Full 128 generated outputs and raw arrays were independently inspected. See `reports/image-evidence-review.md`.
- DSW runtime process cleanup was independently reviewed and corrected without changing solver mathematics. Linux 19/19 lifecycle tests passed; final 12 cloud jobs all have terminal receipts, no project processes or GPU compute users remain. No additional image training was launched.

## Final implementation checks

- Final CPU suite: **116 passed, 1 skipped**, 39.09 seconds (`reports/full-tests-final.txt`). The skipped test is Linux-specific and passed in the DSW runtime suite; the remaining warning is upstream Optax's `global_norm` deprecation. Ruff passed for source, tests and deployment code.
- All 45 declared acceptance tasks produced scientific results with terminal process receipts. Cumulative receipt time was 3656.912 seconds, including reused fixed baselines; longest task was 594.008 seconds. The three stress outcomes were explicit numerical failure, plateau and the 12-round budget limit, all retained.
- After the frozen campaign ended, only the campaign worker timeout path changed: a single 7200-second monotonic deadline includes ten seconds of TERM grace and one second for KILL/reaping. An unconfirmed reap stops the campaign and blocks resubmission. This local helper supervises the known single worker, not arbitrary descendants; the DSW process-group backend is separate.
- Fourteen deadline regressions passed, including a real TERM-resistant worker. Controller, model mathematics, frozen specification, learned parameters and all scientific results remain unchanged. Final wheel revalidation is recorded in `reports/delivery.json`.
- The acceptance chart was visually inspected and report links checked. Independent heldout review confirms KL failures at 12/21 points and TV failures at 11/21 points in `audit-1-s1`; no post-audit algorithm adjustment or rerun occurred.

- Final artifact verified: `dist/fpe_self_consistency-0.1.0-py3-none-any.whl`, SHA256 `5f672c5e1bbf8178b59ec22daaf1d11a9e0909e3c4425cbaa02f32db04ed5238`. Fresh external non-editable installation passed all 16 module-origin/hash checks, CLI/API and terminal-resume tests. `reports/campaign-deadline-validation.json` establishes runtime-only equivalence to frozen scientific runs. No task job remains active.
