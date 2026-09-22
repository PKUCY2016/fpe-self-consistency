# Delivery closeout — completed 2026-09-22

- [x] All 45 declared acceptance-v2 tasks ended and produced scientific records. Development adaptive tests: 15/15; heldout: 5/6. Complete engineering acceptance remains **FAILED**, with no missing task or process failure. No post-audit algorithm change or rerun.
- [x] The campaign deadline is now 7200 seconds including termination/reaping. Only campaign execution code changed; mathematical functions and frozen specification are unchanged. See `campaign-deadline-validation.json` for AST/source equivalence and actual receipt durations.
- [x] Final CPU tests: 116 passed, one Linux-specific skip. DSW runtime tests: 19 passed, including the locally skipped case. Ruff passed.
- [x] Final wheel rebuilt and installed non-editably with locked dependencies in a fresh external environment. All installed module hashes, CLI/API and recovery smoke checks passed. See `delivery.json`; prior packages and reports remain preserved.
- [x] Both real image trials completed and were independently inspected: rank 32 and rank 128 fail visual generation. All outputs, raw arrays, parameters and evidence remain retained; no additional image training.
- [x] DSW process lifecycle repair and CPU regression completed. All 12 jobs terminal, no live recorded/project-path processes, no GPU compute users in final inventory.
- [x] Acceptance report, image report, independent failure reviews and artifact links checked. Unit-test success, scientific acceptance failure and uncertified error bounds are clearly separated.

No background job remains active for this task. Failed scientific acceptance is an observed outcome, not a pending validation step.
