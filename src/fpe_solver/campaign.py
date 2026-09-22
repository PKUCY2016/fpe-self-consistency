"""Frozen, offline acceptance driver. Never called by the evolution controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from .basis import Basis
from .config import SolverConfig
from .controller import sample_key, validate_candidate
from .evaluation import THRESHOLDS, evaluate_model, sample_flow, sde_baseline
from .problems import Problem, benchmark
from .runner import create_run, environment, execute, get_model
from .storage import read_json, run_lock, save_arrays, write_json
from .training import optimizer, train_candidate


def specification():
    return {
        "version": 1, "seeds": [0, 1, 2], "times": 21,
        "thresholds": THRESHOLDS, "default_config": SolverConfig().to_dict(),
        "cases": ["uniform", "heat1d", "heat2d", "doublewell", "coupled2d"],
        "evolution_case": "evolution", "control_budget_seconds": 120,
        "control_max_blocks": 12, "control_median_kl_reduction": .30,
        "control_min_improved_seeds": 2, "campaign_seconds": 28800,
        "stress": ["stress_barrier", "stress_frequency"],
        "audit": [{"base": "heat1d", "amplitude": .27, "phase": .71},
                  {"base": "coupled2d", "amplitude": .31, "phase": 1.17}],
        "audit_policy": "If algorithm is changed using these results, this batch becomes development data.",
        "cost_observable_target": .01,
        "evidence": "empirical; thresholds never relaxed; all seeds retained",
    }


def fixed_control(path, problem, seed, seconds):
    """Same K=2 configuration, at most 12x500 updates, same total wall budget cap.

    Validation and compilation consume the cap. The unused 15s validation
    reserve is explicit. Optimizer state continues across blocks. No reference
    density is queried during any training or model selection.
    """
    path.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = replace(SolverConfig(), max_seconds=seconds)
    basis = Basis(problem.dim, config.K, config.p, problem.T)
    coeff, key = basis.zeros(), sample_key(seed, 0, 100)
    opt, records, compile_seconds = optimizer(config).init(coeff), [], 0.
    deadline = started + seconds
    compiled = None
    for block in range(12):
        result = train_candidate(basis, coeff, problem, config, key, opt_state=opt,
                                 deadline=deadline-15., compiled_update=compiled)
        compiled = result["compiled_update"]
        coeff, key, opt = result["coeff"], result["key"], result["opt_state"]
        compile_seconds += result["first_step_compile_and_execution_seconds"]
        records.append({k: result[k] for k in ("status", "update", "seconds", "history")})
        filename = f"checkpoints/block-{block:03d}.npz"
        digest = save_arrays(path/filename, coeff, key, opt)
        write_json(path/"progress.json", {"blocks": records, "checkpoint": filename, "sha256": digest})
        if result["status"] != "TRAINED":
            break
    validation = validate_candidate(basis, coeff, config, basis, coeff, config,
                                    problem, seed, 1000, deadline)
    elapsed = time.monotonic()-started
    qualified = validation.get("numerical_consistency", False) and validation.get("score_consistency", {}).get("passed", False)
    champion = {"basis": basis.to_dict(), "config": config.to_dict(),
                "checkpoint": {"file": filename, "sha256": digest}, "qualified": qualified,
                "validation": validation}
    state = {"problem": problem.to_dict(), "champion": champion if qualified else None,
             "candidate": champion, "elapsed_seconds": elapsed, "blocks": records,
             "environment": environment(), "seed": seed, "mode": "fixed_budget_control",
             "status": result["status"] if qualified else "NUMERICAL_FAILURE",
             "training_stop_reason": result["status"], "model_qualified": qualified,
             "budget_cap_seconds": seconds, "within_budget": elapsed <= seconds,
             "first_step_compile_and_execution_seconds": compile_seconds}
    write_json(path/"state.json", state)
    return state


def evaluation_with_cost(path, include_sde=False):
    basis, coeff, problem = get_model(path)
    state = read_json(path/"state.json")
    report = evaluate_model(basis, coeff, problem)
    report["checkpoint"] = state["champion"]["checkpoint"]
    report["solver_status"] = state["status"]
    report["costs_seconds"]["training_search_validation_total"] = state["elapsed_seconds"]
    report["costs_seconds"]["first_updates_compile_and_execution"] = sum(
        event.get("first_step_compile_and_execution_seconds", 0.) for event in state.get("events", [])) or state.get("first_step_compile_and_execution_seconds", 0.)
    x = np.asarray(problem.sample(jax.random.PRNGKey(90813), 2048))
    started = time.perf_counter()
    sample_flow(basis, coeff, problem, x, problem.T)
    report["costs_seconds"]["dop853_sample_2048"] = time.perf_counter()-started
    report["costs_seconds"]["total_including_training_and_sampling"] = report["costs_seconds"]["total"] + state["elapsed_seconds"] + report["costs_seconds"]["dop853_sample_2048"]
    if include_sde:
        attempts = []
        for samples, steps in ((4096, 256), (16384, 512), (65536, 1024)):
            result = sde_baseline(problem, seed=state["seed"], samples=samples, steps=steps)
            errors = [np.max(np.abs(np.asarray(record["fine_mean"])-truth["observable_reference"]))
                      for record, truth in zip(result["records"], report["metrics"])]
            result["observable_max_error"] = float(max(errors))
            result["empirical_3se_max"] = float(3*max(max(r["fine_standard_error"]) for r in result["records"]))
            result["matched_observable_target"] = bool(max(errors) <= .01 and result["empirical_3se_max"] <= .01
                and max(r["step_refinement_max"] for r in result["records"]) <= .01)
            attempts.append(result)
            if result["matched_observable_target"]:
                break
        model_error = max(r["observable_max_error"] for r in report["metrics"])
        report["classical_comparison"] = {"target": .01, "observables": result["observable_order"],
            "sde_attempts": attempts, "sde_total_seconds": sum(a["total_seconds"] for a in attempts),
            "model_max_error": model_error,
            "same_observable_precision_reached": model_error <= .01 and result["matched_observable_target"],
            "finite_volume_reference_seconds": report["costs_seconds"]["reference"],
            "note": "Costs include refinement attempts. No SDE KL/TV assertion. Analytic heat reference is labeled separately."}
    write_json(path/"evaluation.json", report)
    return {"status": state["status"], "passed": report["passed"], "maxima": report["maxima"],
            "elapsed_seconds": state["elapsed_seconds"], "report": str(path/"evaluation.json")}


def worker(spec, path):
    problem = Problem(**spec["problem"])
    if spec["mode"] == "fixed_budget_control":
        state = fixed_control(path, problem, spec["seed"], spec["seconds"])
    else:
        config = replace(SolverConfig(), max_seconds=spec.get("seconds", 7200.))
        if "config_overrides" in spec:
            config = replace(config, **spec["config_overrides"])
        create_run(path, problem, config, spec["seed"], spec["mode"])
        state = execute(path)
    if state["champion"] is None:
        return {"status": state["status"], "passed": False, "reason": "No qualified model", "elapsed_seconds": state["elapsed_seconds"]}
    return evaluation_with_cost(path, include_sde=spec.get("costs", False))


def tasks(phase):
    manifest = specification()
    for seed in manifest["seeds"]:
        if phase == "fixed":
            for case in manifest["cases"]:
                yield f"fixed-{case}-s{seed}", {"mode": "train", "problem": benchmark(case).to_dict(), "seed": seed, "costs": seed == 0}
        elif phase == "adaptive":
            for case in manifest["cases"]:
                yield f"adaptive-{case}-s{seed}", {"mode": "evolve", "problem": benchmark(case).to_dict(), "seed": seed, "costs": seed == 0}
        elif phase == "control":
            for mode in ("evolve", "fixed_budget_control"):
                yield f"control-{mode}-s{seed}", {"mode": mode, "problem": benchmark("evolution").to_dict(), "seed": seed, "seconds": manifest["control_budget_seconds"]}
        elif phase == "audit":
            for i, audit in enumerate(manifest["audit"]):
                yield f"audit-{i}-s{seed}", {"mode": "evolve", "problem": benchmark(audit["base"], amplitude=audit["amplitude"], phase=audit["phase"]).to_dict(), "seed": seed}
    if phase == "stress":
        for case in manifest["stress"]:
            yield f"stress-{case}", {"mode": "evolve", "problem": benchmark(case).to_dict(), "seed": 0, "seconds": 300.}
        yield "stress-coarse-small", {"mode": "evolve", "problem": benchmark("doublewell").to_dict(), "seed": 0,
                                      "seconds": 300., "config_overrides": {"steps": 4, "batch": 8}}


def summarize(root):
    results = {p.parent.name: read_json(p) for p in root.glob("*/result.json")}
    spec = specification()
    for receipt in root.glob("*.receipt.json"):
        name = receipt.name.removesuffix(".receipt.json")
        observed = read_json(receipt)
        if name not in results:
            results[name] = {"passed": False, "status": observed["status"],
                             "reason": "No completed scientific result", "receipt": str(receipt)}
    comparison = []
    for seed in spec["seeds"]:
        names = [f"control-{m}-s{seed}" for m in ("evolve", "fixed_budget_control")]
        if all((root/n/"evaluation.json").exists() for n in names):
            evaluations = [read_json(root/n/"evaluation.json") for n in names]
            values = [e["metrics"][-1]["kl"] for e in evaluations]
            states = [read_json(root/n/"state.json") for n in names]
            comparison.append({"seed": seed, "adaptive_final_kl": values[0], "fixed_final_kl": values[1],
                "improved": values[0] < values[1], "adaptive_seconds": states[0]["elapsed_seconds"],
                "fixed_seconds": states[1]["elapsed_seconds"], "shared_budget_cap_seconds": spec["control_budget_seconds"],
                "budget_compliant": all(s["elapsed_seconds"] <= spec["control_budget_seconds"] for s in states),
                "numerically_resolved": all(all(e["checks"][k] for k in ("mass", "reference_refinement", "model_numerical_refinement", "model_quadrature_refinement")) for e in evaluations)})
    if len(comparison) == 3 and all(np.isfinite(r["adaptive_final_kl"]) and np.isfinite(r["fixed_final_kl"])
        and r["fixed_final_kl"] > 1e-4 and r["adaptive_final_kl"] >= -1e-4
        and r["budget_compliant"] and r["numerically_resolved"] for r in comparison):
        reduction = 1-np.median([r["adaptive_final_kl"] for r in comparison])/np.median([r["fixed_final_kl"] for r in comparison])
        control = {"passed": bool(reduction >= .3 and sum(r["improved"] for r in comparison) >= 2), "median_kl_reduction": float(reduction), "seeds": comparison}
    else:
        control = {"passed": False, "status": "INCOMPLETE", "seeds": comparison}
    required = [name for phase in ("fixed", "adaptive", "control", "audit", "stress") for name, _ in tasks(phase)]
    adaptive = [f"adaptive-{case}-s{seed}" for case in spec["cases"] for seed in spec["seeds"]]
    audit = [name for name, _ in tasks("audit")]
    report = {"specification_sha256": hashlib.sha256((root/"specification.json").read_bytes()).hexdigest(),
              "results": results, "control": control, "missing": [name for name in required if name not in results],
              "adaptive_cases_passed": all(results.get(n, {}).get("passed", False) for n in adaptive),
              "heldout_audit_passed": all(results.get(n, {}).get("passed", False) for n in audit),
              "scientific_proof_complete": False,
              "all_tasks_produced_scientific_results": all((root/n/"result.json").exists() for n in required),
              "process_failures": {n: read_json(root/f"{n}.receipt.json")["status"] for n in required
                                   if (root/f"{n}.receipt.json").exists()
                                   and read_json(root/f"{n}.receipt.json")["status"] != "EXITED"}}
    report["engineering_acceptance_passed"] = report["adaptive_cases_passed"] and report["heldout_audit_passed"] and control["passed"] and not report["missing"] and report["all_tasks_produced_scientific_results"] and not report["process_failures"]
    write_json(root/"summary.json", report)
    return report


def wait_worker_until(process, deadline):
    """Bound this single Popen worker by one total monotonic deadline.

    Ten seconds are reserved for TERM and one for KILL/reaping. Popen's
    terminate/kill target the tracked leader; this helper does not certify
    descendant-process cleanup. Campaign workers launch no child processes.
    An unconfirmed reap is unresolved and must block further submission.
    """
    try:
        code = process.wait(timeout=max(0., deadline-time.monotonic()-11.))
        if time.monotonic() <= deadline:
            return code, "EXITED" if code == 0 else "FAILED"
    except (subprocess.TimeoutExpired, OSError):
        pass

    term_deadline = min(deadline-1., time.monotonic()+10.)
    try:
        process.terminate()
    except OSError:
        pass  # A concurrent exit is resolved by the bounded wait below.
    try:
        code = process.wait(timeout=max(0., term_deadline-time.monotonic()))
        return code, "TIMEOUT"
    except (subprocess.TimeoutExpired, OSError):
        pass

    reap_deadline = min(deadline, time.monotonic()+1.)
    try:
        process.kill()
    except OSError:
        pass
    try:
        code = process.wait(timeout=max(0., reap_deadline-time.monotonic()))
        return code, "TIMEOUT"
    except (subprocess.TimeoutExpired, OSError):
        return process.returncode, "TIMEOUT_UNREAPED"


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("fixed", "adaptive", "control", "audit", "stress", "summary"), default="fixed")
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args(argv)
    if args.worker_spec:
        result = worker(read_json(args.worker_spec), args.out)
        write_json(args.out/"result.json", result)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out/"specification.json"
    if not manifest.exists():
        write_json(manifest, specification())
        write_json(args.out/"environment.json", environment())
    elif read_json(manifest) != specification():
        raise ValueError("Frozen specification differs; use a new campaign")
    if args.phase == "summary":
        print(json.dumps(summarize(args.out), indent=2))
        return
    spent = sum(read_json(p).get("wall_seconds", 0.) for p in args.out.glob("*.receipt.json"))
    unresolved = [p for p in args.out.glob("*.receipt.json") if read_json(p)["status"] in {"RUNNING", "TIMEOUT_UNREAPED"}]
    if unresolved:
        raise RuntimeError(f"Unresolved campaign jobs: {unresolved}; query their original processes first")
    for name, spec in tasks(args.phase):
        path, receipt = args.out/name, args.out/f"{name}.receipt.json"
        if receipt.exists():
            if read_json(receipt)["status"] in {"RUNNING", "TIMEOUT_UNREAPED"}:
                raise RuntimeError(f"Uncertain existing task {name}; inspect process before resubmission")
            continue
        if path.exists():
            raise RuntimeError(f"Existing result directory without receipt: {path}")
        if spent >= 28800:
            break
        spec_path = args.out/f"{name}.request.json"
        write_json(spec_path, spec)
        started = time.monotonic()
        with (args.out/f"{name}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "fpe_solver.campaign", "--out", str(path), "--worker-spec", str(spec_path)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(receipt, {"status": "RUNNING", "pid": process.pid, "request": spec})
            deadline = started + min(7200., 28800-spent)
            code, status = wait_worker_until(process, deadline)
        duration = time.monotonic()-started
        spent += duration
        write_json(receipt, {"status": status, "pid": process.pid, "exit_code": code, "wall_seconds": duration, "request": spec})
        print(json.dumps({"task": name, "process_status": status, "seconds": duration}), flush=True)
        summarize(args.out)
        if status == "TIMEOUT_UNREAPED":
            raise RuntimeError(f"Worker {name} was not confirmed reaped; inspect the original worker before continuing")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--worker-spec" in argv:
        return _main(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path)
    known, _ = parser.parse_known_args(argv)
    if known.out is None:
        return _main(argv)
    known.out.mkdir(parents=True, exist_ok=True)
    with run_lock(known.out):
        return _main(argv)


if __name__ == "__main__":
    main()

