"""Resumable, single-writer experiment orchestration; no reference-answer feedback."""
import hashlib
import importlib.metadata
import platform
import time
from pathlib import Path

import jax
import numpy as np

from .basis import Basis
from .config import SolverConfig
from .controller import diagnose, sample_key, select_action, validate_candidate
from .problems import Problem
from .storage import load_arrays, read_json, run_lock, save_arrays, write_json
from .training import optimizer, train_candidate

TERMINAL = {"EMPIRICAL_TARGET_REACHED", "PLATEAU", "NUMERICAL_FAILURE", "BUDGET_EXHAUSTED", "TRAINED"}


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": {m: importlib.metadata.version(m) for m in ("jax","jaxlib","optax","numpy","scipy")},
            "source_hashes": {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
            "devices": [str(x) for x in jax.devices()], "float64": bool(jax.config.jax_enable_x64)}


def create_run(path, problem, config, seed, mode):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    state = {"schema": 1, "problem": problem.to_dict(), "initial_config": config.to_dict(),
             "seed": int(seed), "mode": mode, "status": "RUNNING", "elapsed_seconds": 0.,
             "next_round": 0, "champion": None, "candidate": None, "events": [],
             "rejected_actions": [], "consecutive_rejections": 0, "environment": environment(),
             "evidence_type": "empirical", "certified_error_bound": None}
    write_json(path/"state.json", state)
    return state


def get_model(path, which="champion"):
    path = Path(path)
    state = read_json(path/"state.json")
    item = state.get(which)
    if not item or not item.get("checkpoint"):
        raise ValueError(f"Run does not contain a {which} checkpoint")
    cp = item["checkpoint"]
    coeff, _, _ = load_arrays(path/cp["file"],expected_hash=cp["sha256"])
    return Basis(**item["basis"]), coeff, Problem(**state["problem"])


def execute(path, *, resume=False):
    path = Path(path)
    with run_lock(path):
        state = read_json(path/"state.json")
        if state["status"] in TERMINAL:
            return state  # Resuming a completed experiment never silently starts another.
        if not resume and state["status"] == "INTERRUPTED":
            raise ValueError("Use resume for an interrupted run")
        if resume:
            current = environment()
            previous = state["environment"]
            differences = {key: {"recorded": previous.get(key), "current": current.get(key)}
                           for key in ("source_hashes", "versions", "python", "float64")
                           if previous.get(key) != current.get(key)}
            state.setdefault("resume_environments", []).append({"environment": current,
                                                               "differences": differences})
            write_json(path/"state.json", state)
            if differences:
                raise ValueError("Resume environment changed; recorded differences. Start a new run for changed code/dependencies.")
        initial_config = SolverConfig(**state["initial_config"])
        problem = Problem(**state["problem"])
        started = time.monotonic()
        prior_seconds = state["elapsed_seconds"]
        deadline = started + max(0., initial_config.max_seconds-prior_seconds)

        def persist():
            state["elapsed_seconds"] = prior_seconds + time.monotonic()-started
            write_json(path/"state.json", state)

        state["status"] = "RUNNING"
        try:
            while state["next_round"] < initial_config.max_rounds:
                if time.monotonic() >= deadline:
                    state["status"] = "BUDGET_EXHAUSTED"
                    break
                champion = state["champion"]
                if state["candidate"] is None:
                    if champion is None:
                        config = initial_config
                        basis = Basis(problem.dim,config.K,config.p,problem.T)
                        coeff = basis.zeros()
                        action = "initial"
                    else:
                        basis, coeff, _ = get_model(path)
                        config = SolverConfig(**champion["config"])
                        diagnostic = diagnose(basis,coeff,config,problem,state["seed"],state["next_round"],
                                              champion.get("history",[]),
                                              rejected_events=state["events"][-state["consecutive_rejections"]:]
                                              if state["consecutive_rejections"] else ())
                        state["last_diagnostic"] = diagnostic
                        if "failure" in diagnostic:
                            state["status"] = "NUMERICAL_FAILURE"
                            break
                        if diagnostic["numerically_stable"] and diagnostic["residual"] <= config.residual_target \
                                and champion.get("qualified",False):
                            state["status"] = "EMPIRICAL_TARGET_REACHED"
                            break
                        action, config = select_action(config,diagnostic,state["rejected_actions"])
                        if action is None or state["consecutive_rejections"] >= 3:
                            state["status"] = "PLATEAU"
                            break
                        expanded = Basis(problem.dim,config.K,config.p,problem.T)
                        new_coeff = basis.prolong(coeff,expanded)
                        test_x = problem.sample(sample_key(state["seed"],state["next_round"],777),16)
                        errors = [np.max(np.abs(np.asarray(basis.velocity(coeff,t,test_x))-
                                               np.asarray(expanded.velocity(new_coeff,t,test_x))))
                                  for t in (0.,.37*problem.T,problem.T)]
                        if max(errors)>1e-10:
                            raise RuntimeError("Prolongation changed the incumbent function")
                        basis, coeff = expanded, new_coeff
                    r = state["next_round"]
                    candidate = {"round":r,"action":action,"basis":basis.to_dict(),"config":config.to_dict(),
                                 "update":0,"history":[],"checkpoint":None,"parent":None if champion is None
                                 else champion["checkpoint"]["sha256"]}
                    state["candidate"] = candidate
                    key = sample_key(state["seed"],r,100)
                    opt = optimizer(config).init(coeff)
                else:
                    candidate = state["candidate"]
                    config = SolverConfig(**candidate["config"])
                    basis = Basis(**candidate["basis"])
                    template = optimizer(config).init(basis.zeros())
                    cp = candidate["checkpoint"]
                    if cp is None:
                        if candidate["action"] == "initial":
                            coeff = basis.zeros()
                        else:
                            parent_basis,parent_coeff,_ = get_model(path)
                            coeff = parent_basis.prolong(parent_coeff,basis)
                        key = sample_key(state["seed"],candidate["round"],100)
                        opt = optimizer(config).init(coeff)
                    else:
                        coeff,key,opt = load_arrays(path/cp["file"],template,cp["sha256"])

                def checkpoint(c,o,k,update,history,candidate=candidate):
                    # Unique update files make a crash between artifact and manifest writes recoverable.
                    filename = f"checkpoints/round-{candidate['round']:03d}-update-{update:05d}.npz"
                    digest = save_arrays(path/filename,c,k,o)
                    candidate["checkpoint"] = {"file":filename,"sha256":digest}
                    candidate["update"] = update
                    known = {h["update"]:h for h in candidate["history"]}
                    known.update({h["update"]:h for h in history})
                    candidate["history"] = list(known.values())[-100:]
                    persist()

                if candidate["checkpoint"] is None:
                    checkpoint(coeff,opt,key,0,[])
                result = train_candidate(basis,coeff,problem,config,key,opt_state=opt,
                                         start_update=candidate["update"],deadline=deadline,checkpoint=checkpoint)
                coeff = result["coeff"]
                candidate["training_seconds"] = candidate.get("training_seconds",0.)+result["seconds"]
                if result["first_step_compile_and_execution_seconds"] > 0:
                    candidate["first_step_compile_and_execution_seconds"] = candidate.get(
                        "first_step_compile_and_execution_seconds", 0.) + result["first_step_compile_and_execution_seconds"]
                if result["status"] != "TRAINED":
                    state["status"] = result["status"]
                    break
                if champion is None:
                    old_basis,old_coeff,old_config = basis,coeff,config
                else:
                    old_basis,old_coeff,_ = get_model(path)
                    old_config = SolverConfig(**champion["config"])
                validation = validate_candidate(old_basis,old_coeff,old_config,basis,coeff,config,problem,
                                                state["seed"],candidate["round"],deadline)
                if time.monotonic() >= deadline:
                    validation.update(promote=False, reason="BUDGET_EXHAUSTED")
                candidate["validation"] = validation
                qualified = bool(validation.get("reason") != "BUDGET_EXHAUSTED" and
                                 validation.get("numerical_consistency",False) and
                                 validation.get("score_consistency",{}).get("passed",False))
                candidate["qualified"] = qualified
                promote = qualified and (champion is None or validation["promote"])
                event = {"round":candidate["round"],"action":candidate["action"],"promoted":promote,
                         "qualified":qualified,"validation":validation,"checkpoint":candidate["checkpoint"],
                         "history":candidate["history"],
                         "config":candidate["config"],"parent":candidate["parent"],
                         "training_seconds":candidate["training_seconds"],
                         "first_step_compile_and_execution_seconds":candidate.get("first_step_compile_and_execution_seconds",0.)}
                state["events"].append(event)
                if champion is None and not qualified:
                    write_json(path/f"round-{candidate['round']:03d}.json",event)
                    state["status"] = "BUDGET_EXHAUSTED" if validation.get("reason") == "BUDGET_EXHAUSTED" else "NUMERICAL_FAILURE"
                    break
                write_json(path/f"round-{candidate['round']:03d}.json",event)
                if promote:
                    state["champion"] = dict(candidate)
                    state["rejected_actions"] = []
                    state["consecutive_rejections"] = 0
                else:
                    state["rejected_actions"].append(candidate["action"])
                    state["consecutive_rejections"] += 1
                state["candidate"] = None
                state["next_round"] += 1
                persist()
                if validation.get("reason") == "BUDGET_EXHAUSTED":
                    state["status"] = "BUDGET_EXHAUSTED"
                    break
                if state["mode"] == "train":
                    state["status"] = "TRAINED" if qualified else "NUMERICAL_FAILURE"
                    break
            else:
                state["status"] = "BUDGET_EXHAUSTED"
        except KeyboardInterrupt:
            state["status"] = "INTERRUPTED"
        except Exception as exc:
            state["status"] = "INTERRUPTED"
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            persist()
            raise
        persist()
        return state
