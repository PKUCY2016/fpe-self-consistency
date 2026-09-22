"""Finite-action evolution. This module cannot access a reference solution."""
import math
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from .basis import Basis
from .diagnostics import check_score
from .training import residual_samples


def sample_key(seed, round_index, purpose):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), int(round_index))
    return jax.random.fold_in(key, int(purpose))


def paired_gate(old_values, new_values, numerical_margin=0.):
    old, new = np.asarray(old_values), np.asarray(new_values)
    if old.shape != new.shape or old.ndim != 1 or len(old) < 2:
        raise ValueError("Paired nonempty residual vectors required")
    if not (np.isfinite(old).all() and np.isfinite(new).all()):
        return {"promote": False, "reason": "NONFINITE", "evidence_type": "empirical"}
    difference = old-new
    improvement = float(np.mean(difference))
    se = float(np.std(difference, ddof=1)/math.sqrt(len(difference)))
    old_mean, new_mean = float(old.mean()), float(new.mean())
    threshold = max(.05*old_mean, 3*se + 2*numerical_margin, 1e-10)
    return {"promote": improvement > threshold, "old_residual": old_mean,
            "new_residual": new_mean, "improvement": improvement, "paired_standard_error": se,
            "required_improvement": threshold, "samples": len(old),
            "reason": "IMPROVED" if improvement > threshold else "NO_RESOLVED_IMPROVEMENT",
            "evidence_type": "empirical", "strict_confidence_claim": False}


def validate_candidate(old_basis, old_coeff, old_config, basis, coeff, config, problem, seed, round_index,
                       deadline=None):
    started = time.monotonic()
    key = sample_key(seed, round_index, 9101)
    n = config.validation_samples
    report = {"evidence_type": "empirical", "promote": False}
    while n <= config.max_validation_samples:
        if deadline is not None and time.monotonic() >= deadline:
            return report | {"reason": "BUDGET_EXHAUSTED"}
        # A larger retry uses a fresh paired batch; neither batch is used to train.
        x0 = problem.sample(jax.random.fold_in(key, n), n)
        fine_steps = min(2*max(old_config.steps, config.steps), config.max_steps*2)
        old_fine = residual_samples(old_basis, old_coeff, problem, x0, fine_steps)
        new_fine = residual_samples(basis, coeff, problem, x0, fine_steps)
        subset = x0[:min(128, n)]
        old_coarse = residual_samples(old_basis, old_coeff, problem, subset, fine_steps//2)
        new_coarse = residual_samples(basis, coeff, problem, subset, fine_steps//2)
        e_old = float(np.mean(np.abs(old_coarse-old_fine[:len(subset)])))
        e_new = float(np.mean(np.abs(new_coarse-new_fine[:len(subset)])))
        if not all(math.isfinite(x) for x in (e_old, e_new)):
            return report | {"reason": "NONFINITE_NUMERICS"}
        stable = e_old <= max(1e-6, .01*float(np.mean(old_fine))) and \
                 e_new <= max(1e-6, .01*float(np.mean(new_fine)))
        report = paired_gate(old_fine, new_fine, e_old+e_new)
        if report["reason"] == "NONFINITE":
            return report
        report.update(numerical_change_old=e_old, numerical_change_new=e_new,
                      numerical_consistency=stable, validation_key=np.asarray(jax.random.fold_in(key,n)).tolist())
        if not stable:
            return report | {"promote": False, "reason": "REFINE_INTEGRATION"}
        if report["promote"] or n == config.max_validation_samples:
            break
        # Increase sample size only when sampling uncertainty dominates, not when mean gain is absent.
        if report["improvement"] <= .05*report["old_residual"]:
            break
        n = min(n*2, config.max_validation_samples)
    diagnostic_x = problem.sample(sample_key(seed, round_index, 9102), 4)
    score = check_score(basis, coeff, problem, diagnostic_x, fine_steps)
    old_score = check_score(old_basis, old_coeff, problem, diagnostic_x, fine_steps)
    report.update(score_consistency=score, incumbent_score_consistency=old_score,
                  seconds=time.monotonic()-started)
    if not score["passed"] or not old_score["passed"]:
        report.update(promote=False, reason="SCORE_INCONSISTENCY")
    return report


def diagnose(basis, coeff, config, problem, seed, round_index, recent_history, rejected_events=()):
    """Current-flow tests only. No true density, KL, or offline evaluator."""
    x = problem.sample(sample_key(seed, round_index, 8101), min(config.validation_samples, 1024))
    r = residual_samples(basis, coeff, problem, x, config.steps)
    coarse = r[:min(128, len(r))]
    fine = residual_samples(basis, coeff, problem, x[:len(coarse)], 2*config.steps)
    result = {"residual": float(r.mean()), "standard_error": float(r.std(ddof=1)/np.sqrt(len(r))),
              "numerical_change": float(np.mean(np.abs(fine-coarse)))}
    if not np.isfinite(r).all() or not np.isfinite(fine).all():
        return result | {"failure": "NONFINITE"}
    result["numerically_stable"] = result["numerical_change"] <= max(1e-6,.01*float(fine.mean()))
    result["sampling_unstable"] = 3*result["standard_error"] > max(1e-6,.2*result["residual"])
    losses = [h["loss"] for h in recent_history if math.isfinite(h["loss"])]
    result["oscillatory"] = bool(len(losses) >= 20 and np.mean(losses[-10:]) > 1.1*np.mean(losses[:10]))
    result["potential_optimization_regression"] = any(
        rejected_optimization_regression(event) for event in rejected_events)
    # Probe newly available coefficient directions at an exact embedding of the incumbent.
    # RMS normalization prevents more coefficients from automatically winning.
    from .flow import integrate
    probe_x = x[:min(32,len(x))]
    scores = {}
    for name, K, p in (("increase_space", min(config.K+1,8), config.p),
                       ("increase_time", config.K, min(config.p+2,15))):
        if K == config.K and p == config.p:
            scores[name] = 0.
            continue
        expanded = Basis(problem.dim,K,p,problem.T)
        prolonged = basis.prolong(coeff, expanded)
        fun = lambda c, expanded=expanded: jnp.mean(integrate(expanded,c,problem,probe_x,config.steps)[:,-1])
        g = np.asarray(jax.jit(jax.grad(fun))(prolonged))
        if name == "increase_space":
            old_modes = {tuple(k) for k in basis.modes}
            new_features = [j for i,k in enumerate(expanded.modes) if tuple(k) not in old_modes
                            for j in (1+2*i,2+2*i)]
            innovation = g[:,new_features,:]
        else:
            # Columns of E are the old Bernstein polynomials in the elevated basis.
            E = np.eye(config.p+1)
            for degree in range(config.p,p):
                w = np.arange(degree+2)/(degree+1)
                E = w[:,None]*np.vstack((np.zeros((1,E.shape[1])),E)) + \
                    (1-w[:,None])*np.vstack((E,np.zeros((1,E.shape[1]))))
            projector = np.eye(p+1)-E@np.linalg.pinv(E)
            innovation = np.einsum("ab,bfd->afd",projector,g)
        scores[name] = float(np.sqrt(np.mean(innovation*innovation)))
    result["probe_gradient_rms"] = scores
    return result


def rejected_optimization_regression(event):
    """Internal paired evidence that training worsened an exactly embedded model.

    This labels a possible optimization failure, not proof of a large learning
    rate. It only prioritizes the already declared one-setting reduce_lr trial.
    Reference densities and offline distribution errors are never inspected.
    """
    evidence = event.get("validation", {})
    if event.get("promoted", False) or not event.get("qualified", False):
        return False
    if not evidence.get("numerical_consistency", False) or not all(
        evidence.get(key, {}).get("passed", False)
        for key in ("score_consistency", "incumbent_score_consistency")):
        return False
    keys = ("old_residual", "new_residual", "paired_standard_error",
            "numerical_change_old", "numerical_change_new")
    values = [evidence.get(key) for key in keys]
    if any(v is None or not math.isfinite(v) for v in values):
        return False
    old, new, se, error_old, error_new = values
    margin = max(.1*old, 3*se+2*(error_old+error_new), 1e-10)
    return bool(new-old > margin)


def select_action(config, diagnostic, rejected_actions=()):
    priority = []
    if not diagnostic.get("numerically_stable", False):
        priority.append("refine_steps")
    if diagnostic.get("sampling_unstable",False):
        priority.append("increase_batch")
    if diagnostic.get("oscillatory",False) or diagnostic.get("potential_optimization_regression",False):
        priority.append("reduce_lr")
    probes = diagnostic.get("probe_gradient_rms",{})
    priority.extend(sorted(("increase_space","increase_time"), key=lambda k:-probes.get(k,0.)))
    priority.extend(("refine_steps","increase_batch","reduce_lr"))
    changes = {"refine_steps": {"steps": config.steps*2},
               "increase_batch": {"batch": config.batch*2},
               "reduce_lr": {"learning_rate": config.learning_rate/2},
               "increase_space": {"K": config.K+1},
               "increase_time": {"p": config.p+2}}
    for action in dict.fromkeys(priority):
        if action in rejected_actions:
            continue
        c = changes[action]
        if c.get("steps",0)>config.max_steps or c.get("batch",0)>1024 or c.get("K",0)>8 or \
           c.get("p",0)>15 or c.get("learning_rate",1)<1e-4:
            continue
        return action, replace(config,**c)
    return None, config
