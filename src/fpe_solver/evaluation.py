"""Independent density evaluation and empirical acceptance; no training imports.

All reported KL, TV and refinement checks are empirical quadratures. Passing
does not certify the population residual, numerical error, or a theorem.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import solve_ivp

from .reference import (
    grid_points,
    initial_log_density,
    potential_gradient,
    reference_density,
    restrict_density,
)

THRESHOLDS = {"kl": 1e-3, "tv": 0.02, "well_mass": 0.01, "mass": 1e-4,
              "reference_refinement_tv": 2e-4, "model_refinement_tv": 1e-3}


def numpy_velocity_divergence(basis, coefficients, t, x):
    """Independent implementation of the documented Fourier/Bernstein layout."""
    x = np.asarray(x, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != tuple(basis.shape):
        raise ValueError("Coefficient shape does not match basis")
    u = float(t) / basis.T
    weights = np.array([math.comb(basis.p, i) * u ** i * (1 - u) ** (basis.p - i) for i in range(basis.p + 1)])
    c = np.tensordot(weights, coefficients, axes=(0, 0))
    modes = np.asarray(basis.modes)
    angle = x @ modes.T
    cos, sin = np.cos(angle), np.sin(angle)
    v = c[0] + cos @ c[1::2] + sin @ c[2::2]
    div = -sin @ np.sum(modes * c[1::2], axis=1) + cos @ np.sum(modes * c[2::2], axis=1)
    return v, div


def _check_time(basis, problem, t):
    if basis.dim != problem.dim or not np.isclose(basis.T, problem.T):
        raise ValueError("Basis and problem must have matching dimension and horizon")
    if not np.isfinite(t) or t < 0 or t > problem.T + 1e-14:
        raise ValueError("Query time lies outside [0,T]")


def log_density(basis, coefficients, problem, x, t, *, rtol=1e-9, atol=1e-11, chunk_size=1024):
    """Backward DOP853 characteristics. No modulo operation enters the ODE.

    Along a trajectory, log rho_t(x_t)=log rho_0(x_0)-int_0^t div v.
    Integrating div v backwards yields the required *negative* integral.
    """
    _check_time(basis, problem, t)
    x = np.asarray(x, dtype=float)
    if x.ndim < 1 or x.shape[-1] != problem.dim or not np.all(np.isfinite(x)):
        raise ValueError("Points must be finite and have trailing dimension dim")
    shape = x.shape[:-1]
    flat = x.reshape(-1, problem.dim)
    result = np.empty(len(flat))
    if t == 0:
        return initial_log_density(problem, x)
    for first in range(0, len(flat), chunk_size):
        points = flat[first:first + chunk_size]
        count = len(points)
        initial = np.concatenate((points.ravel(), np.zeros(count)))
        def rhs(time_value, state, count=count):
            positions = state[:count * problem.dim].reshape(count, problem.dim)
            velocity, divergence = numpy_velocity_divergence(basis, coefficients, time_value, positions)
            return np.concatenate((velocity.ravel(), divergence))
        solution = solve_ivp(rhs, (float(t), 0.), initial, method="DOP853", rtol=rtol, atol=atol)
        if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
            raise RuntimeError(f"Independent backward flow failed: {solution.message}")
        end = solution.y[:, -1]
        source = end[:count * problem.dim].reshape(count, problem.dim)
        result[first:first + count] = initial_log_density(problem, source) + end[count * problem.dim:]
    return result.reshape(shape)


def sample_flow(basis, coefficients, problem, x0, t, *, rtol=1e-9, atol=1e-11):
    """Forward deterministic flow of explicitly supplied initial samples."""
    _check_time(basis, problem, t)
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 2 or x0.shape[1] != problem.dim:
        raise ValueError("Initial samples require shape (count, dim)")
    if t == 0:
        return x0.copy()
    def rhs(time_value, state):
        return numpy_velocity_divergence(basis, coefficients, time_value, state.reshape(x0.shape))[0].ravel()
    sol = solve_ivp(rhs, (0., float(t)), x0.ravel(), method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1].reshape(x0.shape)


def density_metrics(model, reference, dim, n):
    """Raw quadrature: no renormalization or clipping of either density."""
    model, reference = np.asarray(model), np.asarray(reference)
    if model.shape != reference.shape or model.shape != (n ** dim,):
        raise ValueError("Density arrays do not match the grid")
    if not np.all(np.isfinite(model)) or not np.all(np.isfinite(reference)) or np.any(model <= 0) or np.any(reference <= 0):
        raise ValueError("Metrics require finite strictly positive raw densities")
    volume = (2 * np.pi / n) ** dim
    model_mass, reference_mass = model.sum() * volume, reference.sum() * volume
    kl = np.sum(model * (np.log(model) - np.log(reference))) * volume
    # The KL of unnormalised quadratures may be slightly negative; retain it and
    # expose the generalized relative entropy rather than silently clamping it.
    generalized_kl = kl - model_mass + reference_mass
    tv = 0.5 * np.abs(model - reference).sum() * volume
    x = grid_points(dim, n)
    well_error, well_model, well_reference = 0., [], []
    labels = (np.cos(x) < 0).astype(int) @ (2 ** np.arange(dim))
    for label in range(2 ** dim):
        mask = labels == label
        p, q = model[mask].sum() * volume, reference[mask].sum() * volume
        well_model.append(float(p))
        well_reference.append(float(q))
        well_error = max(well_error, abs(p - q))
    observables = np.concatenate((np.cos(x), np.sin(x), (np.cos(x) >= 0).astype(float)), axis=1)
    observable_model = model @ observables * volume
    observable_reference = reference @ observables * volume
    return {"raw_mass": float(model_mass), "reference_mass": float(reference_mass),
            "mass_error": float(abs(model_mass - 1)), "kl": float(kl),
            "generalized_kl": float(generalized_kl), "tv": float(tv),
            "well_mass_error": float(well_error), "well_model": well_model,
            "well_reference": well_reference,
            "observable_order": "cos coordinates, sin coordinates, cos(x)>=0 coordinate wells",
            "observable_model": observable_model.tolist(),
            "observable_reference": observable_reference.tolist(),
            "observable_max_error": float(np.max(np.abs(observable_model - observable_reference)))}


def _time_grid(problem, times):
    times = np.linspace(0, problem.T, 21) if times is None else np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(times > problem.T) or np.any(np.diff(times) < 0):
        raise ValueError("Times must be a sorted nonempty sequence in [0,T]")
    return times


def evaluate_model(basis, coefficients, problem, grid_size=None, times=None, *, max_grid_size=None):
    """Evaluate a frozen model, independently of the evolution controller.

    Adaptive grids must pass reference and model quadrature checks. Resource
    caps produce an explicit unresolved result, never relaxed thresholds.
    """
    started = time.perf_counter()
    times = _time_grid(problem, times)
    n = int(grid_size or (256 if problem.dim == 1 else 32))
    cap = int(max_grid_size or (4096 if problem.dim == 1 else 256))
    if n < 4 or n % 2 or cap < n:
        raise ValueError("Use an even starting grid >=4 and cap >= starting grid")
    report = {"kind": "empirical_uncertified", "thresholds": THRESHOLDS.copy(),
              "problem": problem.to_dict(), "times": times.tolist(),
              "reference_method": "analytic_heat_cell_averages" if not (problem.kappa or problem.coupling) else "scharfetter_gummel_expm",
              "controller_access": "forbidden", "refinement_history": [],
              "roundoff_clipping": "none", "renormalization": "none"}
    ref_started = time.perf_counter()
    coarse = reference_density(problem, times, n)
    reference_seconds = time.perf_counter() - ref_started
    reference_resolved = False
    reference_tv = None
    while 2 * n <= cap:
        ref_started = time.perf_counter()
        fine = reference_density(problem, times, 2 * n)
        delta = 0.5 * np.abs(coarse - restrict_density(fine, problem.dim, n)).sum(-1) * (2 * np.pi / n) ** problem.dim
        reference_seconds += time.perf_counter() - ref_started
        reference_tv = float(delta.max())
        report["refinement_history"].append({"kind": "reference", "coarse": n, "fine": 2 * n, "max_tv": reference_tv})
        n, coarse = 2 * n, fine
        if reference_tv <= THRESHOLDS["reference_refinement_tv"]:
            reference_resolved = True
            break
    flow_started = time.perf_counter()
    reference_before_flow = reference_seconds
    points = grid_points(problem.dim, n)
    model = np.stack([np.exp(log_density(basis, coefficients, problem, points, float(t))) for t in times])
    numerical = np.stack([np.exp(log_density(basis, coefficients, problem, points, float(t), rtol=1e-11, atol=1e-13)) for t in times])
    volume = (2 * np.pi / n) ** problem.dim
    numeric_tv = float((0.5 * np.abs(model - numerical).sum(-1) * volume).max())
    # Model quadrature is checked independently: comparison across grids at the
    # same solver tolerance must not be mistaken for an ODE error estimate.
    coarse_points = grid_points(problem.dim, n // 2)
    coarse_model = np.stack([np.exp(log_density(basis, coefficients, problem, coarse_points, float(t), rtol=1e-11, atol=1e-13)) for t in times])
    grid_tv = float((0.5 * np.abs(coarse_model - restrict_density(numerical, problem.dim, n // 2)).sum(-1) * (4 * np.pi / n) ** problem.dim).max())
    while 2 * n <= cap and (grid_tv > THRESHOLDS["model_refinement_tv"] or np.max(np.abs(numerical.sum(-1) * volume - 1)) > THRESHOLDS["mass"]):
        old, old_n = numerical, n
        n *= 2
        points = grid_points(problem.dim, n)
        numerical = np.stack([np.exp(log_density(basis, coefficients, problem, points, float(t), rtol=1e-11, atol=1e-13)) for t in times])
        model = np.stack([np.exp(log_density(basis, coefficients, problem, points, float(t))) for t in times])
        volume = (2 * np.pi / n) ** problem.dim
        numeric_tv = max(numeric_tv, float((0.5 * np.abs(model - numerical).sum(-1) * volume).max()))
        grid_tv = float((0.5 * np.abs(old - restrict_density(numerical, problem.dim, old_n)).sum(-1) * (2 * np.pi / old_n) ** problem.dim).max())
        report["refinement_history"].append({"kind": "model_quadrature", "coarse": old_n, "fine": n, "max_tv": grid_tv})
        ref_started = time.perf_counter()
        coarse = reference_density(problem, times, n)
        reference_seconds += time.perf_counter() - ref_started
    flow_seconds = time.perf_counter() - flow_started - (reference_seconds - reference_before_flow)
    records = [dict(time=float(t), **density_metrics(p, q, problem.dim, n)) for t, p, q in zip(times, numerical, coarse)]
    maxima = {key: max(record[key] for record in records) for key in ("mass_error", "kl", "generalized_kl", "tv", "well_mass_error")}
    checks = {"mass": maxima["mass_error"] <= THRESHOLDS["mass"],
              "kl": maxima["kl"] <= THRESHOLDS["kl"] and min(r["kl"] for r in records) >= -THRESHOLDS["mass"],
              "tv": maxima["tv"] <= THRESHOLDS["tv"],
              "well_mass": maxima["well_mass_error"] <= THRESHOLDS["well_mass"],
              "reference_refinement": reference_resolved,
              "model_numerical_refinement": numeric_tv <= THRESHOLDS["model_refinement_tv"],
              "model_quadrature_refinement": grid_tv <= THRESHOLDS["model_refinement_tv"]}
    report.update(grid_size=n, metrics=records, maxima=maxima, checks=checks,
                  passed=bool(all(checks.values())), status="passed_empirical" if all(checks.values()) else "failed_or_unresolved",
                  reference_refinement_tv=reference_tv, model_numerical_refinement_tv=numeric_tv,
                  model_quadrature_refinement_tv=grid_tv,
                  costs_seconds={"reference": reference_seconds, "density_queries_and_refinement": flow_seconds,
                                 "total": time.perf_counter() - started},
                  limitations=["Grid and tolerance refinement are empirical, not certified error bounds.",
                               "Population residual and statistical confidence are not certified.",
                               "Positive raw densities are never renormalized before metrics."])
    return report


def _sample_initial(problem, rng, count):
    result = np.empty((count, problem.dim))
    done = np.zeros_like(result, dtype=bool)
    while not np.all(done):
        proposal = rng.uniform(0, 2 * np.pi, result.shape)
        take = (~done) & (rng.random(result.shape) < (1 + problem.amplitude * np.cos(problem.frequency * (proposal - problem.phase))) / (1 + abs(problem.amplitude)))
        result[take] = proposal[take]
        done |= take
    return result


def sde_baseline(problem, *, seed=0, samples=4096, steps=256, times=None):
    """Euler-Maruyama observable baseline with coupled step and sample checks.

    This estimates Fourier observables and well masses, not density KL. A cost
    comparison is valid only when compared on these same observable errors.
    """
    if samples < 4 or steps < 1:
        raise ValueError("Invalid SDE sample count or step count")
    times = _time_grid(problem, times)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    coarse = _sample_initial(problem, rng, 2 * samples)
    fine = coarse.copy()
    records = []
    previous = 0.
    def observables(x):
        return np.concatenate((np.cos(x), np.sin(x), (np.cos(x) >= 0).astype(float)), axis=1)
    for end in times:
        substeps = max(1, math.ceil((end - previous) / problem.T * steps))
        h = (end - previous) / substeps
        if h > 0:
            for _ in range(substeps):
                dw1 = rng.normal(size=fine.shape) * math.sqrt(h / 2)
                dw2 = rng.normal(size=fine.shape) * math.sqrt(h / 2)
                coarse += -potential_gradient(problem, coarse) * h + math.sqrt(2) * (dw1 + dw2)
                fine += -potential_gradient(problem, fine) * (h / 2) + math.sqrt(2) * dw1
                fine += -potential_gradient(problem, fine) * (h / 2) + math.sqrt(2) * dw2
        a, b = observables(coarse), observables(fine)
        records.append({"time": float(end), "fine_mean": b.mean(0).tolist(),
                        "fine_standard_error": (b.std(0, ddof=1) / math.sqrt(2 * samples)).tolist(),
                        "step_refinement_max": float(np.max(np.abs((a - b).mean(0)))),
                        "sample_refinement_max": float(np.max(np.abs(b[:samples].mean(0) - b.mean(0))))})
        previous = end
    return {"kind": "empirical_sde_observables", "seed": seed, "coarse_samples": samples,
            "fine_samples": 2 * samples, "base_steps": steps,
            "observable_order": "cos coordinates, sin coordinates, cos(x)>=0 coordinate wells",
            "records": records, "total_seconds": time.perf_counter() - started,
            "limitations": ["Sampling errors are empirical standard errors; step refinement is not a bound.",
                            "These observables do not establish TV or KL accuracy."]}
