import json

import numpy as np
import pytest

from fpe_solver.basis import Basis
from fpe_solver.evaluation import (
    density_metrics,
    evaluate_model,
    log_density,
    numpy_velocity_divergence,
    sample_flow,
    sde_baseline,
)
from fpe_solver.problems import Problem, benchmark
from fpe_solver.reference import (
    analytic_heat_density,
    fv_generator,
    grid_points,
    potential,
    reference_density,
    restrict_density,
)


@pytest.mark.parametrize("dim", [1, 2])
def test_fv_conservative_positive_stationary(dim):
    problem = Problem(dim=dim, kappa=.6, coupling=.25 if dim == 2 else 0)
    generator = fv_generator(problem, 16)
    np.testing.assert_allclose(np.asarray(generator.sum(axis=0)), 0, atol=2e-14)
    matrix = generator.toarray()
    np.fill_diagonal(matrix, 0)
    assert np.min(matrix) >= 0
    pi = np.exp(-potential(problem, grid_points(dim, 16)))
    np.testing.assert_allclose(generator @ pi, 0, atol=2e-14)
    rho = reference_density(problem, [0, .05, .1], 16)
    assert np.min(rho) > 0
    np.testing.assert_allclose(rho.sum(-1) * (2 * np.pi / 16) ** dim, 1, atol=5e-14)


@pytest.mark.parametrize("dim", [1, 2])
def test_fv_heat_second_order(dim):
    problem = Problem(dim=dim, amplitude=.4, T=.3)
    errors = []
    for n in (16, 32):
        finite_volume = reference_density(problem, [problem.T], n, force_fv=True)[0]
        exact = analytic_heat_density(problem, grid_points(dim, n), problem.T, 2 * np.pi / n)
        errors.append(np.abs(finite_volume - exact).sum() * (2 * np.pi / n) ** dim)
    assert errors[1] < .3 * errors[0]


def test_restrict_exact_cell_heat():
    problem = Problem(dim=2, frequency=3)
    coarse = reference_density(problem, [0, .2], 16)
    fine = reference_density(problem, [0, .2], 32)
    np.testing.assert_allclose(coarse, restrict_density(fine, 2, 16), atol=2e-17)


@pytest.mark.parametrize("dim", [1, 2])
def test_numpy_velocity_independent_matches_jax(dim):
    basis = Basis(dim=dim, K=2, p=3)
    rng = np.random.default_rng(7)
    coefficient = rng.normal(size=basis.shape)
    x = rng.normal(size=(7, dim))
    v, div = numpy_velocity_divergence(basis, coefficient, .3, x)
    jv, _, jd, _ = basis.spatial(coefficient, .3, x)
    np.testing.assert_allclose(v, jv, atol=4e-14)
    np.testing.assert_allclose(div, jd, atol=4e-14)


def test_backward_density_translation_and_sine_compression():
    problem = Problem(amplitude=.4, T=.7)
    basis = Basis(K=1, p=1, T=problem.T)
    coefficient = np.zeros(basis.shape)
    coefficient[:, 0, 0] = .3
    x = np.linspace(.01, 2 * np.pi, 30)[:, None]
    expected = np.asarray(problem.log_density0(x - .3 * problem.T))
    np.testing.assert_allclose(log_density(basis, coefficient, problem, x, problem.T), expected, atol=1e-10)
    np.testing.assert_allclose(sample_flow(basis, coefficient, problem, x, problem.T), x + .3 * problem.T, atol=1e-10)
    # Uniform initial density, v=a sin(x): exact density tests divergence sign.
    uniform = Problem(amplitude=0, T=.7)
    coefficient[:] = 0
    coefficient[:, 2, 0] = .8
    a = .8 * problem.T
    expected = -np.log(2 * np.pi * (np.cosh(a) + np.sinh(a) * np.cos(x[:, 0])))
    np.testing.assert_allclose(log_density(basis, coefficient, uniform, x, problem.T), expected, atol=2e-9)


def test_metrics_keep_mass_error_and_raw_kl():
    uniform = np.full(16, 1 / (2 * np.pi))
    result = density_metrics(.99 * uniform, uniform, 1, 16)
    assert result["mass_error"] == pytest.approx(.01)
    assert result["kl"] < 0
    assert result["generalized_kl"] > 0
    with pytest.raises(ValueError, match="positive"):
        density_metrics(np.zeros(16), uniform, 1, 16)


def test_evaluate_uniform_pass_and_wrong_heat_fail():
    basis = Basis(K=1, p=1)
    coefficient = np.zeros(basis.shape)
    uniform = evaluate_model(basis, coefficient, benchmark("uniform"), grid_size=16, times=[0, .5, 1], max_grid_size=32)
    assert uniform["passed"]
    assert uniform["renormalization"] == "none"
    json.dumps(uniform, allow_nan=False)
    heat = evaluate_model(basis, coefficient, benchmark("heat1d"), grid_size=16, times=[0, 1], max_grid_size=64)
    assert not heat["passed"]
    assert not heat["checks"]["tv"]


def test_reference_cap_cannot_pass_without_refinement():
    basis = Basis(K=1, p=1)
    result = evaluate_model(basis, np.zeros(basis.shape), benchmark("uniform"), grid_size=16, times=[0, 1], max_grid_size=16)
    assert not result["passed"]
    assert not result["checks"]["reference_refinement"]


def test_sde_baseline_reports_sampling_not_density_certificate():
    result = sde_baseline(benchmark("uniform"), samples=64, steps=8, times=[0, .5, 1])
    assert result["kind"] == "empirical_sde_observables"
    assert len(result["records"]) == 3
    assert result["records"][-1]["step_refinement_max"] < 1e-14
    json.dumps(result, allow_nan=False)
