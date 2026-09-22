"""Regression tests for candidate isolation and independent numerical safeguards."""
import ast
import inspect
import json
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from fpe_solver import controller, runner, storage, training
from fpe_solver.basis import Basis
from fpe_solver.config import SolverConfig
from fpe_solver.problems import Problem


def tiny_config(**overrides):
    return SolverConfig(**({"K": 1, "p": 1, "batch": 4, "steps": 4, "updates": 1,
                               "validation_samples": 4, "max_validation_samples": 4,
                               "max_rounds": 2, "max_seconds": 30} | overrides))


def successful_validation(*args, **kwargs):
    return {"promote": True, "numerical_consistency": True,
            "score_consistency": {"passed": True},
            "incumbent_score_consistency": {"passed": True}, "reason": "IMPROVED"}


def fake_training(basis, coeff, problem, config, key, *, opt_state=None,
                  start_update=0, deadline=None, checkpoint=None):
    checkpoint(coeff, opt_state, key, config.updates, [{"update": config.updates, "loss": 0.1}])
    return {"coeff": coeff, "status": "TRAINED", "seconds": 0.001,
            "first_step_compile_and_execution_seconds": 0.001}


def test_time_probe_ignores_old_representable_directions(monkeypatch):
    """A gradient wholly in the old polynomial span must not imply new capacity."""
    config, problem = tiny_config(), Problem(amplitude=0)
    basis = Basis(problem.dim, config.K, config.p, problem.T)
    monkeypatch.setattr(controller, "residual_samples", lambda b, c, p, x, s: np.ones(len(x)))
    monkeypatch.setattr(controller.jax, "jit", lambda f: f)

    def fake_grad(fun):
        expanded = fun.__defaults__[0]
        gradient = np.zeros(expanded.shape)
        gradient[:, 0, 0] = 1  # The constant Bernstein polynomial belongs to old p=1.
        return lambda c: gradient

    monkeypatch.setattr(controller.jax, "grad", fake_grad)
    diagnostic = controller.diagnose(basis, basis.zeros(), config, problem, 0, 0, [])
    assert diagnostic["probe_gradient_rms"]["increase_space"] == 0
    assert diagnostic["probe_gradient_rms"]["increase_time"] < 1e-14


def test_time_probe_detects_new_temporal_direction(monkeypatch):
    config, problem = tiny_config(), Problem(amplitude=0)
    basis = Basis(problem.dim, config.K, config.p, problem.T)
    monkeypatch.setattr(controller, "residual_samples", lambda b, c, p, x, s: np.ones(len(x)))
    monkeypatch.setattr(controller.jax, "jit", lambda f: f)

    def fake_grad(fun):
        expanded = fun.__defaults__[0]
        gradient = np.zeros(expanded.shape)
        if expanded.p == 3:
            gradient[:, 0, 0] = [1, -3, 3, -1]  # Orthogonal to elevated linear polynomials.
        return lambda c: gradient

    monkeypatch.setattr(controller.jax, "grad", fake_grad)
    diagnostic = controller.diagnose(basis, basis.zeros(), config, problem, 0, 0, [])
    assert diagnostic["probe_gradient_rms"]["increase_space"] == 0
    assert diagnostic["probe_gradient_rms"]["increase_time"] > 0.1


def test_numerical_errors_cannot_cancel_in_validation(monkeypatch):
    config, problem = tiny_config(), Problem(amplitude=0)
    basis = Basis(problem.dim, config.K, config.p, problem.T)
    old, new = basis.zeros(), basis.zeros() + 1

    def residual(b, coefficients, p, x, steps):
        mean = 1.0 if np.asarray(coefficients).max() == 0 else .9
        result = np.full(len(x), mean)
        if steps == config.steps:
            result += np.resize([.1, -.1], len(x))
        return result

    monkeypatch.setattr(controller, "residual_samples", residual)
    result = controller.validate_candidate(basis, old, config, basis, new, config, problem, 0, 0)
    assert not result["promote"]
    assert result["reason"] == "REFINE_INTEGRATION"
    assert result["numerical_change_old"] == pytest.approx(.1)
    assert result["numerical_change_new"] == pytest.approx(.1)


def test_numerical_margin_blocks_apparent_improvement():
    result = controller.paired_gate(np.ones(4), np.full(4, .94), numerical_margin=.04)
    assert not result["promote"]
    assert result["required_improvement"] >= .08


def test_unqualified_initial_candidate_never_becomes_champion(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "environment", dict)
    monkeypatch.setattr(runner, "train_candidate", fake_training)
    monkeypatch.setattr(runner, "validate_candidate", lambda *a, **k: {
        "promote": False, "numerical_consistency": True,
        "score_consistency": {"passed": False}, "reason": "SCORE_INCONSISTENCY"})
    path = tmp_path / "bad-initial"
    runner.create_run(path, Problem(), tiny_config(), 0, "train")
    state = runner.execute(path)
    assert state["status"] == "NUMERICAL_FAILURE"
    assert state["champion"] is None
    assert not state["events"][0]["promoted"]
    with pytest.raises(ValueError, match="champion"):
        runner.get_model(path)


def test_resume_initial_candidate_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "environment", dict)
    monkeypatch.setattr(runner, "train_candidate", fake_training)
    monkeypatch.setattr(runner, "validate_candidate", successful_validation)
    problem, config = Problem(), tiny_config()
    basis = Basis(problem.dim, config.K, config.p, problem.T)
    path = tmp_path / "resume-initial"
    state = runner.create_run(path, problem, config, 7, "train")
    state["status"] = "INTERRUPTED"
    state["candidate"] = {"round": 0, "action": "initial", "basis": basis.to_dict(),
                          "config": config.to_dict(), "update": 0, "history": [],
                          "checkpoint": None, "parent": None}
    storage.write_json(path / "state.json", state)
    result = runner.execute(path, resume=True)
    assert result["status"] == "TRAINED"
    assert result["champion"]["qualified"]
    assert result["champion"]["checkpoint"]
    assert len(result["events"]) == 1


def test_resume_expanded_candidate_reconstructs_incumbent(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "environment", dict)
    monkeypatch.setattr(runner, "validate_candidate", successful_validation)
    problem, config = Problem(), tiny_config()
    basis = Basis(problem.dim, config.K, config.p, problem.T)
    coefficients = jnp.arange(np.prod(basis.shape)).reshape(basis.shape) / 100
    new_config = replace(config, K=2)
    expanded = Basis(problem.dim, 2, config.p, problem.T)
    path = tmp_path / "resume-expanded"
    state = runner.create_run(path, problem, config, 11, "train")
    filename = "checkpoints/old.npz"
    key = controller.sample_key(11, 0, 100)
    digest = storage.save_arrays(path / filename, coefficients, key, training.optimizer(config).init(coefficients))
    state.update(status="INTERRUPTED", next_round=1)
    state["champion"] = {"basis": basis.to_dict(), "config": config.to_dict(), "qualified": True,
                         "checkpoint": {"file": filename, "sha256": digest}, "history": []}
    state["candidate"] = {"round": 1, "action": "increase_space", "basis": expanded.to_dict(),
                          "config": new_config.to_dict(), "update": 0, "history": [],
                          "checkpoint": None, "parent": digest}
    storage.write_json(path / "state.json", state)
    calls = []

    def train(b, c, p, cfg, rng, **kwargs):
        calls.append(1)
        np.testing.assert_allclose(c, basis.prolong(coefficients, expanded), atol=0)
        np.testing.assert_array_equal(rng, controller.sample_key(11, 1, 100))
        return fake_training(b, c, p, cfg, rng, **kwargs)

    monkeypatch.setattr(runner, "train_candidate", train)
    result = runner.execute(path, resume=True)
    assert result["status"] == "TRAINED"
    assert result["champion"]["parent"] == digest
    assert len(calls) == 1
    terminal_state = runner.execute(path, resume=True)
    assert terminal_state == result
    assert len(calls) == 1  # Resume never resubmits completed work.


def test_nonfinite_report_is_json_with_failure_evidence(tmp_path):
    path = tmp_path / "failed.json"
    storage.write_json(path, {"reason": "NONFINITE", "passed": False,
                              "diagnostics": np.array([np.nan, np.inf, -np.inf]),
                              "nested": {"loss": float("nan")}})
    raw = path.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    result = json.loads(raw)
    assert result["diagnostics"] == [None, None, None]
    assert result["nested"]["loss"] is None
    assert result["reason"] == "NONFINITE" and not result["passed"]


def test_single_writer_lock_rejects_duplicate_execution(tmp_path):
    with (storage.run_lock(tmp_path),
          pytest.raises(RuntimeError, match="active writer"), storage.run_lock(tmp_path)):
        pytest.fail("second writer entered")


def test_training_and_control_cannot_import_reference_answers():
    for module in (controller, runner, training):
        tree = ast.parse(inspect.getsource(module))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Import):
                imports.extend(name.name for name in node.names)
        assert all(name.split(".")[-1] not in {"reference", "evaluation"} for name in imports)


def test_intermediate_time_sampling_preserves_bernstein_horizon(monkeypatch):
    from fpe_solver.api import ProbabilityFlow

    problem = Problem(T=1., amplitude=0.)
    basis = Basis(dim=1, K=1, p=1, T=1.)
    coefficients = np.zeros(basis.shape)
    coefficients[1, 0, 0] = 2.  # v(t)=2t/T; endpoint displacement is t**2/T.
    monkeypatch.setattr(Problem, "sample", lambda self, key, n: jnp.full((n, 1), .2))
    flow = ProbabilityFlow(basis, coefficients, problem, steps=8)
    np.testing.assert_allclose(flow.sample(3, t=.4), .36, atol=1e-12)


@pytest.mark.parametrize("end_time", [-.1, 1.1, float("nan"), float("inf")])
def test_intermediate_time_rejects_invalid_queries(end_time):
    from fpe_solver.flow import integrate

    basis, problem = Basis(K=1, p=1), Problem()
    with pytest.raises(ValueError, match="end_time"):
        integrate(basis, basis.zeros(), problem, jnp.zeros((1, 1)), 2, end_time=end_time)


def test_loading_preserves_trained_integrator_resolution(monkeypatch):
    from fpe_solver import api

    basis, problem = Basis(K=1, p=1), Problem()
    monkeypatch.setattr(api, "get_model", lambda _: (basis, basis.zeros(), problem))
    monkeypatch.setattr(api, "read_json", lambda _: {"champion": {"config": {"steps": 1024}}})
    assert api.ProbabilityFlow.load("unused").steps == 1024
    assert api.ProbabilityFlow.load("unused", steps=16).steps == 16


def test_round_budget_includes_initial_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "environment", dict)
    monkeypatch.setattr(runner, "train_candidate", fake_training)
    monkeypatch.setattr(runner, "validate_candidate", successful_validation)
    monkeypatch.setattr(runner, "diagnose", lambda *a: pytest.fail("second round exceeded cap"))
    path = tmp_path / "one-round"
    runner.create_run(path, Problem(), tiny_config(max_rounds=1), 0, "evolve")
    state = runner.execute(path)
    assert len(state["events"]) == 1
    assert state["status"] == "BUDGET_EXHAUSTED"


def test_resume_refuses_changed_source_and_records_difference(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "environment", lambda: {"source_hashes": {"training.py": "old"}})
    path = tmp_path / "source-change"
    state = runner.create_run(path, Problem(), tiny_config(), 0, "train")
    state["status"] = "INTERRUPTED"
    storage.write_json(path / "state.json", state)
    monkeypatch.setattr(runner, "environment", lambda: {"source_hashes": {"training.py": "changed"}})
    monkeypatch.setattr(runner, "train_candidate", lambda *a, **k: pytest.fail("changed code executed"))
    with pytest.raises(ValueError, match="environment changed"):
        runner.execute(path, resume=True)
    persisted = storage.read_json(path / "state.json")
    assert persisted["resume_environments"][-1]["differences"]["source_hashes"]["current"] == {"training.py": "changed"}


def test_validation_batch_needs_two_independent_samples():
    with pytest.raises(ValueError):
        tiny_config(validation_samples=1)


def test_image_affine_base_matches_exact_single_gaussian_ou():
    import jax

    from fpe_solver.image_experiment import GaussianMixture, init_parameters, transport
    from fpe_solver.image_experiment import integrate as image_integrate

    center = np.array([[.2, -.3, .1]])
    mixture = GaussianMixture(center, sigma=.4)
    params = init_parameters(jax.random.PRNGKey(9), dim=3, rank=2)
    initial = jnp.asarray(center + np.array([[.1, .2, -.1], [-.2, .3, .1]]))
    horizon = .5
    x, score, log_density, residual = image_integrate(params, mixture, initial, steps=128, T=horizon)
    variance = 1 - (1 - mixture.sigma**2)*np.exp(-2*horizon)
    mean = np.exp(-horizon)*center
    expected = mean + np.sqrt(variance/mixture.sigma**2)*(np.asarray(initial)-center)
    np.testing.assert_allclose(x, expected, atol=2e-9)
    np.testing.assert_allclose(score, -(expected-mean)/variance, atol=2e-9)
    exact_log = -.5*(3*np.log(2*np.pi*variance) + ((expected-mean)**2).sum(-1)/variance)
    np.testing.assert_allclose(log_density, exact_log, atol=2e-9)
    assert np.max(np.asarray(residual)) < 1e-12
    reverse = transport(params, x, mean=mixture.mean, base_variance=mixture.sigma**2,
                        T=horizon, steps=128, reverse=True)
    np.testing.assert_allclose(reverse, initial, atol=2e-9)


def test_image_comoving_flow_matches_independent_physical_ode():
    import math

    import jax
    from scipy.integrate import solve_ivp
    from scipy.special import logsumexp

    from fpe_solver.image_experiment import GaussianMixture, init_parameters
    from fpe_solver.image_experiment import integrate as image_integrate

    centers = np.array([[.2, -.1, .3], [-.3, .15, -.2]])
    mixture = GaussianMixture(centers, sigma=.4)
    params = init_parameters(jax.random.PRNGKey(4), dim=3, rank=2)
    rng = np.random.default_rng(12)
    params = {name: jnp.asarray(rng.normal(size=value.shape)*.05) for name, value in params.items()}
    frozen = {name: np.asarray(value) for name, value in params.items()}
    initial = centers + np.array([[.1, .05, -.1], [-.05, -.1, .08]])
    horizon, dim, count = .2, 3, 2
    squared = np.sum((initial[:, None, :]-centers[None, :, :])**2, axis=-1)
    logits = -squared/(2*mixture.sigma**2)
    normalization = logsumexp(logits, axis=1)
    log0 = normalization - math.log(len(centers)) - .5*dim*math.log(2*math.pi*mixture.sigma**2)
    score0 = (np.exp(logits-normalization[:, None])@centers-initial)/mixture.sigma**2
    state0 = np.concatenate((initial.ravel(), score0.ravel(), log0, np.zeros(count)))

    def physical_rhs(t, flat):
        x = flat[:count*dim].reshape(count, dim)
        score = flat[count*dim:2*count*dim].reshape(count, dim)
        degree = frozen["B"].shape[0]-1
        u = t/horizon
        weights = np.array([math.comb(degree, k)*u**k*(1-u)**(degree-k) for k in range(degree+1)])
        B = np.tensordot(weights, frozen["B"], axes=(0, 0))
        hidden = np.tanh(x@frozen["A"].T + weights@frozen["b"])
        derivative = 1-hidden**2
        q = 1 - (1-mixture.sigma**2)*np.exp(-2*t)
        a = 1/q-1
        velocity = a*x-np.exp(-t)*centers.mean(0)/q + hidden@B.T + weights@frozen["c"]
        projection = np.sum(frozen["A"]*B.T, axis=1)
        divergence = dim*a+derivative@projection
        jacobian_score = a*score+((score@B)*derivative)@frozen["A"]
        gradient_divergence = (-2*hidden*derivative*projection)@frozen["A"]
        return np.concatenate((velocity.ravel(), (-jacobian_score-gradient_divergence).ravel(),
                               -divergence, np.mean((velocity+x+score)**2, axis=1)))

    independent = solve_ivp(physical_rhs, (0, horizon), state0, method="DOP853", rtol=1e-12, atol=1e-13)
    assert independent.success
    actual = image_integrate(params, mixture, jnp.asarray(initial), steps=64, T=horizon)
    actual_flat = np.concatenate([np.asarray(value).ravel() for value in actual])
    np.testing.assert_allclose(actual_flat, independent.y[:, -1], atol=1e-8, rtol=1e-8)
