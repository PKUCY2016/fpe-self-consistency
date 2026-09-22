"""Synthetic dimension-eight checks; never download or train on a real image dataset."""
import jax
import jax.numpy as jnp
import numpy as np

from fpe_solver.image_experiment import (
    GaussianMixture,
    exact_terminal_log_density_numpy,
    init_parameters,
    integrate,
    lowrank_terms,
    nearest_neighbors,
    train_image_model,
    transport,
    velocity,
)


def test_mixture_score_matches_log_density_gradient():
    rng = np.random.default_rng(19)
    mixture = GaussianMixture(rng.normal(size=(5, 8)), .4)
    x = jnp.asarray(rng.normal(size=(3, 8)))
    for t in (0., .7):
        log_density, score = mixture.log_density_score(x, t)
        automatic = jax.vmap(jax.grad(lambda y, time=t: mixture.log_density_score(y, time)[0]))(x)
        np.testing.assert_allclose(score, automatic, atol=2e-13)
        oracle = exact_terminal_log_density_numpy(mixture, np.asarray(x), t)
        np.testing.assert_allclose(log_density, oracle, atol=2e-13)


def test_lowrank_exact_derivatives_match_full_ad():
    rng = np.random.default_rng(23)
    params = init_parameters(jax.random.PRNGKey(0), 8, rank=3, degree=3)
    params = {k: jnp.asarray(rng.normal(scale=.2, size=v.shape)) for k, v in params.items()}
    x, score, mean = [jnp.asarray(rng.normal(size=8)) for _ in range(3)]
    field = lambda y: velocity(params, .37, y, mean, .04, 3.)
    v, divergence, jts, grad_divergence = lowrank_terms(params, .37, x, score, mean, .04, 3.)
    jac = jax.jacfwd(field)(x)
    np.testing.assert_allclose(v, field(x), atol=1e-13)
    np.testing.assert_allclose(divergence, jnp.trace(jac), atol=1e-13)
    np.testing.assert_allclose(jts, jac.T@score, atol=1e-13)
    reference = jax.grad(lambda y: jnp.trace(jax.jacfwd(field)(y)))(x)
    np.testing.assert_allclose(grad_divergence, reference, atol=1e-13)


def test_single_gaussian_base_and_discrete_gradient():
    mean = np.linspace(-.2, .3, 8)
    mixture = GaussianMixture(mean[None], .5)
    params = init_parameters(jax.random.PRNGKey(2), 8, rank=2)
    x = mixture.sample(jax.random.PRNGKey(3), 3)
    endpoint, score, _, residual = integrate(params, mixture, x, steps=64, T=.5)
    q = 1+(.25-1)*np.exp(-1.)
    exact = np.exp(-.5)*mean + np.sqrt(q/.25)*(np.asarray(x)-mean)
    np.testing.assert_allclose(endpoint, exact, atol=2e-8)
    np.testing.assert_allclose(score, -(endpoint-np.exp(-.5)*mean)/q, atol=2e-8)
    assert float(jnp.max(residual)) < 1e-14
    direction = jax.tree_util.tree_map(lambda p: jnp.ones_like(p)*.1, params)
    objective = lambda p: jnp.mean(integrate(p, mixture, x, steps=4, T=.15)[-1])
    perturbed = {k: v+.01*direction[k] for k, v in params.items()}
    grad = jax.grad(objective)(perturbed)
    expected = sum(jnp.vdot(grad[k], direction[k]) for k in grad)
    h = 1e-5
    plus = {k: v+h*direction[k] for k, v in perturbed.items()}
    minus = {k: v-h*direction[k] for k, v in perturbed.items()}
    difference = (objective(plus)-objective(minus))/(2*h)
    np.testing.assert_allclose(expected, difference, atol=2e-8, rtol=1e-5)


def test_inverse_refinement_and_nearest_neighbor_metric():
    mixture = GaussianMixture(np.zeros((1, 8)), .5)
    params = init_parameters(jax.random.PRNGKey(4), 8, rank=2)
    noise = jax.random.normal(jax.random.PRNGKey(5), (3, 8), dtype=jnp.float64)
    q = 1+(.25-1)*np.exp(-2.)
    exact = np.sqrt(.25/q)*noise
    coarse = transport(params, noise, mean=mixture.mean, base_variance=.25, T=1., steps=16)
    fine = transport(params, noise, mean=mixture.mean, base_variance=.25, T=1., steps=64)
    np.testing.assert_allclose(coarse, exact, atol=1e-13)
    np.testing.assert_allclose(fine, exact, atol=1e-13)
    params["B"] = jnp.ones_like(params["B"])*.15
    coarse = transport(params, noise, mean=mixture.mean, base_variance=.25, T=1., steps=8)
    fine = transport(params, noise, mean=mixture.mean, base_variance=.25, T=1., steps=32)
    reference = transport(params, noise, mean=mixture.mean, base_variance=.25, T=1., steps=128)
    assert float(jnp.max(jnp.abs(fine-reference))) < float(jnp.max(jnp.abs(coarse-reference)))/10
    refs = np.asarray([[0., 0.], [1., 1.], [3., 3.]])
    index, rmse = nearest_neighbors(np.asarray([[.1, .1], [2.9, 2.9]]), refs)
    np.testing.assert_array_equal(index, [0, 2])
    np.testing.assert_allclose(rmse, [.1, .1])


def test_synthetic_training_executes_and_reports_raw_residual():
    mixture = GaussianMixture(np.asarray([np.ones(8)*.2, -np.ones(8)*.2]), .5)
    params, report = train_image_model(mixture, seed=7, rank=2, batch=2, updates=2, steps=4, T=.1)
    assert report["status"] == "UPDATES_COMPLETED"
    assert report["updates_accepted"] == 2
    assert all(np.isfinite(v).all() for v in params.values())
    for entry in report["history"]:
        assert entry["accepted_finite_update"]
        assert entry["loss_raw_dimension_sum"] == entry["loss_per_coordinate"]*8
