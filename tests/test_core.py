import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fpe_solver.basis import Basis
from fpe_solver.diagnostics import check_score, high_order_residual
from fpe_solver.flow import integrate
from fpe_solver.problems import Problem


@pytest.mark.parametrize("dim", [1, 2])
def test_analytic_derivatives_and_periodicity(dim):
    basis = Basis(dim=dim, K=2, p=3, T=.7)
    rng = np.random.default_rng(3)
    coeff = jnp.asarray(rng.normal(scale=.1, size=basis.shape))
    x = jnp.asarray(rng.normal(size=dim))
    t = .31
    v, jac, div, grad_div = basis.spatial(coeff, t, x)
    field = lambda y: basis.velocity(coeff, t, y)
    np.testing.assert_allclose(jac, jax.jacfwd(field)(x), atol=2e-14)
    np.testing.assert_allclose(div, jnp.trace(jax.jacfwd(field)(x)), atol=2e-14)
    expected = jax.grad(lambda y: jnp.trace(jax.jacfwd(field)(y)))(x)
    np.testing.assert_allclose(grad_div, expected, atol=2e-14)
    np.testing.assert_allclose(v, field(x + 2*np.pi*jnp.arange(1, dim+1)), atol=2e-14)
    assert basis.nfeatures == (2*basis.K+1)**dim
    assert len({tuple(k) for k in basis.modes}) == len(basis.modes)
    if dim == 2:
        assert (1, -1) in map(tuple, basis.modes)
    batch = jnp.stack((x, 2*x, 3*x)).reshape(1, 3, dim)
    assert basis.velocity(coeff, t, batch).shape == (1, 3, dim)


@pytest.mark.parametrize("dim", [1, 2])
def test_prolongation_preserves_field_and_derivatives(dim):
    old, new = Basis(dim, 1, 2, .8), Basis(dim, 3, 6, .8)
    coeff = jnp.asarray(np.random.default_rng(7).normal(size=old.shape))
    elevated = old.prolong(coeff, new)
    x = jnp.asarray(np.random.default_rng(8).normal(size=(5, dim)))
    for t in [0., .13, .51, .8]:
        for a, b in zip(old.spatial(coeff, t, x), new.spatial(elevated, t, x)):
            np.testing.assert_allclose(a, b, atol=4e-14)
    with pytest.raises(ValueError):
        new.prolong(elevated, old)


def test_zero_and_constant_modes():
    basis = Basis(2, 0, 0, .4)
    problem = Problem(dim=2, T=.4, amplitude=0.)
    x0 = jnp.asarray([[.2, 2.1], [3.4, 5.5]])
    history = integrate(basis, basis.zeros(), problem, x0, 4, save=True)
    assert history.shape == (5, 2, 6)
    np.testing.assert_allclose(history[-1, :, :2], x0)
    np.testing.assert_allclose(history[..., -1], 0)
    np.testing.assert_allclose(history[..., 2], -2*np.log(2*np.pi))
    coeff = basis.zeros().at[0, 0].set(jnp.array([1., -2.]))
    final = integrate(basis, coeff, problem, x0, 4)
    np.testing.assert_allclose(final[:, :2], x0 + .4*jnp.array([1., -2.]))
    np.testing.assert_allclose(final[:, -1], .4*5)


def test_discrete_directional_gradient():
    problem = Problem(T=.3, amplitude=.35, kappa=.2)
    basis = Basis(1, 2, 2, problem.T)
    rng = np.random.default_rng(17)
    coeff = jnp.asarray(rng.normal(scale=.03, size=basis.shape))
    direction = jnp.asarray(rng.normal(size=basis.shape))
    direction /= jnp.linalg.norm(direction)
    x0 = jnp.asarray([[.2], [1.3], [4.1]])
    loss = lambda c: jnp.mean(integrate(basis, c, problem, x0, 12)[:, -1])
    automatic = jnp.vdot(jax.grad(loss)(coeff), direction)
    h = 1e-5
    difference = (loss(coeff + h*direction)-loss(coeff - h*direction))/(2*h)
    assert abs(float(automatic-difference))/max(1., abs(float(difference))) < 1e-7


@pytest.mark.parametrize("dim", [1, 2])
def test_score_from_change_of_variables(dim):
    problem = Problem(dim=dim, T=.2, amplitude=.3)
    basis = Basis(dim, 1, 2, problem.T)
    coeff = jnp.asarray(np.random.default_rng(21).normal(scale=.04, size=basis.shape))
    x0 = jnp.asarray(np.random.default_rng(24).uniform(0, 2*np.pi, (2, dim)))
    result = check_score(basis, coeff, problem, x0, 8)
    assert result["passed"], result
    assert result["normalized_rms"] < 1e-7
    assert result["min_det"] > 0


def test_sobolev_diagnostic_differentiates_the_score():
    problem = Problem(T=.1, amplitude=.4)
    basis = Basis(1, 0, 0, problem.T)
    x0 = jnp.asarray([[.3], [1.1]])
    result = high_order_residual(basis, basis.zeros(), problem, x0, 2)
    first = jax.vmap(jax.jacfwd(problem.score0))(x0)
    second = jax.vmap(jax.jacfwd(jax.jacfwd(problem.score0)))(x0)
    np.testing.assert_allclose(result["gradient_mean_square"], jnp.mean(first**2), rtol=1e-12)
    np.testing.assert_allclose(result["hessian_mean_square"], jnp.mean(second**2), rtol=1e-12)
    assert result["gradient_mean_square"] > 0
