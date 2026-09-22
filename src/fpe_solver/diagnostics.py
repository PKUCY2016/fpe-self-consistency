"""Small-sample geometric consistency diagnostics, never reference-solution metrics."""
import jax
import jax.numpy as jnp
import numpy as np

from .flow import integrate


def check_score(basis, coeff, problem, x0, steps, tolerance=1e-4):
    """Compare the score ODE with independently differentiated change of variables.

    For the discrete flow F, independently compute log rho(F(y)) as
    log rho0(y) - log det DF(y), then pull its y-gradient to spatial coordinates.
    The two quantities need only agree to discretization accuracy. The check
    rejects a discrete map that loses orientation or produces nonfinite values.
    Use a small batch (typically 4--8); second flow derivatives are expensive.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    if x0.ndim != 2 or not len(x0):
        raise ValueError("check_score requires nonempty (batch, dim) input")
    d = basis.dim

    def state(y):
        return integrate(basis, coeff, problem, y[None], steps)[0]

    def endpoint(y):
        return state(y)[:d]

    flow_jacobian = jax.jacfwd(endpoint)

    def transformed_log_density(y):
        return problem.log_density0(y) - jnp.linalg.slogdet(flow_jacobian(y))[1]

    def check_one(y):
        z = state(y)
        jac = flow_jacobian(y)
        independent_score = jnp.linalg.solve(jac.T, jax.grad(transformed_log_density)(y))
        return (z[d+1:2*d+1], independent_score, jnp.linalg.det(jac),
                z[d] - transformed_log_density(y), jnp.linalg.cond(jac),
                jnp.all(jnp.isfinite(z)))

    actual, expected, determinants, density_error, conditions, state_finite = jax.vmap(check_one)(x0)
    actual, expected, determinants, density_error, conditions = map(
        np.asarray, (actual, expected, determinants, density_error, conditions))
    finite = all(np.all(np.isfinite(a)) for a in
                 (actual, expected, determinants, density_error, conditions)) and bool(np.all(state_finite))
    rms = float(np.sqrt(np.mean((actual - expected)**2)) / max(1., np.sqrt(np.mean(expected**2))))
    log_rms = float(np.sqrt(np.mean(density_error**2)))
    min_det = float(np.min(determinants))
    return {"passed": bool(finite and min_det > 0 and rms <= tolerance and log_rms <= tolerance),
                "normalized_rms": rms, "finite": bool(finite), "min_det": min_det,
                "log_density_rms": log_rms, "max_condition": float(np.max(conditions)),
                "samples": len(x0), "tolerance": tolerance}


def high_order_residual(basis, coeff, problem, x0, steps):
    """Empirical endpoint Sobolev terms with correct Eulerian derivatives.

    Derivatives first follow the initial labels y, including transported score
    dependence; multiplication by (DF)^{-1} converts them to x derivatives.
    The second derivative differentiates that entire converted first derivative.
    This diagnostic is intentionally limited to eight points and is not a
    time-integrated Sobolev loss or a certified bound.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    if x0.ndim != 2 or not 0 < len(x0) <= 8:
        raise ValueError("Use between one and eight diagnostic initial points")
    d = basis.dim

    def state(y):
        return integrate(basis, coeff, problem, y[None], steps)[0]

    def endpoint(y):
        return state(y)[:d]

    def residual(y):
        z = state(y)
        return basis.velocity(coeff, basis.T, z[:d]) + problem.grad_potential(z[:d]) + z[d+1:2*d+1]

    def gradient(y):
        return jax.jacfwd(residual)(y) @ jnp.linalg.inv(jax.jacfwd(endpoint)(y))

    def values(y):
        jac = jax.jacfwd(endpoint)(y)
        hessian = jnp.einsum("ijk,kl->ijl", jax.jacfwd(gradient)(y), jnp.linalg.inv(jac))
        return residual(y), gradient(y), hessian, jnp.linalg.det(jac)

    residuals, gradients, hessians, determinants = map(np.asarray, jax.vmap(values)(x0))
    finite = all(np.all(np.isfinite(a)) for a in (residuals, gradients, hessians, determinants))
    return {"finite": bool(finite), "orientation_positive": bool(np.all(determinants > 0)),
                "residual_mean_square": float(np.mean(np.sum(residuals**2, axis=-1))),
                "gradient_mean_square": float(np.mean(np.sum(gradients**2, axis=(-2, -1)))),
                "hessian_mean_square": float(np.mean(np.sum(hessians**2, axis=(-3, -2, -1)))),
                "samples": len(x0), "time": basis.T, "scope": "endpoint empirical diagnostic only"}
