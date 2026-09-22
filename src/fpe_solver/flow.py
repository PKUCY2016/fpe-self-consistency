"""Differentiable, unwrapped RK4 probability-flow integration.

State order is position[d], log-density[1], score[d], integrated residual[1].
The score always comes from the given initial density and is transported by
the same velocity as the particles. No independently fitted score is used.
"""
import math

import jax
import jax.numpy as jnp


def augmented_rhs(basis, coeff, problem, t, state):
    d = basis.dim
    x = state[..., :d]
    score = state[..., d+1:2*d+1]
    velocity, jacobian, divergence, grad_divergence = basis.spatial(coeff, t, x)
    score_rate = -jnp.einsum("...oi,...o->...i", jacobian, score) - grad_divergence
    residual = velocity + problem.grad_potential(x) + score
    return jnp.concatenate((velocity, -divergence[..., None], score_rate,
                            jnp.sum(residual**2, axis=-1, keepdims=True)), axis=-1)


def integrate(basis, coeff, problem, x0, steps, save=False, *, end_time=None):
    """Integrate to end_time (default problem.T); save=True includes initial state.

    The original basis.T is retained when querying an intermediate time, so
    Bernstein time normalization and the represented velocity do not change.

    For x0 of shape (batch, dim), return (batch, 2*dim+2), or
    (steps+1, batch, 2*dim+2) when saving. Derivatives are through the actual
    RK4 discretization; shrinking the step size is a separate validation.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if problem.dim != basis.dim or problem.T != basis.T:
        raise ValueError("Problem and basis dimensions and final times must agree")
    final_time = basis.T if end_time is None else float(end_time)
    if not math.isfinite(final_time) or not 0 <= final_time <= basis.T:
        raise ValueError("end_time must be finite and lie in [0, T]")
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    if x0.ndim != 2 or x0.shape[1] != basis.dim:
        raise ValueError("x0 must have shape (batch, dim)")
    initial = jnp.concatenate((x0, problem.log_density0(x0)[..., None], problem.score0(x0),
                               jnp.zeros(x0.shape[:-1] + (1,), dtype=jnp.float64)), axis=-1)
    h = final_time / steps

    def step(state, index):
        t = index * h
        k1 = augmented_rhs(basis, coeff, problem, t, state)
        k2 = augmented_rhs(basis, coeff, problem, t + h/2, state + h*k1/2)
        k3 = augmented_rhs(basis, coeff, problem, t + h/2, state + h*k2/2)
        k4 = augmented_rhs(basis, coeff, problem, t + h, state + h*k3)
        result = state + h*(k1 + 2*k2 + 2*k3 + k4)/6
        return result, result if save else None

    final, history = jax.lax.scan(step, initial, jnp.arange(steps))
    if save:
        return jnp.concatenate((initial[None], history), axis=0)
    return final
