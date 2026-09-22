"""Residual-only training. Reference solutions and evaluation metrics are not imported."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .flow import integrate


def optimizer(config):
    return optax.chain(optax.clip_by_global_norm(config.clip_norm), optax.adam(config.learning_rate))


def train_candidate(basis, coeff, problem, config, key, *, opt_state=None, start_update=0,
                    deadline=None, checkpoint=None, compiled_update=None):
    tx = optimizer(config)
    opt_state = tx.init(coeff) if opt_state is None else opt_state

    @jax.jit
    def update(c, state, rng):
        rng, sampling_key = jax.random.split(rng)
        x0 = problem.sample(sampling_key, config.batch)
        def loss(cc):
            return jnp.mean(integrate(basis, cc, problem, x0, config.steps)[:, -1])
        val, g = jax.value_and_grad(loss)(c)
        delta, state = tx.update(g, state, c)
        return optax.apply_updates(c, delta), state, rng, val, optax.global_norm(g)

    update = update if compiled_update is None else compiled_update
    start = time.monotonic()
    history = []
    compile_seconds = 0.
    completed = start_update
    status = "TRAINED"
    for i in range(start_update, config.updates):
        if deadline is not None and time.monotonic() >= deadline:
            status = "BUDGET_EXHAUSTED"
            break
        step_start = time.monotonic()
        candidate, next_opt, next_key, value, norm = update(coeff, opt_state, key)
        value.block_until_ready()
        if i == start_update:
            compile_seconds = time.monotonic() - step_start
        if not (np.isfinite(float(value)) and np.isfinite(float(norm)) and
                np.all(np.isfinite(np.asarray(candidate)))):
            status = "NUMERICAL_FAILURE"
            break
        coeff, opt_state, key = candidate, next_opt, next_key
        completed = i+1
        history.append({"update": completed, "loss": float(value), "gradient_norm": float(norm)})
        if checkpoint is not None and (completed % 25 == 0 or completed == config.updates):
            checkpoint(coeff, opt_state, key, completed, history[-25:])
    if checkpoint is not None:
        checkpoint(coeff, opt_state, key, completed, history[-25:])
    return {"coeff": coeff, "opt_state": opt_state, "key": key, "update": completed,
            "status": status, "history": history, "seconds": time.monotonic()-start,
            "first_step_compile_and_execution_seconds": compile_seconds, "compiled_update": update}


def residual_samples(basis, coeff, problem, x0, steps, chunk=128):
    fn = jax.jit(lambda x: integrate(basis, coeff, problem, x, steps)[:, -1])
    pieces = []
    for offset in range(0, len(x0), chunk):
        pieces.append(np.asarray(fn(jnp.asarray(x0[offset:offset+chunk]))))
    return np.concatenate(pieces)
