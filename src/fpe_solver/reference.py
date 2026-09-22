"""Independent NumPy/SciPy reference densities (never imported by the controller).

The finite-volume unknown is a *cell average*. Columns of the generator sum to
zero, off-diagonal entries are nonnegative, and its exact stationary vector is
proportional to exp(-V) at cell centres. This is a spatial approximation, not a
certified solution of the PDE; callers must check refinement.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply


def grid_points(dim: int, n: int) -> np.ndarray:
    if dim not in (1, 2) or n < 2:
        raise ValueError("Require dimension 1 or 2 and at least two periodic cells")
    axis = (np.arange(n) + 0.5) * (2 * np.pi / n)
    return np.stack(np.meshgrid(*([axis] * dim), indexing="ij"), axis=-1).reshape(-1, dim)


def initial_log_density(problem, x):
    x = np.asarray(x, dtype=float)
    return np.log1p(problem.amplitude * np.cos(problem.frequency * (x - problem.phase))).sum(-1) - problem.dim * math.log(2 * math.pi)


def potential(problem, x):
    value = problem.kappa * np.sum(1 - np.cos(2 * x), axis=-1)
    if problem.dim == 2:
        value = value + problem.coupling * (1 - np.cos(x[..., 0] - x[..., 1]))
    return value


def potential_gradient(problem, x):
    result = 2 * problem.kappa * np.sin(2 * x)
    if problem.dim == 2:
        c = problem.coupling * np.sin(x[..., 0] - x[..., 1])
        result = result + np.stack((c, -c), axis=-1)
    return result


def analytic_heat_density(problem, x, t: float, cell_width: float = 0.):
    if problem.kappa or problem.coupling:
        raise ValueError("Analytic heat reference requires zero potential")
    amplitude = problem.amplitude * np.exp(-problem.frequency ** 2 * t)
    # sinc uses sin(pi*x)/(pi*x); this gives exact cell averages, not midpoint values.
    amplitude *= np.sinc(problem.frequency * cell_width / (2 * np.pi))
    return np.prod(1 + amplitude * np.cos(problem.frequency * (x - problem.phase)), axis=-1) / (2 * np.pi) ** problem.dim


def _bernoulli(z):
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    small = np.abs(z) < 1e-4
    a = z[small]
    result[small] = 1 - a / 2 + a ** 2 / 12 - a ** 4 / 720
    positive = (~small) & (z > 0)
    a = z[positive]
    result[positive] = a * np.exp(-a) / (-np.expm1(-a))
    negative = (~small) & ~positive
    a = z[negative]
    result[negative] = a / np.expm1(a)
    return result


def fv_generator(problem, n: int):
    """Scharfetter-Gummel generator for rho_t = div(rho grad V) + Delta rho."""
    shape = (n,) * problem.dim
    index = np.arange(n ** problem.dim).reshape(shape)
    source = index.ravel()
    values = potential(problem, grid_points(problem.dim, n))
    h = 2 * np.pi / n
    rows, cols, data = [], [], []
    for axis in range(problem.dim):
        target = np.roll(index, -1, axis=axis).ravel()
        delta = values[target] - values[source]
        forward, backward = _bernoulli(delta) / h ** 2, _bernoulli(-delta) / h ** 2
        rows.extend((target, source, source, target))
        cols.extend((source, target, source, target))
        data.extend((forward, backward, -forward, -backward))
    size = n ** problem.dim
    return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size)).tocsr()


def initial_cell_density(problem, n: int):
    x = grid_points(problem.dim, n)
    amplitude = problem.amplitude * np.sinc(problem.frequency / n)
    return np.prod(1 + amplitude * np.cos(problem.frequency * (x - problem.phase)), axis=-1) / (2 * np.pi) ** problem.dim


def reference_density(problem, times, n: int, *, force_fv: bool = False):
    """Return raw (time, flattened-cell) densities without normalization/clipping."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError("Reference times must be finite, nonnegative and sorted")
    x = grid_points(problem.dim, n)
    if not force_fv and not (problem.kappa or problem.coupling):
        return np.stack([analytic_heat_density(problem, x, t, 2 * np.pi / n) for t in times])
    generator = fv_generator(problem, n)
    initial = initial_cell_density(problem, n)
    # Equally spaced requested times share a single matrix-exponential Krylov call.
    if len(times) > 1 and times[-1] > times[0] and np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-12, atol=1e-15):
        return expm_multiply(generator, initial, start=times[0], stop=times[-1], num=len(times), endpoint=True, traceA=generator.diagonal().sum())
    return np.stack([initial.copy() if t == 0 else expm_multiply(t * generator, initial, traceA=t * generator.diagonal().sum()) for t in times])


def restrict_density(fine, dim: int, n: int):
    """Average each group of 2**dim fine cells to its parent coarse cell."""
    fine = np.asarray(fine)
    leading = fine.shape[:-1]
    shape = leading + tuple(value for _ in range(dim) for value in (n, 2))
    value = fine.reshape(shape)
    for axis in reversed(range(dim)):
        value = value.mean(axis=len(leading) + 2 * axis + 1)
    return value.reshape(leading + (n ** dim,))
