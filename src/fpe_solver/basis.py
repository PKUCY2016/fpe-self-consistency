"""Real periodic Fourier fields with Bernstein time coefficients.

The spatial feature order is ``[1, cos(k1.x), sin(k1.x), ...]``. Each
nonzero integer wave vector in the coordinatewise frequency box occurs once,
with its first nonzero component positive. In particular, mixed 2D modes are
present, but the redundant pair k and -k is not.
"""
import math
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import product

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Basis:
    dim: int = 1
    K: int = 2
    p: int = 3
    T: float = 1.0

    def __post_init__(self):
        if self.dim not in (1, 2):
            raise ValueError("Basis supports dimensions 1 and 2")
        if not isinstance(self.K, int) or not isinstance(self.p, int) or self.K < 0 or self.p < 0:
            raise ValueError("K and p must be nonnegative integers")
        if not math.isfinite(self.T) or self.T <= 0:
            raise ValueError("T must be positive and finite")

    @cached_property
    def modes(self):
        modes = [
            k for k in product(range(-self.K, self.K + 1), repeat=self.dim)
            if any(k) and next(v for v in k if v) > 0
        ]
        result = np.asarray(modes, dtype=np.int64).reshape(-1, self.dim)
        result.setflags(write=False)
        return result

    @property
    def nfeatures(self):
        return 1 + 2 * len(self.modes)

    @property
    def shape(self):
        return self.p + 1, self.nfeatures, self.dim

    def zeros(self):
        return jnp.zeros(self.shape, dtype=jnp.float64)

    def to_dict(self):
        return asdict(self)

    def time_weights(self, t):
        """Bernstein weights; t is scalar in the interval [0, T]."""
        tau = jnp.asarray(t, dtype=jnp.float64) / self.T
        i = jnp.arange(self.p + 1)
        choose = jnp.asarray([math.comb(self.p, n) for n in range(self.p + 1)], dtype=jnp.float64)
        return choose * tau**i * (1 - tau)**(self.p - i)

    def _coefficients_at(self, coeff, t):
        if jnp.shape(coeff) != self.shape:
            raise ValueError(f"Expected coefficients {self.shape}, got {jnp.shape(coeff)}")
        return jnp.einsum("p,pfd->fd", self.time_weights(t), coeff)

    def velocity(self, coeff, t, x):
        """Evaluate a field for x with arbitrary leading batch dimensions."""
        c = self._coefficients_at(coeff, t)
        angle = jnp.einsum("...d,md->...m", x, jnp.asarray(self.modes))
        return c[0] + jnp.cos(angle) @ c[1::2] + jnp.sin(angle) @ c[2::2]

    def spatial(self, coeff, t, x):
        """Return v, Dv, div(v), grad(div(v)) with Dv[out, input]."""
        c = self._coefficients_at(coeff, t)
        modes = jnp.asarray(self.modes, dtype=jnp.float64)
        angle = jnp.einsum("...d,md->...m", x, modes)
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        cc, cs = c[1::2], c[2::2]
        v = c[0] + cosine @ cc + sine @ cs
        derivative_amplitudes = -sine[..., :, None] * cc + cosine[..., :, None] * cs
        jac = jnp.einsum("...mo,mi->...oi", derivative_amplitudes, modes)
        projection_cos = jnp.einsum("mi,mi->m", cc, modes)
        projection_sin = jnp.einsum("mi,mi->m", cs, modes)
        div = -sine @ projection_cos + cosine @ projection_sin
        grad_div = (-cosine * projection_cos - sine * projection_sin) @ modes
        return v, jac, div, grad_div

    def prolong(self, coeff, new_basis):
        """Embed exactly in a larger spatial and/or temporal space.

        New Fourier coefficients are zero. Repeated Bernstein degree elevation
        preserves the polynomial, including its endpoints, without refitting.
        """
        if self.dim != new_basis.dim or self.T != new_basis.T:
            raise ValueError("Prolongation requires the same dimension and final time")
        if new_basis.K < self.K or new_basis.p < self.p:
            raise ValueError("Prolongation only permits increasing K and p")
        if jnp.shape(coeff) != self.shape:
            raise ValueError("Coefficient shape does not match source basis")
        out = jnp.zeros((self.p + 1, new_basis.nfeatures, self.dim), dtype=jnp.float64)
        out = out.at[:, 0].set(coeff[:, 0])
        destinations = {tuple(k): n for n, k in enumerate(new_basis.modes)}
        for n, k in enumerate(self.modes):
            target = destinations[tuple(k)]
            out = out.at[:, 1 + 2*target:3 + 2*target].set(coeff[:, 1 + 2*n:3 + 2*n])
        for degree in range(self.p, new_basis.p):
            weight = jnp.arange(degree + 2, dtype=jnp.float64) / (degree + 1)
            zero = jnp.zeros_like(out[:1])
            out = (weight[:, None, None] * jnp.concatenate((zero, out), axis=0)
                   + (1 - weight[:, None, None]) * jnp.concatenate((out, zero), axis=0))
        return out
