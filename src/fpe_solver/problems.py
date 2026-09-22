"""Frozen problem definitions; deliberately no ground-truth solution interface."""
import math
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Problem:
    name: str = "heat1d"
    dim: int = 1
    T: float = 1.0
    amplitude: float = 0.5
    frequency: int = 1
    phase: float = 0.0
    kappa: float = 0.0
    coupling: float = 0.0

    def __post_init__(self):
        if self.dim not in (1, 2):
            raise ValueError("Periodic v1 supports dimensions 1 and 2 only")
        if not all(math.isfinite(v) for v in (self.T, self.amplitude, self.phase, self.kappa, self.coupling)):
            raise ValueError("Problem values must be finite")
        if self.T <= 0 or abs(self.amplitude) >= 1 or self.frequency < 1:
            raise ValueError("Require T>0, |amplitude|<1, frequency>=1")
        if self.frequency != int(self.frequency) or self.kappa < 0 or self.coupling < 0:
            raise ValueError("Invalid frequency or potential")
        if self.dim == 1 and self.coupling:
            raise ValueError("Coupling requires dimension 2")

    def log_density0(self, x):
        angle = self.frequency * (x - self.phase)
        return jnp.sum(jnp.log1p(self.amplitude * jnp.cos(angle)), axis=-1) - self.dim * math.log(2*math.pi)

    def density0(self, x):
        return jnp.exp(self.log_density0(x))

    def score0(self, x):
        a = self.frequency * (x - self.phase)
        return -self.amplitude * self.frequency * jnp.sin(a) / (1 + self.amplitude*jnp.cos(a))

    def potential(self, x):
        v = self.kappa * jnp.sum(1-jnp.cos(2*x), axis=-1)
        if self.dim == 2:
            v = v + self.coupling*(1-jnp.cos(x[..., 0]-x[..., 1]))
        return v

    def grad_potential(self, x):
        g = 2*self.kappa*jnp.sin(2*x)
        if self.dim == 2:
            c = self.coupling*jnp.sin(x[..., 0]-x[..., 1])
            g = g + jnp.stack((c, -c), axis=-1)
        return g

    def sample(self, key, n):
        """Exact product rejection sampler, no estimated score or fitted density."""
        shape = (n, self.dim)
        def cond(state):
            return jnp.any(~state[2])
        def body(state):
            key, x, accepted = state
            key, kx, ku = jax.random.split(key, 3)
            proposal = jax.random.uniform(kx, shape, minval=0., maxval=2*math.pi, dtype=jnp.float64)
            ratio = (1+self.amplitude*jnp.cos(self.frequency*(proposal-self.phase)))/(1+abs(self.amplitude))
            take = (~accepted) & (jax.random.uniform(ku, shape, dtype=jnp.float64) < ratio)
            return key, jnp.where(take, proposal, x), accepted | take
        return jax.lax.while_loop(cond, body, (key, jnp.zeros(shape), jnp.zeros(shape, dtype=bool)))[1]

    def to_dict(self):
        return asdict(self)


def benchmark(name, **overrides):
    cases = {
        "uniform": {"amplitude": 0.},
        "heat1d": {},
        "heat2d": {"dim": 2},
        "doublewell": {"kappa": 1.},
        "coupled2d": {"dim": 2, "kappa": .5, "coupling": .25},
        "evolution": {"amplitude": .35, "frequency": 3, "T": .25},
        "stress_barrier": {"kappa": 4., "T": 3.},
        "stress_frequency": {"frequency": 4},
    }
    if name not in cases:
        raise ValueError(f"Unknown benchmark {name}; choose from {list(cases)}")
    return Problem(name=name, **(cases[name] | overrides))
