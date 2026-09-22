from pathlib import Path

import jax
import numpy as np

from .flow import integrate
from .runner import get_model
from .storage import read_json


class ProbabilityFlow:
    """Time-marginal sampler, not a simulator of Brownian path observables."""
    def __init__(self,basis,coeff,problem,steps=256):
        self.basis,self.coeff,self.problem,self.steps = basis,coeff,problem,steps

    @classmethod
    def load(cls,run_dir,steps=None):
        state = read_json(Path(run_dir)/"state.json")
        saved_steps = state["champion"]["config"]["steps"]
        return cls(*get_model(run_dir), steps=saved_steps if steps is None else steps)

    def sample(self,n,t=None,seed=0):
        t = self.problem.T if t is None else float(t)
        if not np.isfinite(t) or not 0 <= t <= self.problem.T or n < 1:
            raise ValueError("Require n>=1 and 0<=t<=T")
        x = self.problem.sample(jax.random.PRNGKey(seed),n)
        if t == 0:
            return np.asarray(x)
        state = integrate(self.basis,self.coeff,self.problem,x,self.steps,end_time=t)
        return np.asarray(state[:,:self.problem.dim]) % (2*np.pi)

    def log_density(self,x,t=None):
        from .evaluation import log_density
        t = self.problem.T if t is None else float(t)
        if not np.isfinite(t) or not 0 <= t <= self.problem.T:
            raise ValueError("Require 0<=t<=T")
        return log_density(self.basis,self.coeff,self.problem,np.asarray(x),t)
