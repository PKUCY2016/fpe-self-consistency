"""Periodic probability flows. Empirical validation is not a certified error bound."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax

jax.config.update("jax_enable_x64", True)
__version__ = "0.1.0"

