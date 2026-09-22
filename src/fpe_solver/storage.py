"""Atomic, non-pickle checkpoints. No overwritten run directories."""
import hashlib
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(json_value(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        name = f.name
    os.replace(name, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_arrays(path, coeff, key, opt_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"coeff": np.asarray(coeff), "key": np.asarray(key)}
    if opt_state is not None:
        arrays.update({f"opt_{i}": np.asarray(x) for i, x in enumerate(jax.tree_util.tree_leaves(opt_state))})
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
        name = f.name
    os.replace(name, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arrays(path, opt_template=None, expected_hash=None):
    path = Path(path)
    if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Checkpoint hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        coeff, key = jnp.asarray(data["coeff"]), jnp.asarray(data["key"])
        opt = None
        if opt_template is not None:
            leaves, tree = jax.tree_util.tree_flatten(opt_template)
            values = [jnp.asarray(data[f"opt_{i}"]) for i in range(len(leaves))]
            if any(v.shape != np.shape(t) for v, t in zip(values, leaves)):
                raise ValueError("Optimizer state shape mismatch")
            opt = jax.tree_util.tree_unflatten(tree, values)
    return coeff, key, opt


@contextmanager
def run_lock(directory):
    """OS advisory lock releases automatically on process death (including SIGKILL)."""
    import fcntl
    path = Path(directory) / ".run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This run already has an active writer") from exc
        f.seek(0)
        f.truncate()
        f.write(json.dumps({"pid": os.getpid(), "started": time.time()}))
        f.flush()
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
