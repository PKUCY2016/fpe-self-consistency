import ast
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from fpe_solver.config import SolverConfig
from fpe_solver.controller import paired_gate, sample_key, select_action
from fpe_solver.problems import Problem, benchmark
from fpe_solver.storage import load_arrays, run_lock, save_arrays


def test_pairing_and_no_uncertain_promotion():
    old = np.linspace(.9,1.1,100)
    assert paired_gate(old,.9*old)["promote"]
    assert not paired_gate(old,.97*old)["promote"]
    assert not paired_gate(old,old)["promote"]
    assert not paired_gate(old,.9*old,numerical_margin=.1)["promote"]
    assert not paired_gate(old,np.full(100,np.nan))["promote"]


def test_actions_change_one_mechanism_and_caps():
    c = SolverConfig()
    action,new = select_action(c,{"numerically_stable":False})
    assert action == "refine_steps"
    assert [k for k,v in c.to_dict().items() if v != new.to_dict()[k]] == ["steps"]
    maxed = replace(c,K=8,p=15,batch=1024,steps=4096,learning_rate=1e-4)
    assert select_action(maxed,{"numerically_stable":True})[0] is None


def test_independent_validation_stream():
    assert not np.array_equal(sample_key(0,0,100),sample_key(0,0,9101))
    assert not np.array_equal(sample_key(0,1,9101),sample_key(0,0,9101))


def test_controller_does_not_import_oracle():
    from fpe_solver import controller
    tree = ast.parse(Path(controller.__file__).read_text())
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any("evaluation" in n or "reference" in n for n in imports)


def test_problem_sample_and_score():
    p = benchmark("heat1d")
    x = p.sample(jax.random.PRNGKey(22),20000)
    assert abs(float(np.cos(x).mean())-.25)<.015
    score = jax.vmap(jax.grad(p.log_density0))(x[:10])
    np.testing.assert_allclose(score,p.score0(x[:10]),atol=1e-12)
    with pytest.raises(ValueError):
        Problem(amplitude=1.)
    with pytest.raises(ValueError):
        Problem(dim=3)


def test_checkpoint_integrity_and_no_pickle(tmp_path):
    import optax
    c = np.zeros((2,3,1))
    state = optax.adam(.01).init(c)
    digest = save_arrays(tmp_path/"c.npz",c,jax.random.PRNGKey(0),state)
    out,_key,opt = load_arrays(tmp_path/"c.npz",state,digest)
    np.testing.assert_array_equal(c,out)
    assert opt is not None
    with pytest.raises(ValueError):
        load_arrays(tmp_path/"c.npz",expected_hash="bad")


def test_lock_rejects_second_writer(tmp_path):
    with run_lock(tmp_path), pytest.raises(RuntimeError), run_lock(tmp_path):
        pass
