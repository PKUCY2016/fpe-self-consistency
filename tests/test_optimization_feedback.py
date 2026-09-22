"""Rejected candidates inform one-setting optimization trials without oracle feedback."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from fpe_solver import controller
from fpe_solver.basis import Basis
from fpe_solver.config import SolverConfig
from fpe_solver.problems import Problem


def rejected_event(**validation_overrides):
    return {
        "action": "increase_space", "promoted": False, "qualified": True,
        "config": SolverConfig(K=5).to_dict(),
        "validation": {
            "old_residual": .01252, "new_residual": .07565,
            "paired_standard_error": .0001,
            "numerical_change_old": 1e-7, "numerical_change_new": 1e-7,
            "numerical_consistency": True, "score_consistency": {"passed": True},
            "incumbent_score_consistency": {"passed": True},
        } | validation_overrides,
        "history": [{"update": 500, "loss": .08}],
    }


def clean_diagnostic(event):
    return {"numerically_stable": True, "sampling_unstable": False, "oscillatory": False,
            "potential_optimization_regression": controller.rejected_optimization_regression(event),
            "probe_gradient_rms": {"increase_space": 100., "increase_time": 99.}}


def test_significant_rejected_regression_prefers_only_champion_learning_rate_change():
    champion = SolverConfig(K=4)
    event = rejected_event()
    assert controller.rejected_optimization_regression(event)
    action, candidate = controller.select_action(champion, clean_diagnostic(event), ["increase_space"])
    assert action == "reduce_lr"
    changed = [name for name, value in champion.to_dict().items() if candidate.to_dict()[name] != value]
    assert changed == ["learning_rate"]
    assert candidate.K == champion.K == 4  # The rejected K=5 template must not be silently reused.
    assert candidate.learning_rate == champion.learning_rate/2


@pytest.mark.parametrize("overrides", [
    {"new_residual": .0126},
    {"new_residual": .01},
    {"paired_standard_error": .03},
    {"numerical_change_old": .02, "numerical_change_new": .02},
])
def test_weak_or_unresolved_evidence_does_not_trigger(overrides):
    assert not controller.rejected_optimization_regression(rejected_event(**overrides))


@pytest.mark.parametrize("key", ["score_consistency", "incumbent_score_consistency"])
def test_both_score_checks_are_required(key):
    assert not controller.rejected_optimization_regression(rejected_event(**{key: {"passed": False}}))
    assert not controller.rejected_optimization_regression(rejected_event(**{key: {}}))


def test_numerical_mismatch_is_not_labeled_optimization_regression():
    event = rejected_event(numerical_consistency=False)
    assert not controller.rejected_optimization_regression(event)
    diagnostic = clean_diagnostic(event) | {"numerically_stable": False}
    assert controller.select_action(SolverConfig(), diagnostic)[0] == "refine_steps"


@pytest.mark.parametrize("unresolved", ["numerically_stable", "sampling_unstable"])
def test_numerics_and_sampling_keep_priority_over_lr(unresolved):
    diagnostic = clean_diagnostic(rejected_event())
    diagnostic[unresolved] = unresolved != "numerically_stable"
    expected = "refine_steps" if unresolved == "numerically_stable" else "increase_batch"
    assert controller.select_action(SolverConfig(), diagnostic)[0] == expected


@pytest.mark.parametrize("key,value", [("promoted", True), ("qualified", False)])
def test_promoted_or_unqualified_events_are_not_regression_evidence(key, value):
    event = rejected_event()
    event[key] = value
    assert not controller.rejected_optimization_regression(event)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_and_nonfinite_internal_statistics_fail_closed(value):
    assert not controller.rejected_optimization_regression(rejected_event(paired_standard_error=value))


def test_lr_trial_respects_floor_and_previous_rejection():
    diagnostic = clean_diagnostic(rejected_event())
    at_floor = replace(SolverConfig(), learning_rate=1e-4)
    assert controller.select_action(at_floor, diagnostic)[0] != "reduce_lr"
    assert controller.select_action(SolverConfig(), diagnostic, ["reduce_lr"])[0] != "reduce_lr"


def test_oracle_metadata_is_never_read_to_classify_regression():
    class OracleForbidden(dict):
        def __getitem__(self, key):
            if key in {"true_kl", "reference_density", "heldout_tv", "oracle"}:
                raise AssertionError("The controller tried to read an offline oracle metric")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in {"true_kl", "reference_density", "heldout_tv", "oracle"}:
                raise AssertionError("The controller tried to read an offline oracle metric")
            return super().get(key, default)

    event = OracleForbidden(rejected_event())
    event["validation"] = OracleForbidden(event["validation"])
    event.update(true_kl=1e-30, heldout_tv=0., oracle="must not be read")
    event["validation"].update(true_kl=1e30, reference_density="must not be read")
    assert controller.rejected_optimization_regression(event)
    assert controller.select_action(SolverConfig(), clean_diagnostic(event))[0] == "reduce_lr"
    weak = OracleForbidden(deepcopy(rejected_event(new_residual=.0126)))
    weak["true_kl"] = 1e30  # A bad external error must not manufacture an internal regression.
    assert not controller.rejected_optimization_regression(weak)


def test_diagnose_uses_rejection_even_when_champion_history_is_improving(monkeypatch):
    config = SolverConfig(K=8, p=15, validation_samples=8, max_validation_samples=8, steps=4)
    problem = Problem(amplitude=0.)
    basis = Basis(problem.dim, config.K, config.p, problem.T)
    monkeypatch.setattr(controller, "residual_samples", lambda b, c, p, x, s: np.ones(len(x)))
    history = [{"loss": 1.}]*10 + [{"loss": .1}]*10
    with_rejection = controller.diagnose(basis, basis.zeros(), config, problem, 0, 1, history,
                                        rejected_events=[rejected_event()])
    assert not with_rejection["oscillatory"]
    assert with_rejection["potential_optimization_regression"]
    assert controller.select_action(config, with_rejection)[0] == "reduce_lr"
    without_rejection = controller.diagnose(basis, basis.zeros(), config, problem, 0, 2, history)
    assert not without_rejection["potential_optimization_regression"]
