"""Offline acceptance must not turn missing, invalid, or over-budget evidence into success."""
from pathlib import Path

import pytest

from fpe_solver.campaign import specification, summarize, tasks
from fpe_solver.storage import read_json, write_json

PHASES = ("fixed", "adaptive", "control", "audit", "stress")


def initialize(root, complete=True):
    write_json(root/"specification.json", specification())
    if complete:
        for phase in PHASES:
            for name, _ in tasks(phase):
                write_json(root/name/"result.json", {"passed": True, "status": "TRAINED"})
    for seed in (0, 1, 2):
        for mode, kl in (("evolve", .0003), ("fixed_budget_control", .001)):
            path = root/f"control-{mode}-s{seed}"
            write_json(path/"state.json", {"elapsed_seconds": 110., "status": "TRAINED"})
            write_json(path/"evaluation.json", {
                "metrics": [{"time": .25, "kl": kl}],
                "checks": {"mass": True, "reference_refinement": True,
                           "model_numerical_refinement": True, "model_quadrature_refinement": True},
            })


def change(path, key, value):
    payload = read_json(path)
    payload[key] = value
    write_json(path, payload)


def test_full_task_manifest_covers_every_seed_and_frozen_scope():
    by_phase = {phase: list(tasks(phase)) for phase in PHASES}
    assert {phase: len(rows) for phase, rows in by_phase.items()} == {
        "fixed": 15, "adaptive": 15, "control": 6, "audit": 6, "stress": 3,
    }
    names = [name for rows in by_phase.values() for name, _ in rows]
    assert len(names) == len(set(names)) == 45
    spec = specification()
    assert spec["seeds"] == [0, 1, 2]
    assert spec["times"] == 21
    assert spec["campaign_seconds"] == 8*3600
    assert spec["thresholds"]["kl"] == 1e-3
    for seed in spec["seeds"]:
        controls = [record for _, record in by_phase["control"] if record["seed"] == seed]
        assert {record["mode"] for record in controls} == {"evolve", "fixed_budget_control"}
        assert all(record["seconds"] == 120 for record in controls)
        assert controls[0]["problem"] == controls[1]["problem"]


def test_complete_qualified_evidence_can_pass(tmp_path):
    initialize(tmp_path)
    result = summarize(tmp_path)
    assert result["control"]["passed"]
    assert result["control"]["median_kl_reduction"] == pytest.approx(.7)
    assert result["engineering_acceptance_passed"]
    assert not result["scientific_proof_complete"]
    assert result["missing"] == []


@pytest.mark.parametrize("mode", ["evolve", "fixed_budget_control"])
def test_over_budget_control_cannot_pass_even_with_better_kl(tmp_path, mode):
    initialize(tmp_path)
    change(tmp_path/f"control-{mode}-s1"/"state.json", "elapsed_seconds", 120.01)
    result = summarize(tmp_path)
    assert not result["control"]["passed"]
    assert not result["engineering_acceptance_passed"]
    seed_record = next(row for row in result["control"]["seeds"] if row["seed"] == 1)
    assert not seed_record["budget_compliant"]


@pytest.mark.parametrize("baseline", [0., -1e-6, 1e-4])
def test_zero_or_unresolved_kl_baseline_cannot_pass(tmp_path, baseline):
    initialize(tmp_path)
    for seed in (0, 1, 2):
        change(tmp_path/f"control-fixed_budget_control-s{seed}"/"evaluation.json",
               "metrics", [{"time": .25, "kl": baseline}])
        change(tmp_path/f"control-evolve-s{seed}"/"evaluation.json",
               "metrics", [{"time": .25, "kl": 0.}])
    result = summarize(tmp_path)
    assert not result["control"]["passed"]
    assert not result["engineering_acceptance_passed"]


def test_unresolved_numerics_cannot_pass_control(tmp_path):
    initialize(tmp_path)
    path = tmp_path/"control-evolve-s0"/"evaluation.json"
    record = read_json(path)
    record["checks"]["model_quadrature_refinement"] = False
    write_json(path, record)
    result = summarize(tmp_path)
    assert not result["control"]["passed"]
    assert not result["engineering_acceptance_passed"]


def test_missing_results_and_failed_receipts_remain_distinguishable(tmp_path):
    initialize(tmp_path, complete=False)
    failed = "fixed-uniform-s0"
    write_json(tmp_path/f"{failed}.receipt.json", {"status": "FAILED", "exit_code": 1, "wall_seconds": 2.})
    result = summarize(tmp_path)
    assert failed in result["results"]
    assert result["results"][failed]["status"] == "FAILED"
    assert not result["results"][failed]["passed"]
    assert "adaptive-heat1d-s2" in result["missing"]
    assert not result["engineering_acceptance_passed"]


@pytest.mark.parametrize("phase_name", ["fixed-uniform-s0", "stress-stress_frequency"])
def test_worker_failure_cannot_complete_the_campaign(tmp_path, phase_name):
    initialize(tmp_path)
    (tmp_path/phase_name/"result.json").unlink()
    write_json(tmp_path/f"{phase_name}.receipt.json", {
        "status": "TIMEOUT", "exit_code": -15, "wall_seconds": 120.,
    })
    result = summarize(tmp_path)
    assert result["results"][phase_name]["status"] == "TIMEOUT"
    assert not result["results"][phase_name]["passed"]
    assert not result["engineering_acceptance_passed"]


def test_report_evidence_paths_are_real_files(tmp_path):
    initialize(tmp_path, complete=False)
    receipt = tmp_path/"fixed-uniform-s0.receipt.json"
    write_json(receipt, {"status": "FAILED", "wall_seconds": 1.})
    result = summarize(tmp_path)
    assert Path(result["results"]["fixed-uniform-s0"]["receipt"]).is_file()
    assert read_json(tmp_path/"summary.json")["specification_sha256"] == result["specification_sha256"]
