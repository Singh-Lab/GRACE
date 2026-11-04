"""Smoke tests. They run in a couple of minutes on a laptop CPU.

    pytest -q tests/

Tests that need trained oracles skip themselves when the oracles are absent,
so a fresh clone without the Zenodo bundle still gives a green run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grace import list_tasks, load_task, optimize
from grace.api import METHODS
from grace.registry import TaskError


def ready(key):
    return not load_task(key).missing_models()


def test_every_task_config_loads():
    keys = list_tasks()
    assert len(keys) == 14
    for k in keys:
        t = load_task(k)
        assert t.constraints, f"{k} has no constraints"
        assert t.target_sign in (1.0, -1.0)
        assert t.K > 0 and t.steps > 0
        assert t.describe()


def test_starts_match_the_declared_length():
    for k in list_tasks():
        t = load_task(k)
        if not t.starts_file.exists():
            continue
        s = t.starts()
        expect = 1 if k.startswith("protein_tem") else 30   # TEM has one reference
        assert len(s) == expect, f"{k} ships {len(s)} starts, expected {expect}"
        assert all(set(x) <= set(t.alphabet) for x in s), f"{k} start outside alphabet"
        if isinstance(t.length, int) and t.domain != "dna":
            assert {len(x) for x in s} == {t.length}


def test_unknown_task_and_method_raise():
    with pytest.raises(TaskError):
        load_task("not_a_task")
    with pytest.raises(TaskError):
        optimize("protein_canya", method="not_a_method")


@pytest.mark.skipif(not ready("protein_canya"), reason="oracles not downloaded")
def test_grace_improves_and_stays_feasible():
    d = optimize("protein_canya", start=1, steps=60, quiet=True)
    assert d.gain > 0
    assert d.feasible
    assert len(d.sequence) == len(d.start_sequence)
    assert d.n_oracle_calls > 0


@pytest.mark.skipif(not ready("protein_canya"), reason="oracles not downloaded")
def test_a_custom_sequence_runs():
    wt = load_task("protein_canya").start(2)
    d = optimize("protein_canya", sequence=wt, steps=40, quiet=True)
    assert d.start_sequence == wt


@pytest.mark.skipif(not ready("protein_canya"), reason="oracles not downloaded")
def test_tolerance_scale_reaches_the_run():
    d = optimize("protein_canya", start=1, steps=40, tolerance_scale=0.5, quiet=True)
    assert d.tolerances == [0.5]


@pytest.mark.skipif(not ready("protein_canya"), reason="oracles not downloaded")
def test_a_sequence_off_the_alphabet_is_rejected():
    with pytest.raises(TaskError):
        optimize("protein_canya", sequence="XXXXXXXXXXXXXXXXXXXX", quiet=True)


@pytest.mark.skipif(not ready("rna_mfe"), reason="oracles not downloaded")
def test_builtin_constraint_task_runs():
    d = optimize("rna_mfe", start=1, steps=40, quiet=True)
    assert d.sequence and set(d.sequence) <= set("ACGU")


@pytest.mark.skipif(not ready("protein_canya"), reason="oracles not downloaded")
@pytest.mark.parametrize("method", ["adalead", "simulated_annealing", "random"])
def test_baselines_run_through_the_same_entry_point(method):
    d = optimize("protein_canya", start=1, method=method, steps=40, quiet=True)
    assert d.method == method
