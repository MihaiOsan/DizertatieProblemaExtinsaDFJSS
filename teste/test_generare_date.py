# tests/test_dataset.py
from copy import deepcopy

import generare as gnerator
import pytest

@pytest.mark.parametrize("seed", [101, 202, 303])
def test_deterministic_for_same_seed(seed):
    params = dict(
        num_jobs=10, num_machines=3, machine_util=0.7, ec_percent=0.05,
        min_num_ops=2, max_num_ops=3,
        min_num_candidate_machines=2, max_num_candidate_machines=3,
        min_proc_time=5, max_proc_time=8,
        allowance_factors=[2], max_time_horizon=200,
        mean_time_to_failure=50, mean_repair_time=5,
        seed=seed, cancelled_job_frac=0.1,
        cancel_delay_range=(5, 20)
    )
    ds1 = gnerator.generate_flex_dataset_with_breakdowns(**params)
    ds2 = gnerator.generate_flex_dataset_with_breakdowns(**deepcopy(params))
    assert ds1 == ds2, "același seed → același dataset"

@pytest.fixture
def small_dataset():
    """Un set minimal, rapid de generat, pentru majoritatea testelor."""
    return gnerator.generate_flex_dataset_with_breakdowns(
        num_jobs=20,
        num_machines=5,
        machine_util=0.8,
        ec_percent=0.1,
        min_num_ops=2,
        max_num_ops=4,
        min_num_candidate_machines=2,
        max_num_candidate_machines=3,
        min_proc_time=10,
        max_proc_time=20,
        allowance_factors=[2, 4],
        max_time_horizon=500,
        mean_time_to_failure=100,
        mean_repair_time=10,
        seed=999,
        cancelled_job_frac=0.2,
        cancel_delay_range=(5, 30)
    )

def test_job_count(small_dataset):
    assert len(small_dataset["jobs"]) == 20

def test_unique_job_ids(small_dataset):
    ids = [job["id"] for job in small_dataset["jobs"]]
    assert len(ids) == len(set(ids)), "numar joburi invalid"

def test_arrival_times_non_negative_sorted(small_dataset):
    arrivals = [job["arrival_time"] for job in small_dataset["jobs"]]
    assert min(arrivals) >= 0, "timp de sosire negativ"
    assert arrivals == sorted(arrivals), "lista de sosiri nevalida"

    # tests/test_cancelled_jobs.py
def test_cancelled_jobs_are_valid(small_dataset):
    dyn = small_dataset.get("dynamic_events", {})
    cancelled = dyn.get("cancelled_jobs", [])
    job_dict = {j["id"]: j for j in small_dataset["jobs"]}

    for ev in cancelled:
        jid = ev["job_id"]
        cancel_t = ev["time"]
        assert jid in job_dict, "job_id invalid în dynamic_events"
        assert cancel_t > job_dict[jid]["arrival_time"], "anularea trebuie să fie după sosirea job-ului"


def test_etpc_consistency(small_dataset):
    ids = set(job["id"] for job in small_dataset["jobs"])
    for c in small_dataset["etpc_constraints"]:
        assert c["fore_job"] in ids, "id fore invalid"
        assert c["hind_job"] in ids, "id hind invalid"
        # verifică că op_idx sunt valide
        fore_ops = len(small_dataset["jobs"][c["fore_job"]]["operations"])
        hind_ops = len(small_dataset["jobs"][c["hind_job"]]["operations"])
        assert 0 <= c["fore_op_idx"] < fore_ops
        assert 0 <= c["hind_op_idx"] < hind_ops
        assert c["fore_job"] != c["hind_job"], "acelasi job cu precedenta"


def test_etpc_excludes_cancelled_jobs(small_dataset):
    cancelled = {ev["job_id"] for ev in small_dataset.get("dynamic_events", {}).get("cancelled_jobs", [])}
    for c in small_dataset["etpc_constraints"]:
        assert c["fore_job"] not in cancelled


def test_breakdowns_present_for_all_machines(small_dataset):
    assert len(small_dataset["machine_breakdowns"]) == 5
    for m_id, ev in small_dataset["machine_breakdowns"].items():
        assert isinstance(m_id, int)
        assert all(e["start_time"] <= 500 for e in ev)

def test_due_date_after_arrival(small_dataset):
    for job in small_dataset["jobs"]:
        assert job["due_date"] >= job["arrival_time"]

def test_weights_distribution(small_dataset):
    """Verifică raport 4:2:1 la limită (±1 per categorie)."""
    weights = [j["weight"] for j in small_dataset["jobs"]]
    n = len(weights)
    exp = {4: 0.2*n, 2: 0.6*n, 1: 0.2*n}
    for w in (1, 2, 4):
        # admitem +/-1 deoarece rotunjim în script
        assert abs(weights.count(w) - exp[w]) <= 1

def test_candidate_machines_non_empty(small_dataset):
    for job in small_dataset["jobs"]:
        for op in job["operations"]:
            cm = op["candidate_machines"]
            assert isinstance(cm, dict) and cm, "candidate_machines trebuie să fie dict non-gol"
            # cheile sunt mașini int, valorile sunt timpi int >0
            for mc_id, p in cm.items():
                assert isinstance(mc_id, int) and isinstance(p, int) and p > 0


