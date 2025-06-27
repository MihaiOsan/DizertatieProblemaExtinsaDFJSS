import pytest
import numpy as np

import utils as ut
from data_reader import FJSPInstance, Job

# helper synthetic instance & schedule

def _inst_two_jobs():
    inst = FJSPInstance("test")
    inst.num_machines = 2
    for jid, (dd, wt) in enumerate([(10, 2), (20, 1)]):
        job = Job(sim_id=-1, original_json_index=jid, arrival_time=0,
                  due_date=dd, weight=wt)
        inst._assign_sim_id(job)
        inst.jobs_defined_in_file.append(job)
    return inst


def test_sanitize_filename():
    s = " A*weird  name!!.txt "
    clean = ut.sanitize_filename_str(s)
    assert clean in {"Aweird_name", "Aweird_name.txt", "Aweird_nametxt"}



def test_calculate_std_dev():
    assert ut.calculate_std_dev([]) == 0.0
    assert ut.calculate_std_dev([5]) == 0.0
    data = [1, 2, 3, 4]
    assert abs(ut.calculate_std_dev(data) - float(np.std(data))) < 1e-9


def test_safe_avg_list():
    assert ut.safe_avg_list([]) == 0.0
    assert ut.safe_avg_list([2, 4]) == 3.0


def test_per_machine_op_counts():
    sched = [(0, 0, 0, 0, 3), (1, 0, 1, 1, 4), (0, 1, 0, 5, 7)]
    counts = ut.get_per_machine_operation_counts(sched, 2)
    assert counts == {0: 2, 1: 1}


def test_calc_machine_metrics_simple():
    sched = [(0, 0, 0, 0, 5), (0, 1, 0, 6, 8)]  # one machine only
    res = ut.calc_machine_metrics(sched, 1, schedule_makespan=8)
    total_idle, avg_idle, _, total_busy, util, idle_list, busy_list, counts = res
    assert total_busy == 7  # 5 + 2
    assert util == pytest.approx(7 / 8)
    assert idle_list[0] == 1  # gap between 5 and 6
    assert counts == {0: 2}


def test_calc_job_metrics():
    inst = _inst_two_jobs()
    sched = [(0, 0, 0, 0, 6), (0, 1, 1, 6, 9)]  # job 0 ops, but job objects have no ops → 0 completed
    metrics = ut.calc_job_related_metrics(sched, inst, schedule_makespan=12)
    assert metrics["num_completed_jobs"] == 0
    assert metrics["total_weighted_tardiness"] >= 0

    assert metrics["total_weighted_tardiness"] >= 0


def test_generate_csvs(tmp_path):
    idle, busy = [1.0], [3.0]
    counts = {0: 2}
    file_m = tmp_path / "machines.csv"
    ut.generate_per_machine_metrics_csv(idle, busy, counts, 1, 4.0, file_m)
    assert file_m.exists() and file_m.stat().st_size > 0

    inst = _inst_two_jobs()
    job_metrics = ut.calc_job_related_metrics([], inst, 0)
    file_j = tmp_path / "jobs.csv"
    ut.generate_per_job_metrics_csv(inst, job_metrics, file_j)
    assert file_j.exists() and file_j.stat().st_size > 0
