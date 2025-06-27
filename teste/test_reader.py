# tests/test_reader_setmic_json.py
"""
Teste unitare dedicate fișierului JSON existent: setmic.json
Necesită:
  * data_reader.py în PYTHONPATH
  * fișierul setmic.json situat în rădăcina repo-ului (același nivel cu data_reader.py)
"""

from pathlib import Path
import pytest
import data_reader as dr


# ---------- fixture ce încarcă instanța ---------------- #

@pytest.fixture(scope="module")
def instance():
    json_path = Path(__file__).resolve().parent.parent / "dynamic_data/extended/test_sets_micro/test_small_flex_events_0_util0.75_ec0.08_nm10_v0.json"
    assert json_path.exists(), f"Nu găsesc setmic.json la: {json_path}"
    inst = dr.read_dynamic_fjsp_instance_json(str(json_path))
    assert inst is not None, "Parserul a întors None"
    return inst


# ---------- teste de bază ------------------------------ #

def test_basic_counts(instance):
    """Verifică numărul de joburi și de mașini detectate."""
    assert instance.num_total_defined_jobs == 15
    assert instance.num_machines == 10

def test_initial_vs_dynamic_jobs(instance):
    """Jobs cu arrival_time 0 trebuie să fie inițiale (10)"""
    assert instance.num_initial_jobs_in_sim == 10
    # toți cei inițiali au arrival_time 0
    for sid in instance.initial_job_sim_ids:
        job = instance.get_job_by_sim_id(sid)
        assert job.arrival_time == 0

def test_etpc_loaded(instance):
    """Există exact o constrângere ETPC, cu valorile corecte."""
    assert len(instance.etpc_constraints) == 1
    etpc = instance.etpc_constraints[0]
    assert etpc.fore_job_orig_id_ref == 2
    assert etpc.hind_job_orig_id_ref == 8
    assert etpc.time_lapse == 7

def test_breakdowns_presence_and_validity(instance):
    """Cel puțin un breakdown pe fiecare mașină definită; times valide."""
    # grupăm breakdown-urile pe mașină
    bd_by_machine = {}
    for bd in instance.breakdown_events:
        assert bd.event_time < bd.end_time
        bd_by_machine.setdefault(bd.machine_id, 0)
        bd_by_machine[bd.machine_id] += 1
    # trebuie să existe chei pentru toate mașinile menționate în JSON (0-9)
    assert set(bd_by_machine.keys()) == set(range(10))
    # fiecare listă are cel puțin 1 eveniment
    assert all(cnt > 0 for cnt in bd_by_machine.values())

def test_cancel_event(instance):
    """Există un singur eveniment de tip cancel pentru jobul 13 la t=1065."""
    cancels = [ev for ev in instance.dynamic_event_timeline
               if isinstance(ev, dr.CancelJobEvent)]
    assert len(cancels) == 1
    cancel_ev = cancels[0]
    assert cancel_ev.job_to_cancel_orig_id_ref == 13
    assert cancel_ev.event_time == 1065

def test_timeline_sorted(instance):
    """Evenimentele din timeline trebuie să fie sortate crescător după timp."""
    times = [ev.event_time for ev in instance.dynamic_event_timeline]
    assert times == sorted(times)
