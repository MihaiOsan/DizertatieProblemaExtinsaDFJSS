import os
import copy
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
from pathlib import Path

from data_reader import load_instances_from_directory, FJSPInstance, Job, Operation, ETPCConstraint, BaseEvent, \
    BreakdownEvent, AddJobDynamicEvent, CancelJobEvent

###############################################################################
# 0) UTILITARE COMUNE
###############################################################################
TUPLE_FIELDS = {"job": 0, "op": 1, "machine": 2, "start": 3, "end": 4}


def field(op_tuple: Tuple, name: str) -> Any:
    if not isinstance(op_tuple, tuple) or len(op_tuple) < max(TUPLE_FIELDS.values()) + 1:
        return None
    return op_tuple[TUPLE_FIELDS[name]]


###############################################################################
# 2) Metrici
###############################################################################
def metric_makespan(schedule: List[Tuple[int, int, int, float, float]]) -> float:
    valid_ends = [field(op, "end") for op in schedule if field(op, "end") is not None]
    return max(valid_ends) if valid_ends else 0.0


def get_job_completion_times_from_schedule(
        schedule_tuples: List[Tuple]
) -> Dict[int, float]:
    job_completion_times: Dict[int, float] = defaultdict(float)
    for op_tuple in schedule_tuples:
        job_sim_id = field(op_tuple, "job")
        op_end_time = field(op_tuple, "end")
        if job_sim_id is None or op_end_time is None:
            continue
        job_completion_times[job_sim_id] = max(job_completion_times[job_sim_id], float(op_end_time))
    return job_completion_times


def calculate_total_weighted_tardiness(
        schedule_tuples: List[Tuple],
        fjsp_instance: FJSPInstance,
        schedule_makespan: float
) -> float:
    if not fjsp_instance.jobs_defined_in_file: return 0.0
    job_completion_times = get_job_completion_times_from_schedule(schedule_tuples)
    total_weighted_tardiness = 0.0
    for job_obj in fjsp_instance.jobs_defined_in_file:
        if hasattr(job_obj, 'is_cancelled_sim') and job_obj.is_cancelled_sim:
            continue
        completion_time: float
        if job_obj.sim_id in job_completion_times:
            completion_time = job_completion_times[job_obj.sim_id]
        elif job_obj.num_operations > 0:
            completion_time = schedule_makespan
        else:
            completion_time = job_obj.arrival_time
        due_date = job_obj.due_date
        weight = job_obj.weight
        if due_date == float('inf'):
            tardiness = 0.0
        else:
            tardiness = max(0.0, completion_time - due_date)
        total_weighted_tardiness += weight * tardiness
    return total_weighted_tardiness


def calc_machine_idle_time(sched: List[Tuple], num_total_machines: int) -> Tuple[float, float]:
    if not sched: return 0.0, 0.0
    ops_by_m = defaultdict(list)
    max_finish_time = 0.0
    present_machines = set()
    for op_tuple in sched:
        machine_idx = field(op_tuple, "machine")
        op_start = field(op_tuple, "start")
        op_end = field(op_tuple, "end")
        if machine_idx is None or op_start is None or op_end is None: continue
        ops_by_m[machine_idx].append((op_start, op_end))
        present_machines.add(machine_idx)
        if op_end is not None:  # Adaugat verificare None
            max_finish_time = max(max_finish_time, op_end)
    idle_total = 0.0
    for m_idx, ops_list in ops_by_m.items():
        ops_list.sort(key=lambda x: x[0])
        prev_end = 0.0
        for st, en in ops_list:
            idle_total += max(0.0, st - prev_end)
            prev_end = en
        idle_total += max(0.0, max_finish_time - prev_end)
    if num_total_machines > 0:
        machines_with_no_ops = num_total_machines - len(present_machines)
        idle_total += machines_with_no_ops * max_finish_time
        idle_avg = idle_total / num_total_machines
    else:
        idle_avg = 0.0
    return idle_total, idle_avg


def calc_job_waiting_time(sched: List[Tuple], instance: FJSPInstance) -> Tuple[float, float]:
    if not sched: return 0.0, 0.0
    ops_by_j = defaultdict(list)
    scheduled_job_sim_ids = set()
    for op_tuple in sched:
        job_sim_id = field(op_tuple, "job")
        if job_sim_id is None: continue
        ops_by_j[job_sim_id].append(
            (field(op_tuple, "op"), field(op_tuple, "start"), field(op_tuple, "end"))
        )
        scheduled_job_sim_ids.add(job_sim_id)
    wait_total = 0.0
    num_jobs_with_wait_calculated = 0
    for job_sim_id in scheduled_job_sim_ids:
        job_object = instance.get_job_by_sim_id(job_sim_id)
        if not job_object or (hasattr(job_object, 'is_cancelled_sim') and job_object.is_cancelled_sim): continue
        job_ops_in_schedule = ops_by_j.get(job_sim_id, [])
        if not job_ops_in_schedule: continue
        job_ops_in_schedule.sort(key=lambda x: x[0])
        prev_op_end_or_arrival_time = job_object.arrival_time
        job_wait_time_for_this_job = 0.0
        for _op_idx, op_start_time, op_end_time in job_ops_in_schedule:
            if op_start_time is None or op_end_time is None: continue
            job_wait_time_for_this_job += max(0.0, op_start_time - prev_op_end_or_arrival_time)
            prev_op_end_or_arrival_time = op_end_time
        wait_total += job_wait_time_for_this_job
        num_jobs_with_wait_calculated += 1
    wait_avg = wait_total / num_jobs_with_wait_calculated if num_jobs_with_wait_calculated > 0 else 0.0
    return wait_total, wait_avg


def metric_append(store: Dict[str, List[float]], rule_name: str, value: float):
    store.setdefault(rule_name, []).append(value)


def metric_average(store: Dict[str, List[float]]) -> Dict[str, float]:
    return {r_name: (sum(v_list) / len(v_list) if v_list else 0.0) for r_name, v_list in store.items()}


###############################################################################
# 3) Regulile de prioritizare / tie‑break
###############################################################################
def remaining_processing_time(
        job_ops_list_for_sim: List[List[Tuple[int, float]]],
        current_op_idx_arg: int
) -> float:
    total_remaining_time = 0.0
    if 0 <= current_op_idx_arg < len(job_ops_list_for_sim):
        for op_list_idx in range(current_op_idx_arg, len(job_ops_list_for_sim)):
            operation_alternatives = job_ops_list_for_sim[op_list_idx]
            if operation_alternatives:
                total_remaining_time += min(float(p_time) for _m, p_time in operation_alternatives)
    return total_remaining_time


def compute_priority(
        rule_name: str,
        job_sim_id: int,
        op_idx_in_job: int,
        target_machine_id: int,
        processing_time_on_target: float,
        current_simulation_time: float,
        *,
        all_sim_jobs_ops: Dict[int, List[List[Tuple[int, float]]]],
        job_current_progress: Dict[int, int],
        job_arrival_times_map: Dict[int, float],
        job_weights_map: Dict[int, float],
        job_due_dates_map: Dict[int, float],  # <-- ADAUGAT: Due Dates
        current_machine_loads: Dict[int, float]
) -> float:
    ptime_val = float(processing_time_on_target)
    current_job_ops_list = all_sim_jobs_ops.get(job_sim_id)
    job_weight = job_weights_map.get(job_sim_id, 1.0)
    job_due_date = job_due_dates_map.get(job_sim_id, float('inf'))

    if rule_name == "SPT": return ptime_val
    if rule_name == "LPT": return -ptime_val
    if rule_name == "FIFO":
        return job_arrival_times_map.get(job_sim_id, 0.0)
    if rule_name == "LIFO":
        return -job_arrival_times_map.get(job_sim_id, 0.0)
    if rule_name == "SRPT":
        if current_job_ops_list:
            return remaining_processing_time(current_job_ops_list, op_idx_in_job)
        return float('inf')
    if rule_name == "OPR":
        if current_job_ops_list:
            return float(len(current_job_ops_list) - op_idx_in_job)
        return float('inf')
    if rule_name == "ECT":
        if current_job_ops_list:
            rpt_after_current = remaining_processing_time(current_job_ops_list, op_idx_in_job + 1)
            return current_simulation_time + ptime_val + rpt_after_current
        return float('inf')
    if rule_name == "LLM":
        return current_machine_loads.get(target_machine_id, 0.0)
    if rule_name == "Random":
        return random.random()

        # --- Reguli Noi Adaugate ---
    if rule_name == "EDD":  # Earliest Due Date
        return job_due_date  # Valori mai mici (due date mai apropiat) sunt mai prioritare
    if rule_name == "MST":  # Minimum Slack Time
        if job_due_date == float('inf'):
            return float('inf')  # Joburile fara due date au slack infinit (prioritate mica)
        rpt_job = 0.0
        if current_job_ops_list:
            rpt_job = remaining_processing_time(current_job_ops_list, op_idx_in_job)
        slack_time = job_due_date - current_simulation_time - rpt_job
        return slack_time  # Valori mai mici (slack mai small sau negativ) sunt mai prioritare

    return ptime_val  # Fallback la SPT


###############################################################################
# 4) Simulare incrementală REFACTORIZATA PENTRU OOP
###############################################################################
def schedule_dynamic_no_parallel(
        fjsp_instance: FJSPInstance,
        rule_name: str,
        max_simulation_time: float = 200000.0
) -> Tuple[float, List[Tuple[int, int, int, float, float]]]:
    n_machines = fjsp_instance.num_machines
    etpc_map: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = defaultdict(list)
    min_start_due_to_etpc: Dict[Tuple[int, int], float] = defaultdict(float)

    for constr_obj in fjsp_instance.etpc_constraints:
        fore_job_obj = fjsp_instance.get_job_by_original_id(constr_obj.fore_job_orig_id_ref)
        hind_job_obj = fjsp_instance.get_job_by_original_id(constr_obj.hind_job_orig_id_ref)
        if fore_job_obj and hind_job_obj:
            fore_sim_id, fore_op_idx = fore_job_obj.sim_id, constr_obj.fore_op_idx
            hind_sim_id, hind_op_idx = hind_job_obj.sim_id, constr_obj.hind_op_idx
            if not (0 <= fore_op_idx < fore_job_obj.num_operations and \
                    0 <= hind_op_idx < hind_job_obj.num_operations):
                continue
            etpc_map[(fore_sim_id, fore_op_idx)].append((hind_sim_id, hind_op_idx, constr_obj.time_lapse))

    current_jobs_sim_ops: Dict[int, List[List[Tuple[int, float]]]] = {}
    job_arrival_times_map_sim: Dict[int, float] = {}
    job_weights_map_sim: Dict[int, float] = {}
    job_due_dates_map_sim: Dict[int, float] = {}  # <-- ADAUGAT

    for sim_id in fjsp_instance.initial_job_sim_ids:
        job_obj = fjsp_instance.get_job_by_sim_id(sim_id)
        if job_obj:
            ops_list = [[(alt[0], float(alt[1])) for alt in op_obj.alternatives] for op_obj in job_obj.operations]
            current_jobs_sim_ops[sim_id] = ops_list
            job_arrival_times_map_sim[sim_id] = job_obj.arrival_time
            job_weights_map_sim[sim_id] = job_obj.weight
            job_due_dates_map_sim[sim_id] = job_obj.due_date  # <-- ADAUGAT

    bds_per_machine: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    temp_bds = defaultdict(list)
    for bd_event in fjsp_instance.dynamic_event_timeline:
        if isinstance(bd_event, BreakdownEvent):
            temp_bds[bd_event.machine_id].append((bd_event.event_time, bd_event.end_time))
    for m_id_bd_sort in range(n_machines):
        bds_per_machine[m_id_bd_sort] = sorted(temp_bds[m_id_bd_sort], key=lambda x: x[0])

    job_progress_sim: Dict[int, int] = {sim_id: 0 for sim_id in current_jobs_sim_ops.keys()}
    job_current_machine_sim: Dict[int, Optional[int]] = {sim_id: None for sim_id in current_jobs_sim_ops.keys()}
    job_earliest_next_op_start_sim: Dict[int, float] = \
        {sim_id: job_arrival_times_map_sim.get(sim_id, 0.0) for sim_id in current_jobs_sim_ops.keys()}

    effective_op_ready_time_sim: Dict[Tuple[int, int], float] = {}
    for sim_id in current_jobs_sim_ops.keys():
        if current_jobs_sim_ops[sim_id]:
            etpc_min = min_start_due_to_etpc.get((sim_id, 0), 0.0)
            effective_op_ready_time_sim[(sim_id, 0)] = max(job_arrival_times_map_sim.get(sim_id, 0.0), etpc_min)

    active_ops_on_machines: Dict[int, Optional[Tuple[int, int, float, float]]] = {m: None for m in range(n_machines)}
    schedule_output: List[Tuple[int, int, int, float, float]] = []
    t: float = 0.0

    event_timeline_sim_copy = sorted(list(fjsp_instance.dynamic_event_timeline), key=lambda ev: ev.event_time)
    current_event_timeline_idx = 0

    while t < max_simulation_time:
        active_uncompleted_jobs_exist = False
        for sim_id_check, ops_list_check in current_jobs_sim_ops.items():
            job_obj_check = fjsp_instance.get_job_by_sim_id(sim_id_check)
            if job_obj_check and not job_obj_check.is_cancelled_sim and \
                    job_progress_sim.get(sim_id_check, 0) < len(ops_list_check):
                active_uncompleted_jobs_exist = True;
                break
        if not active_uncompleted_jobs_exist and current_event_timeline_idx >= len(event_timeline_sim_copy):
            break

        while current_event_timeline_idx < len(event_timeline_sim_copy) and \
                event_timeline_sim_copy[current_event_timeline_idx].event_time <= t + 1e-9:
            event = event_timeline_sim_copy[current_event_timeline_idx]
            if abs(event.event_time - t) > 1e-9 and event.event_time < t:
                current_event_timeline_idx += 1;
                continue
            current_event_timeline_idx += 1

            if isinstance(event, AddJobDynamicEvent):
                new_job_obj = event.job_object
                new_sim_id = new_job_obj.sim_id
                if new_sim_id not in current_jobs_sim_ops:
                    ops_list_new = [[(alt[0], float(alt[1])) for alt in op_obj.alternatives] for op_obj in
                                    new_job_obj.operations]
                    current_jobs_sim_ops[new_sim_id] = ops_list_new
                    job_progress_sim[new_sim_id] = 0
                    job_current_machine_sim[new_sim_id] = None
                    job_arrival_times_map_sim[new_sim_id] = new_job_obj.arrival_time
                    job_weights_map_sim[new_sim_id] = new_job_obj.weight
                    job_due_dates_map_sim[new_sim_id] = new_job_obj.due_date  # <-- ADAUGAT
                    job_earliest_next_op_start_sim[new_sim_id] = new_job_obj.arrival_time
                    if ops_list_new:
                        etpc_min_new_add = min_start_due_to_etpc.get((new_sim_id, 0), 0.0)
                        effective_op_ready_time_sim[(new_sim_id, 0)] = max(new_job_obj.arrival_time, etpc_min_new_add)
            elif isinstance(event, CancelJobEvent):
                sim_id_to_cancel = event.job_to_cancel_sim_id_mapped
                if sim_id_to_cancel is None:
                    job_to_cancel_obj_lookup = fjsp_instance.get_job_by_original_id(event.job_to_cancel_orig_id_ref)
                    if job_to_cancel_obj_lookup: sim_id_to_cancel = job_to_cancel_obj_lookup.sim_id
                if sim_id_to_cancel is not None and sim_id_to_cancel in current_jobs_sim_ops:
                    job_being_cancelled = fjsp_instance.get_job_by_sim_id(sim_id_to_cancel)
                    if job_being_cancelled and not job_being_cancelled.is_cancelled_sim:
                        job_being_cancelled.is_cancelled_sim = True
                        job_progress_sim[sim_id_to_cancel] = len(current_jobs_sim_ops[sim_id_to_cancel])
                        if job_current_machine_sim.get(sim_id_to_cancel) is not None:
                            active_ops_on_machines[job_current_machine_sim[sim_id_to_cancel]] = None
                            job_current_machine_sim[sim_id_to_cancel] = None

        for m_bd_check in range(n_machines):
            is_breaking_down_now = any(s_bd <= t < e_bd for s_bd, e_bd in bds_per_machine.get(m_bd_check, []))
            if is_breaking_down_now and active_ops_on_machines[m_bd_check] is not None:
                j_b, op_b, _st_b, _rem_b = active_ops_on_machines[m_bd_check]
                active_ops_on_machines[m_bd_check] = None
                if j_b is not None: job_current_machine_sim[j_b] = None
                if j_b is not None: job_earliest_next_op_start_sim[j_b] = t
                if j_b is not None and op_b is not None:
                    etpc_min_interrupted = min_start_due_to_etpc.get((j_b, op_b), 0.0)
                    effective_op_ready_time_sim[(j_b, op_b)] = max(t, etpc_min_interrupted)

        for m_adv in range(n_machines):
            if active_ops_on_machines[m_adv] is not None and \
                    not any(s_bd <= t < e_bd for s_bd, e_bd in bds_per_machine.get(m_adv, [])):
                jop_adv, opidx_adv, st_adv, rem_adv = active_ops_on_machines[m_adv]
                rem_adv -= 1.0
                if rem_adv < 1e-9:
                    finish_time = t + 1.0
                    job_progress_sim[jop_adv] = opidx_adv + 1
                    job_earliest_next_op_start_sim[jop_adv] = finish_time
                    schedule_output.append((jop_adv, opidx_adv, m_adv, st_adv, finish_time))
                    active_ops_on_machines[m_adv] = None
                    job_current_machine_sim[jop_adv] = None
                    if (jop_adv, opidx_adv) in etpc_map:
                        for j_h, o_h, lapse in etpc_map[(jop_adv, opidx_adv)]:
                            new_min_start_for_hind = finish_time + lapse
                            current_min_etpc = min_start_due_to_etpc.get((j_h, o_h), 0.0)
                            min_start_due_to_etpc[(j_h, o_h)] = max(current_min_etpc, new_min_start_for_hind)
                            job_h_obj = fjsp_instance.get_job_by_sim_id(j_h)
                            if job_h_obj and 0 <= o_h < job_h_obj.num_operations:
                                base_ready_for_hind = job_earliest_next_op_start_sim.get(j_h,
                                                                                         job_h_obj.arrival_time) if o_h == 0 else \
                                    job_earliest_next_op_start_sim.get(j_h, float('inf'))
                                effective_op_ready_time_sim[(j_h, o_h)] = max(base_ready_for_hind,
                                                                              min_start_due_to_etpc.get((j_h, o_h),
                                                                                                        0.0))
                else:
                    active_ops_on_machines[m_adv] = (jop_adv, opidx_adv, st_adv, rem_adv)

        current_machine_loads_sim = {
            m_load: (active_ops_on_machines[m_load][3] if active_ops_on_machines[m_load] is not None else 0.0)
            for m_load in range(n_machines)
        }

        for m_dispatch in range(n_machines):
            if active_ops_on_machines[m_dispatch] is None and \
                    not any(s_bd <= t < e_bd for s_bd, e_bd in bds_per_machine.get(m_dispatch, [])):
                best_candidate_dispatch: Optional[Tuple[float, int, int, float]] = None

                for j_cand_sim_id in list(current_jobs_sim_ops.keys()):
                    job_cand_obj_disp = fjsp_instance.get_job_by_sim_id(j_cand_sim_id)
                    if not job_cand_obj_disp or job_cand_obj_disp.is_cancelled_sim: continue

                    current_op_idx_for_job = job_progress_sim.get(j_cand_sim_id, 0)
                    if current_op_idx_for_job >= len(current_jobs_sim_ops.get(j_cand_sim_id, [])): continue
                    if job_current_machine_sim.get(j_cand_sim_id) is not None: continue

                    base_time_cand_disp = job_earliest_next_op_start_sim.get(j_cand_sim_id, t)
                    etpc_min_cand_disp = min_start_due_to_etpc.get((j_cand_sim_id, current_op_idx_for_job), 0.0)
                    current_effective_op_earliest_start = max(base_time_cand_disp, etpc_min_cand_disp)
                    effective_op_ready_time_sim[
                        (j_cand_sim_id, current_op_idx_for_job)] = current_effective_op_earliest_start

                    if t < current_effective_op_earliest_start - 1e-9:
                        continue

                    ops_list_for_cand_job = current_jobs_sim_ops.get(j_cand_sim_id)
                    if not ops_list_for_cand_job or current_op_idx_for_job >= len(ops_list_for_cand_job): continue

                    for m_alt, pt_alt_float in ops_list_for_cand_job[current_op_idx_for_job]:
                        if m_alt == m_dispatch:
                            pt_alt = float(pt_alt_float)
                            if pt_alt < 1e-9: continue
                            pr = compute_priority(
                                rule_name, j_cand_sim_id, current_op_idx_for_job, m_dispatch, pt_alt, t,
                                all_sim_jobs_ops=current_jobs_sim_ops,
                                job_current_progress=job_progress_sim,
                                job_arrival_times_map=job_arrival_times_map_sim,
                                job_weights_map=job_weights_map_sim,
                                job_due_dates_map=job_due_dates_map_sim,  # <-- ADAUGAT
                                current_machine_loads=current_machine_loads_sim
                            )
                            if best_candidate_dispatch is None or pr < best_candidate_dispatch[0]:
                                best_candidate_dispatch = (pr, j_cand_sim_id, current_op_idx_for_job, pt_alt)
                            break

                if best_candidate_dispatch is not None:
                    _prio_sel, j_sel, op_sel, pt_sel = best_candidate_dispatch
                    active_ops_on_machines[m_dispatch] = (j_sel, op_sel, t, pt_sel)
                    job_current_machine_sim[j_sel] = m_dispatch

        t += 1.0

    makespan_val = max(op_tuple[TUPLE_FIELDS["end"]] for op_tuple in schedule_output) if schedule_output else t
    active_uncompleted_at_end = False
    for j_check_end_sim_id, ops_list_end in current_jobs_sim_ops.items():
        job_obj_end = fjsp_instance.get_job_by_sim_id(j_check_end_sim_id)
        if job_obj_end and not job_obj_end.is_cancelled_sim and \
                job_progress_sim.get(j_check_end_sim_id, 0) < len(ops_list_end):
            active_uncompleted_at_end = True;
            break

    if t >= max_simulation_time - 1e-9 and active_uncompleted_at_end:
        makespan_val = max(makespan_val, max_simulation_time)

    return makespan_val, schedule_output


###############################################################################
# 5) Plot Gantt
###############################################################################
def plot_gantt(schedule: List[Tuple[int, int, int, float, float]],
               n_machines: int,
               breakdowns: Dict[int, List[Tuple[float, float]]],
               title: str = "Gantt Chart",
               save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    actual_makespan_plot = 0
    if schedule:
        valid_ends = [field(op, "end") for op in schedule if field(op, "end") is not None]
        if valid_ends:
            actual_makespan_plot = max(valid_ends)
    if not actual_makespan_plot and breakdowns:
        for bd_list in breakdowns.values():
            for _, bd_e in bd_list:
                actual_makespan_plot = max(actual_makespan_plot, bd_e)
    if actual_makespan_plot == 0: actual_makespan_plot = 10
    for m in range(n_machines):
        for s_bd, e_bd in breakdowns.get(m, []):
            if s_bd < actual_makespan_plot:
                ax.barh(m, e_bd - s_bd, left=s_bd, height=0.9, color="lightcoral", alpha=0.6, edgecolor="maroon",
                        hatch='///')
    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        try:
            cmap = plt.cm.get_cmap("tab20", 20)
        except ValueError:
            cmap = plt.cm.get_cmap("viridis", 20)
    if cmap is None: cmap = plt.cm.get_cmap("viridis", 20)
    job_colors = {}
    color_idx = 0
    for op_tuple_plot in schedule:
        job_id = field(op_tuple_plot, "job")
        op_idx = field(op_tuple_plot, "op")
        machine_id = field(op_tuple_plot, "machine")
        start = field(op_tuple_plot, "start")
        end = field(op_tuple_plot, "end")
        if None in [job_id, op_idx, machine_id, start, end]: continue
        if job_id not in job_colors:
            job_colors[job_id] = cmap(color_idx % cmap.N)
            color_idx += 1
        duration = end - start
        if duration < 1e-9: continue
        ax.barh(machine_id, duration, left=start, color=job_colors[job_id],
                edgecolor="black", height=0.7, alpha=0.9)
        ax.text(start + duration / 2, machine_id, f"J{job_id}.{op_idx}",
                ha="center", va="center", color="black", fontsize=6, fontweight='bold')
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Machine", fontsize=10)
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"M{i}" for i in range(n_machines)], fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    if any(breakdowns.values()):
        breakdown_patch = mpatches.Patch(facecolor="lightcoral", alpha=0.6, edgecolor="maroon", hatch='///',
                                         label='Breakdown')
        ax.legend(handles=[breakdown_patch], fontsize=8, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


###############################################################################
# 6) MAIN – evaluare reguli (Refactorizat pentru OOP)
###############################################################################
if __name__ == "__main__":
    INPUT_DIR_CLASSIC_PATH = Path("/Users/mihaiosan/PycharmProjects/DizertatieProblemaExtinsaDFJSS/dynamic_data/fan21/test_sets")
    OUTPUT_DIR_CLASSIC_PATH = Path("rezultate/simplu/clasic/gantt")
    RESULTS_DIR_PATH = Path("rezultate/simplu/clasic/gantt")

    OUTPUT_DIR_CLASSIC_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # --- MODIFICARE: Lista de reguli extinsa ---
    RULES = ["SPT", "LPT", "FIFO", "LIFO", "SRPT", "OPR", "ECT", "LLM", "Random", "EDD", "MST"]  # <-- ADAUGAT EDD, MST

    avg_ms_per_rule = {r: [] for r in RULES}
    avg_twt_per_rule = {r: [] for r in RULES}
    avg_time_per_rule = {r: [] for r in RULES}
    avg_idle_per_rule = {r: [] for r in RULES}
    avg_wait_per_rule = {r: [] for r in RULES}

    RESULTS_FILE_CLASSIC_PATH = RESULTS_DIR_PATH / "classic_oop_main_results_ext_v2.txt"  # Nume fisier nou

    with open(RESULTS_FILE_CLASSIC_PATH, "w", encoding="utf-8") as fout:
        all_loaded_instances: List[FJSPInstance] = load_instances_from_directory(str(INPUT_DIR_CLASSIC_PATH))

        if not all_loaded_instances:
            print(f"No instances found in {INPUT_DIR_CLASSIC_PATH}. Exiting.")
        else:
            print(f"Loaded {len(all_loaded_instances)} instances for classic heuristics evaluation.")

        for instance_obj in all_loaded_instances:
            fout.write(
                f"\n=== INSTANCE: {instance_obj.file_name} (Machines: {instance_obj.num_machines}, Total Jobs Defined: {instance_obj.num_total_defined_jobs}) ===\n")
            print(f"\n=== INSTANCE: {instance_obj.file_name} ===")

            for rule_name_iter in RULES:
                time_start_rule = time.perf_counter()

                makespan, schedule_result = schedule_dynamic_no_parallel(
                    fjsp_instance=instance_obj,
                    rule_name=rule_name_iter
                )
                elapsed_rule = time.perf_counter() - time_start_rule

                _total_idle_val, avg_idle_val = calc_machine_idle_time(schedule_result, instance_obj.num_machines)
                _total_wait_val, avg_wait_val = calc_job_waiting_time(schedule_result, instance_obj)
                twt_val = calculate_total_weighted_tardiness(schedule_result, instance_obj, makespan)

                metric_append(avg_ms_per_rule, rule_name_iter, makespan)
                metric_append(avg_twt_per_rule, rule_name_iter, twt_val)
                metric_append(avg_time_per_rule, rule_name_iter, elapsed_rule)
                metric_append(avg_idle_per_rule, rule_name_iter, avg_idle_val)
                metric_append(avg_wait_per_rule, rule_name_iter, avg_wait_val)

                fout.write(
                    f"  {rule_name_iter:<7} => MS={makespan:<7.2f}, TWT={twt_val:<8.2f}, IdleAvg={avg_idle_val:<6.2f}, WaitAvg={avg_wait_val:<6.2f}, T={elapsed_rule:.3f}s\n")
                print(
                    f"    {rule_name_iter:<7} => MS={makespan:<7.2f}, TWT={twt_val:<8.2f}, IdleAvg={avg_idle_val:<6.2f}, WaitAvg={avg_wait_val:<6.2f}, T={elapsed_rule:.3f}s")

                breakdowns_for_gantt = defaultdict(list)
                for bd_event in instance_obj.dynamic_event_timeline:
                    if isinstance(bd_event, BreakdownEvent):
                        breakdowns_for_gantt[bd_event.machine_id].append((bd_event.event_time, bd_event.end_time))

                save_gantt_path = OUTPUT_DIR_CLASSIC_PATH / f"{Path(instance_obj.file_name).stem}_{rule_name_iter}.png"
                plot_gantt(
                    schedule_result,
                    instance_obj.num_machines,
                    breakdowns_for_gantt,
                    title=f"{instance_obj.file_name} - {rule_name_iter} (MS={makespan:.2f}, TWT={twt_val:.2f})",
                    save_path=str(save_gantt_path)
                )

        fout.write("\n\n=== Average Performance per Rule (across all instances) ===\n")
        print("\n=== Average Performance per Rule (across all instances) ===")

        final_avg_ms = metric_average(avg_ms_per_rule)
        final_avg_twt = metric_average(avg_twt_per_rule)
        final_avg_time = metric_average(avg_time_per_rule)
        final_avg_idle = metric_average(avg_idle_per_rule)
        final_avg_wait = metric_average(avg_wait_per_rule)

        header_line = f"{'Rule':<7}: {'AvgMS':<7}, {'AvgTWT':<8}, {'AvgIdle':<7}, {'AvgWait':<7}, {'AvgT (s)':<7}"
        fout.write(header_line + "\n")
        print(header_line)
        for r_name_avg in RULES:
            fout.write(
                f"{r_name_avg:<7}: {final_avg_ms.get(r_name_avg, 0.0):<7.2f}, {final_avg_twt.get(r_name_avg, 0.0):<8.2f}, {final_avg_idle.get(r_name_avg, 0.0):<7.2f}, {final_avg_wait.get(r_name_avg, 0.0):<7.2f}, {final_avg_time.get(r_name_avg, 0.0):<7.3f}\n")
            print(
                f"{r_name_avg:<7}: {final_avg_ms.get(r_name_avg, 0.0):<7.2f}, {final_avg_twt.get(r_name_avg, 0.0):<8.2f}, {final_avg_idle.get(r_name_avg, 0.0):<7.2f}, {final_avg_wait.get(r_name_avg, 0.0):<7.2f}, {final_avg_time.get(r_name_avg, 0.0):<7.3f}")

    print(f"\nRezultatele pentru euristicile clasice au fost scrise în {RESULTS_FILE_CLASSIC_PATH}")
    print(f"Graficele Gantt pentru euristicile clasice se află în directorul '{OUTPUT_DIR_CLASSIC_PATH}'")

