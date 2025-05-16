import copy
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

from data_reader import FJSPInstance, Job, Operation, ETPCConstraint, BaseEvent, BreakdownEvent, AddJobDynamicEvent, \
    CancelJobEvent


# Clasa MachineState ramane neschimbata
class MachineState:
    """
    Clasă simplă pentru reținerea stării unei mașini.
    """

    def __init__(self, machine_id: int):
        self.id: int = machine_id
        self.busy: bool = False
        self.job_id: Optional[int] = None  # Va stoca sim_id-ul jobului
        self.op_idx: Optional[int] = None  # Indexul operatiei in cadrul jobului
        self.time_remaining: float = 0.0
        self.broken_until: float = 0.0
        self.start_time: float = 0.0
        self.idle_since: float = 0.0


def evaluate_individual(
        individual: Any,
        fjsp_instance: FJSPInstance,
        toolbox: Any,
        max_time: float = 999999.0
) -> Tuple[float, List[Tuple[int, int, int, float, float]]]:
    """
    Rulează simularea discretă a FJSP folosind o abordare OOP și 12 terminale GP.
    """
    MAX_TIME_LIMIT_SAFETY = 200000.0
    dispatch_rule_callable: Any  # Tip pentru functia compilata

    if not callable(individual):
        try:
            dispatch_rule_callable = toolbox.compile(expr=individual)
        except Exception as e:
            print(f"Error compiling GP individual: {e}. Individual: {str(individual)}")
            return float('inf'), []
    else:
        dispatch_rule_callable = individual

    num_machines = fjsp_instance.num_machines
    current_jobs_sim_map: Dict[int, Job] = {}
    for sim_id in fjsp_instance.initial_job_sim_ids:
        job_obj = fjsp_instance.get_job_by_sim_id(sim_id)
        if job_obj:
            sim_job_copy = copy.copy(job_obj)
            sim_job_copy.current_op_idx_sim = 0
            sim_job_copy.completion_time_sim = None
            sim_job_copy.is_cancelled_sim = False
            current_jobs_sim_map[sim_id] = sim_job_copy
        # else:
        # print(f"Warning (Scheduler Init): Initial job with sim_id {sim_id} not found.")

    event_timeline_sim = list(fjsp_instance.dynamic_event_timeline)

    machines = [MachineState(m) for m in range(num_machines)]
    job_end_time_sim: Dict[int, float] = {
        sim_id: 0.0 for sim_id in current_jobs_sim_map.keys()
    }

    ready_ops: set[Tuple[int, int]] = set()
    effective_ready_time: Dict[Tuple[int, int], float] = {}
    job_internal_pred_finish_time: Dict[Tuple[int, int], float] = {}  # Adaugat pentru ETPC_D

    etpc_hind_map: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = defaultdict(list)
    etpc_min_start_for_hind: Dict[Tuple[int, int], float] = defaultdict(float)

    for etpc_constr_obj in fjsp_instance.etpc_constraints:
        fore_job_obj = fjsp_instance.get_job_by_original_id(etpc_constr_obj.fore_job_orig_id_ref)
        hind_job_obj = fjsp_instance.get_job_by_original_id(etpc_constr_obj.hind_job_orig_id_ref)
        if fore_job_obj and hind_job_obj:
            if not (0 <= etpc_constr_obj.fore_op_idx < fore_job_obj.num_operations and \
                    0 <= etpc_constr_obj.hind_op_idx < hind_job_obj.num_operations):
                continue
            fore_key = (fore_job_obj.sim_id, etpc_constr_obj.fore_op_idx)
            etpc_hind_map[fore_key].append(
                (hind_job_obj.sim_id, etpc_constr_obj.hind_op_idx, etpc_constr_obj.time_lapse)
            )

    for sim_id, job_obj in current_jobs_sim_map.items():
        if job_obj.num_operations > 0:
            # Timpul de sosire este baza pentru prima operatie
            job_internal_pred_finish_time[(sim_id, 0)] = job_obj.arrival_time
            etpc_min = etpc_min_start_for_hind.get((sim_id, 0), 0.0)
            effective_ready_time[(sim_id, 0)] = max(job_obj.arrival_time, etpc_min)
            ready_ops.add((sim_id, 0))

    cancelled_sim_ids: set[int] = set()
    event_timeline_idx = 0
    current_time = 0.0
    completed_total_ops = 0
    current_total_ops = 0
    for job_obj_init in current_jobs_sim_map.values():
        if not job_obj_init.is_cancelled_sim:
            current_total_ops += job_obj_init.num_operations

    schedule_output: List[Tuple[int, int, int, float, float]] = []
    rpt_cache: Dict[Tuple[int, int], float] = {}

    def get_job_object_from_sim(sim_id: int) -> Optional[Job]:
        return current_jobs_sim_map.get(sim_id)

    def make_op_ready_sim(job_sim_id: int, op_idx: int, internal_pred_finish_time_val: float):
        job_obj = get_job_object_from_sim(job_sim_id)
        if not job_obj or op_idx >= job_obj.num_operations: return

        job_internal_pred_finish_time[(job_sim_id, op_idx)] = float(internal_pred_finish_time_val)
        etpc_min_val = etpc_min_start_for_hind.get((job_sim_id, op_idx), 0.0)
        actual_eff_ready_time = max(float(internal_pred_finish_time_val), float(etpc_min_val))
        effective_ready_time[(job_sim_id, op_idx)] = actual_eff_ready_time
        ready_ops.add((job_sim_id, op_idx))

    def compute_rpt_sim(job_sim_id: int, from_op_idx: int) -> float:
        key = (job_sim_id, from_op_idx)
        if key in rpt_cache: return rpt_cache[key]
        job_obj = get_job_object_from_sim(job_sim_id)
        if not job_obj or from_op_idx >= job_obj.num_operations:
            rpt_cache[key] = 0.0;
            return 0.0
        s = 0.0
        for op_i in range(from_op_idx, job_obj.num_operations):
            op_obj = job_obj.get_operation(op_i)
            if op_obj and op_obj.alternatives: s += op_obj.get_best_processing_time()
        rpt_cache[key] = s;
        return s

    while current_time < max_time and current_time < MAX_TIME_LIMIT_SAFETY:
        while event_timeline_idx < len(event_timeline_sim) and \
                event_timeline_sim[event_timeline_idx].event_time <= current_time + 1e-9:
            event = event_timeline_sim[event_timeline_idx]
            if abs(event.event_time - current_time) > 1e-9 and event.event_time < current_time:
                event_timeline_idx += 1;
                continue
            event_timeline_idx += 1

            if isinstance(event, BreakdownEvent):
                machine = machines[event.machine_id]
                machine.broken_until = max(machine.broken_until, event.end_time)
                if machine.busy and machine.start_time < machine.broken_until:
                    interrupted_job_sim_id = machine.job_id
                    interrupted_op_idx = machine.op_idx
                    if interrupted_job_sim_id is not None and interrupted_op_idx is not None:
                        make_op_ready_sim(interrupted_job_sim_id, interrupted_op_idx, current_time)
                    machine.busy = False;
                    machine.job_id = None;
                    machine.op_idx = None
                    machine.time_remaining = 0.0;
                    machine.start_time = 0.0
                    machine.idle_since = current_time
            elif isinstance(event, AddJobDynamicEvent):
                new_job_obj_from_event = event.job_object
                if new_job_obj_from_event.sim_id not in current_jobs_sim_map:
                    sim_job_copy_added = copy.copy(new_job_obj_from_event)
                    sim_job_copy_added.current_op_idx_sim = 0
                    sim_job_copy_added.completion_time_sim = None
                    sim_job_copy_added.is_cancelled_sim = False
                    current_jobs_sim_map[new_job_obj_from_event.sim_id] = sim_job_copy_added
                    job_end_time_sim[new_job_obj_from_event.sim_id] = 0.0
                    if not sim_job_copy_added.is_cancelled_sim:
                        current_total_ops += sim_job_copy_added.num_operations
                    if sim_job_copy_added.num_operations > 0:
                        # Timpul de sosire este event.event_time (care e current_time aici)
                        job_internal_pred_finish_time[(sim_job_copy_added.sim_id, 0)] = event.event_time
                        etpc_min_add = etpc_min_start_for_hind.get((sim_job_copy_added.sim_id, 0), 0.0)
                        effective_ready_time[(sim_job_copy_added.sim_id, 0)] = max(event.event_time, etpc_min_add)
                        ready_ops.add((sim_job_copy_added.sim_id, 0))
            elif isinstance(event, CancelJobEvent):
                sim_id_to_cancel = event.job_to_cancel_sim_id_mapped
                if sim_id_to_cancel is None:
                    job_to_cancel_obj_lookup = fjsp_instance.get_job_by_original_id(event.job_to_cancel_orig_id_ref)
                    if job_to_cancel_obj_lookup: sim_id_to_cancel = job_to_cancel_obj_lookup.sim_id
                if sim_id_to_cancel is not None and sim_id_to_cancel not in cancelled_sim_ids:
                    job_being_cancelled = get_job_object_from_sim(sim_id_to_cancel)
                    if job_being_cancelled and not job_being_cancelled.is_cancelled_sim:
                        job_being_cancelled.is_cancelled_sim = True
                        cancelled_sim_ids.add(sim_id_to_cancel)
                        for m_cancel in machines:
                            if m_cancel.busy and m_cancel.job_id == sim_id_to_cancel:
                                m_cancel.busy = False;
                                m_cancel.job_id = None;
                                m_cancel.op_idx = None
                                m_cancel.time_remaining = 0.0;
                                m_cancel.start_time = 0.0
                                m_cancel.idle_since = current_time
                        ready_ops = {(j_ro, o_ro) for (j_ro, o_ro) in ready_ops if j_ro != sim_id_to_cancel}
                        ops_already_done_for_cancelled = 0
                        for sched_j, sched_op_idx, _, _, _ in schedule_output:
                            if sched_j == sim_id_to_cancel: ops_already_done_for_cancelled = sched_op_idx + 1
                        ops_to_remove_from_total = job_being_cancelled.num_operations - ops_already_done_for_cancelled
                        if ops_to_remove_from_total > 0: current_total_ops -= ops_to_remove_from_total
            else:
                break

        for machine in machines:
            if machine.broken_until > current_time + 1e-9: continue
            if abs(machine.broken_until - current_time) < 1e-9 and machine.broken_until != 0.0:
                machine.broken_until = 0.0
                if not machine.busy: machine.idle_since = current_time
            if machine.busy:
                machine.time_remaining -= 1.0
                if machine.time_remaining < 1e-9:
                    job_sim_id_done = machine.job_id
                    op_idx_done = machine.op_idx
                    op_start_time = machine.start_time
                    op_end_time = current_time + 1.0
                    machine.busy = False;
                    machine.job_id = None;
                    machine.op_idx = None
                    machine.time_remaining = 0.0;
                    machine.start_time = 0.0
                    machine.idle_since = op_end_time
                    completed_total_ops += 1
                    job_end_time_sim[job_sim_id_done] = op_end_time
                    schedule_output.append((job_sim_id_done, op_idx_done, machine.id, op_start_time, op_end_time))
                    if (job_sim_id_done, op_idx_done) in etpc_hind_map:
                        for hj_sim, ho_idx, tl_val in etpc_hind_map[(job_sim_id_done, op_idx_done)]:
                            new_min_start = op_end_time + tl_val
                            current_min = etpc_min_start_for_hind.get((hj_sim, ho_idx), 0.0)
                            etpc_min_start_for_hind[(hj_sim, ho_idx)] = max(current_min, new_min_start)
                            if (hj_sim, ho_idx) in job_internal_pred_finish_time:  # Daca op hind e "job-ready"
                                base_jprd_time_etpc = job_internal_pred_finish_time[(hj_sim, ho_idx)]
                                effective_ready_time[(hj_sim, ho_idx)] = max(base_jprd_time_etpc,
                                                                             etpc_min_start_for_hind[(hj_sim, ho_idx)])
                    job_done_obj = get_job_object_from_sim(job_sim_id_done)
                    if job_done_obj and op_idx_done + 1 < job_done_obj.num_operations and job_sim_id_done not in cancelled_sim_ids:
                        make_op_ready_sim(job_sim_id_done, op_idx_done + 1, op_end_time)

        for machine in machines:
            if not machine.busy and machine.broken_until <= current_time + 1e-9:
                best_candidate: Optional[Tuple[float, int, int, float]] = None
                MW_val = (current_time + 1.0) - machine.idle_since
                WIP_val = sum(1 for m_wip in machines if m_wip.busy)
                TUF_val = max(0.0, machine.broken_until - (current_time + 1.0))
                ready_ops_copy = list(ready_ops)
                for job_sim_id_cand, op_idx_cand in ready_ops_copy:
                    if job_sim_id_cand in cancelled_sim_ids:
                        if (job_sim_id_cand, op_idx_cand) in ready_ops: ready_ops.remove((job_sim_id_cand, op_idx_cand))
                        continue
                    job_cand_obj = get_job_object_from_sim(job_sim_id_cand)
                    if not job_cand_obj or op_idx_cand >= job_cand_obj.num_operations:
                        if (job_sim_id_cand, op_idx_cand) in ready_ops: ready_ops.remove((job_sim_id_cand, op_idx_cand))
                        continue

                        # Asiguram ca effective_ready_time e calculat
                    base_ready_cand = job_internal_pred_finish_time.get((job_sim_id_cand, op_idx_cand),
                                                                        job_cand_obj.arrival_time)
                    etpc_min_cand_val = etpc_min_start_for_hind.get((job_sim_id_cand, op_idx_cand), 0.0)
                    op_eff_ready_t_cand = max(base_ready_cand, etpc_min_cand_val)
                    effective_ready_time[(job_sim_id_cand, op_idx_cand)] = op_eff_ready_t_cand  # Stocam/Actualizam

                    if op_eff_ready_t_cand > current_time + 1.0 - 1e-9: continue

                    operation_cand_obj = job_cand_obj.get_operation(op_idx_cand)
                    if not operation_cand_obj: continue
                    ptime_cand_on_machine = operation_cand_obj.get_processing_time(machine.id)

                    if ptime_cand_on_machine is not None and ptime_cand_on_machine > 1e-9:
                        PT_val = ptime_cand_on_machine
                        RO_val = float(job_cand_obj.num_operations - (op_idx_cand + 1))
                        TQ_val = max(0.0, (current_time + 1.0) - op_eff_ready_t_cand)
                        RPT_val = compute_rpt_sim(job_sim_id_cand, op_idx_cand)
                        WJ_val = job_cand_obj.weight
                        DD_val = job_cand_obj.due_date
                        SLK_val = DD_val - (current_time + 1.0) - RPT_val if DD_val != float('inf') else float('inf')

                        # ETPC_D: Cat de mult intarzie ETPC fata de disponibilitatea din job
                        job_internal_ready_time_cand = job_internal_pred_finish_time.get((job_sim_id_cand, op_idx_cand),
                                                                                         job_cand_obj.arrival_time)
                        ETPC_D_val = max(0.0, op_eff_ready_t_cand - job_internal_ready_time_cand)

                        # N_ETPC_S: Numarul de succesori ETPC directi
                        N_ETPC_S_val = float(len(etpc_hind_map.get((job_sim_id_cand, op_idx_cand), [])))

                        try:
                            priority = dispatch_rule_callable(  # Folosim functia compilata
                                PT_val, RO_val, MW_val, TQ_val, WIP_val, RPT_val,
                                TUF_val, DD_val, SLK_val, WJ_val,
                                ETPC_D_val, N_ETPC_S_val  # Argumente noi
                            )
                        except Exception as e_dispatch_rule:
                            priority = float('inf')

                        if best_candidate is None or priority < best_candidate[0]:
                            best_candidate = (priority, job_sim_id_cand, op_idx_cand, PT_val)

                if best_candidate:
                    _prio, j_s, o_s, pt_s = best_candidate
                    machine.busy = True;
                    machine.job_id = j_s;
                    machine.op_idx = o_s
                    machine.time_remaining = pt_s;
                    machine.start_time = current_time + 1.0
                    if (j_s, o_s) in ready_ops: ready_ops.remove((j_s, o_s))

        is_simulation_finished = False
        if completed_total_ops >= current_total_ops:
            all_active_jobs_completed_in_schedule = True
            for sim_id_check, job_obj_check in current_jobs_sim_map.items():
                if sim_id_check not in cancelled_sim_ids and job_obj_check.num_operations > 0:
                    last_op_idx_check = job_obj_check.num_operations - 1
                    found_last_op_in_schedule = any(
                        sched_j == sim_id_check and sched_o == last_op_idx_check
                        for sched_j, sched_o, _, _, _ in schedule_output
                    )
                    if not found_last_op_in_schedule:
                        all_active_jobs_completed_in_schedule = False;
                        break
            future_add_job_events = any(
                isinstance(event_timeline_sim[i], AddJobDynamicEvent)
                for i in range(event_timeline_idx, len(event_timeline_sim))
            )
            if all_active_jobs_completed_in_schedule and not ready_ops and not future_add_job_events:
                is_simulation_finished = True
        if is_simulation_finished: break
        current_time += 1.0

    final_makespan = 0.0
    active_job_present_at_end = any(
        sim_id_mk not in cancelled_sim_ids and job_obj_mk.num_operations > 0
        for sim_id_mk, job_obj_mk in current_jobs_sim_map.items()
    )
    if not schedule_output and active_job_present_at_end:
        final_makespan = float(MAX_TIME_LIMIT_SAFETY) if current_time >= MAX_TIME_LIMIT_SAFETY - 1e-9 else float(
            max_time)
    else:
        for sim_id_mk in current_jobs_sim_map.keys():
            if sim_id_mk not in cancelled_sim_ids:
                final_makespan = max(final_makespan, job_end_time_sim.get(sim_id_mk, 0.0))
    if final_makespan == 0.0 and active_job_present_at_end and current_time >= MAX_TIME_LIMIT_SAFETY - 1e-9:
        final_makespan = float(MAX_TIME_LIMIT_SAFETY)

    return final_makespan, schedule_output

