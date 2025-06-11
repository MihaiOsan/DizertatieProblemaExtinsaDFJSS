import copy
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
from data_reader import FJSPInstance, Job, Operation, ETPCConstraint, BaseEvent, BreakdownEvent, AddJobDynamicEvent, \
    CancelJobEvent


class MachineState:
    """
    Clasă simplă pentru reținerea stării unei mașini.
    """

    def __init__(self, machine_id: int):
        self.id: int = machine_id
        self.busy: bool = False
        self.job_id: Optional[int] = None  # Va stoca sim_id-ul jobului
        self.op_idx: Optional[int] = None  # Indexul operației în cadrul jobului
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

    Args:
        individual: Regula de dispatch compilată sau o expresie care poate fi compilată.
        fjsp_instance: Instanța problemei FJSP care conține datele despre joburi, mașini și evenimente.
        toolbox: Obiectul DEAP toolbox, folosit pentru a compila individul GP.
        max_time: Timpul maxim până la care rulează simularea.

    Returns:
        Un tuplu conținând:
        - makespan-ul final (timpul de finalizare al ultimului job).
        - o listă de tupluri, fiecare reprezentând o operație programată:
          (job_sim_id, op_idx, machine_id, start_time, end_time).
    """

    SIMULATION_MAX_TIME_LIMIT = max_time
    dispatch_rule_callable: Any

    # Compilarea regulii de dispatch
    if not callable(individual):
        try:
            dispatch_rule_callable = toolbox.compile(expr=individual)
        except Exception as e:
            print(f"Eroare la compilarea individului GP: {e}. Individ: {str(individual)}")
            return float('inf'), []
    else:
        dispatch_rule_callable = individual

    # --- Inițializarea stării simulării ---
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

    event_timeline_sim = list(fjsp_instance.dynamic_event_timeline)
    machines = [MachineState(m) for m in range(num_machines)]

    # Calcul inițial pentru numărul total de operații active.
    initial_total_ops_count = sum(job_obj_init.num_operations for job_obj_init in current_jobs_sim_map.values() if
                                  not job_obj_init.is_cancelled_sim)

    # Definirea unei clase de context pentru a gestiona starea simulării
    class SimulationContext:
        def __init__(self, initial_current_total_ops: int):
            self.current_time: float = 0.0
            self.event_timeline_idx: int = 0
            self.completed_total_ops: int = 0
            self.current_total_ops: int = initial_current_total_ops

            self.job_end_time_sim: Dict[int, float] = {
                sim_id: 0.0 for sim_id in current_jobs_sim_map.keys()
            }
            self.ready_ops: set[Tuple[int, int]] = set()
            self.effective_ready_time: Dict[Tuple[int, int], float] = {}
            self.job_internal_pred_finish_time: Dict[Tuple[int, int], float] = {}
            self.etpc_hind_map: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = defaultdict(list)
            self.etpc_min_start_for_hind: Dict[Tuple[int, int], float] = defaultdict(float)
            self.cancelled_sim_ids: set[int] = set()
            self.schedule_output: List[Tuple[int, int, int, float, float]] = []
            self.rpt_cache: Dict[Tuple[int, int], float] = {}

    context = SimulationContext(initial_total_ops_count)

    # Procesarea constrângerilor ETPC (transferată în contextul inițial)
    for etpc_constr_obj in fjsp_instance.etpc_constraints:
        fore_job_obj = fjsp_instance.get_job_by_original_id(etpc_constr_obj.fore_job_orig_id_ref)
        hind_job_obj = fjsp_instance.get_job_by_original_id(etpc_constr_obj.hind_job_orig_id_ref)

        if fore_job_obj and hind_job_obj:
            if not (0 <= etpc_constr_obj.fore_op_idx < fore_job_obj.num_operations and
                    0 <= etpc_constr_obj.hind_op_idx < hind_job_obj.num_operations):
                continue
            fore_key = (fore_job_obj.sim_id, etpc_constr_obj.fore_op_idx)
            context.etpc_hind_map[fore_key].append(
                (hind_job_obj.sim_id, etpc_constr_obj.hind_op_idx, etpc_constr_obj.time_lapse)
            )

    # Inițializarea operațiilor gata (transferată în contextul inițial)
    for sim_id, job_obj in current_jobs_sim_map.items():
        if job_obj.num_operations > 0:
            context.job_internal_pred_finish_time[(sim_id, 0)] = job_obj.arrival_time
            etpc_min = context.etpc_min_start_for_hind.get((sim_id, 0), 0.0)
            context.effective_ready_time[(sim_id, 0)] = max(job_obj.arrival_time, etpc_min)
            context.ready_ops.add((sim_id, 0))

    # --- Funcții helper imbricate ---
    def get_job_object_from_sim(sim_id: int) -> Optional[Job]:
        """Funcție helper pentru a obține un obiect Job din harta simulării."""
        return current_jobs_sim_map.get(sim_id)

    def make_op_ready_sim(job_sim_id: int, op_idx: int, internal_pred_finish_time_val: float, current_sim_time: float):
        """
        Marchează o operație ca fiind gata de procesare și actualizează timpii de disponibilitate.
        """
        job_obj = get_job_object_from_sim(job_sim_id)
        if not job_obj or op_idx >= job_obj.num_operations:
            return

        context.job_internal_pred_finish_time[(job_sim_id, op_idx)] = internal_pred_finish_time_val
        etpc_min_val = context.etpc_min_start_for_hind.get((job_sim_id, op_idx), 0.0)
        actual_eff_ready_time = max(internal_pred_finish_time_val, etpc_min_val)
        context.effective_ready_time[(job_sim_id, op_idx)] = actual_eff_ready_time
        context.ready_ops.add((job_sim_id, op_idx))

    def compute_rpt_sim(job_sim_id: int, from_op_idx: int) -> float:
        """
        Calculează Remaining Processing Time (RPT) pentru un job de la o anumită operație.
        Folosește un cache pentru a evita recalculările.
        """
        key = (job_sim_id, from_op_idx)
        if key in context.rpt_cache:
            return context.rpt_cache[key]

        job_obj = get_job_object_from_sim(job_sim_id)
        if not job_obj or from_op_idx >= job_obj.num_operations:
            context.rpt_cache[key] = 0.0
            return 0.0

        s = 0.0
        for op_i in range(from_op_idx, job_obj.num_operations):
            op_obj = job_obj.get_operation(op_i)
            if op_obj and op_obj.alternatives:
                s += op_obj.get_best_processing_time()
        context.rpt_cache[key] = s
        return s

    # --- Funcții pentru gestionarea evenimentelor ---
    def _handle_breakdown_event(event: BreakdownEvent, current_sim_time: float):
        """Gestionează un eveniment de defecțiune a mașinii."""
        machine = machines[event.machine_id]
        machine.broken_until = max(machine.broken_until, event.end_time)
        if machine.busy and machine.start_time < machine.broken_until:
            interrupted_job_sim_id = machine.job_id
            interrupted_op_idx = machine.op_idx
            if interrupted_job_sim_id is not None and interrupted_op_idx is not None:
                make_op_ready_sim(interrupted_job_sim_id, interrupted_op_idx, current_sim_time, current_sim_time)
            machine.busy = False
            machine.job_id = None
            machine.op_idx = None
            machine.time_remaining = 0.0
            machine.start_time = 0.0
            machine.idle_since = current_sim_time

    def _handle_add_job_event(event: AddJobDynamicEvent, current_sim_time: float):
        """Gestionează un eveniment de adăugare a unui job nou."""
        new_job_obj_from_event = event.job_object
        if new_job_obj_from_event.sim_id not in current_jobs_sim_map:
            sim_job_copy_added = copy.copy(new_job_obj_from_event)
            sim_job_copy_added.current_op_idx_sim = 0
            sim_job_copy_added.completion_time_sim = None
            sim_job_copy_added.is_cancelled_sim = False
            current_jobs_sim_map[new_job_obj_from_event.sim_id] = sim_job_copy_added
            context.job_end_time_sim[new_job_obj_from_event.sim_id] = 0.0

            if not sim_job_copy_added.is_cancelled_sim:
                context.current_total_ops += sim_job_copy_added.num_operations

            if sim_job_copy_added.num_operations > 0:
                context.job_internal_pred_finish_time[(sim_job_copy_added.sim_id, 0)] = event.event_time
                etpc_min_add = context.etpc_min_start_for_hind.get((sim_job_copy_added.sim_id, 0), 0.0)
                context.effective_ready_time[(sim_job_copy_added.sim_id, 0)] = max(event.event_time, etpc_min_add)
                context.ready_ops.add((sim_job_copy_added.sim_id, 0))

    def _handle_cancel_job_event(event: CancelJobEvent, current_sim_time: float):
        """Gestionează un eveniment de anulare a unui job."""
        sim_id_to_cancel = event.job_to_cancel_sim_id_mapped
        if sim_id_to_cancel is None:
            job_to_cancel_obj_lookup = fjsp_instance.get_job_by_original_id(event.job_to_cancel_orig_id_ref)
            if job_to_cancel_obj_lookup:
                sim_id_to_cancel = job_to_cancel_obj_lookup.sim_id

        if sim_id_to_cancel is not None and sim_id_to_cancel not in context.cancelled_sim_ids:
            job_being_cancelled = get_job_object_from_sim(sim_id_to_cancel)
            if job_being_cancelled and not job_being_cancelled.is_cancelled_sim:
                job_being_cancelled.is_cancelled_sim = True
                context.cancelled_sim_ids.add(sim_id_to_cancel)

                for m_cancel in machines:
                    if m_cancel.busy and m_cancel.job_id == sim_id_to_cancel:
                        m_cancel.busy = False
                        m_cancel.job_id = None
                        m_cancel.op_idx = None
                        m_cancel.time_remaining = 0.0
                        m_cancel.start_time = 0.0
                        m_cancel.idle_since = current_sim_time

                # Actualizare setului ready_ops
                context.ready_ops = {op for op in context.ready_ops if op[0] != sim_id_to_cancel}

                ops_already_done_for_cancelled = 0
                for sched_j, sched_op_idx, _, _, _ in context.schedule_output:
                    if sched_j == sim_id_to_cancel:
                        ops_already_done_for_cancelled = max(ops_already_done_for_cancelled, sched_op_idx + 1)

                ops_to_remove_from_total = job_being_cancelled.num_operations - ops_already_done_for_cancelled
                if ops_to_remove_from_total > 0:
                    context.current_total_ops -= ops_to_remove_from_total

    def _process_dynamic_events():
        """Procesează toate evenimentele dinamice la timpul curent."""
        while context.event_timeline_idx < len(event_timeline_sim) and \
                event_timeline_sim[context.event_timeline_idx].event_time <= context.current_time + 0:
            event = event_timeline_sim[context.event_timeline_idx]
            if event.event_time < context.current_time - 0:
                context.event_timeline_idx += 1
                continue

            context.event_timeline_idx += 1

            if isinstance(event, BreakdownEvent):
                _handle_breakdown_event(event, context.current_time)
            elif isinstance(event, AddJobDynamicEvent):
                _handle_add_job_event(event, context.current_time)
            elif isinstance(event, CancelJobEvent):
                _handle_cancel_job_event(event, context.current_time)
            # Nu există 'else: break' aici, deoarece dorim să procesăm toate evenimentele relevante.

    # --- Funcții pentru actualizarea mașinilor ---
    def _update_machine_state():
        """Actualizează starea mașinilor și finalizează operațiile."""
        for machine in machines:
            if machine.broken_until > context.current_time + 0:
                continue

            if machine.broken_until > 0 and machine.broken_until <= context.current_time + 0:
                machine.broken_until = 0.0
                if not machine.busy:
                    machine.idle_since = context.current_time

            if machine.busy:
                machine.time_remaining -= 1.0
                if machine.time_remaining <= 0:
                    job_sim_id_done = machine.job_id
                    op_idx_done = machine.op_idx
                    op_start_time = machine.start_time
                    op_end_time = context.current_time

                    machine.busy = False
                    machine.job_id = None
                    machine.op_idx = None
                    machine.time_remaining = 0.0
                    machine.start_time = 0.0
                    machine.idle_since = op_end_time

                    context.completed_total_ops += 1
                    if job_sim_id_done is not None and op_idx_done is not None:
                        context.job_end_time_sim[job_sim_id_done] = op_end_time
                        context.schedule_output.append(
                            (job_sim_id_done, op_idx_done, machine.id, op_start_time, op_end_time))

                        # Procesează constrângerile ETPC pentru operația finalizată
                        if (job_sim_id_done, op_idx_done) in context.etpc_hind_map:
                            for hj_sim, ho_idx, tl_val in context.etpc_hind_map[(job_sim_id_done, op_idx_done)]:
                                new_min_start = op_end_time + tl_val
                                current_min = context.etpc_min_start_for_hind.get((hj_sim, ho_idx), 0.0)
                                context.etpc_min_start_for_hind[(hj_sim, ho_idx)] = max(current_min, new_min_start)

                                if (hj_sim, ho_idx) in context.job_internal_pred_finish_time:
                                    base_jprd_time_etpc = context.job_internal_pred_finish_time[(hj_sim, ho_idx)]
                                    context.effective_ready_time[(hj_sim, ho_idx)] = max(
                                        base_jprd_time_etpc,
                                        context.etpc_min_start_for_hind[(hj_sim, ho_idx)]
                                    )

                        # Marchează următoarea operație a jobului ca gata
                        job_done_obj = get_job_object_from_sim(job_sim_id_done)
                        if job_done_obj and op_idx_done + 1 < job_done_obj.num_operations and \
                                job_sim_id_done not in context.cancelled_sim_ids:
                            make_op_ready_sim(job_sim_id_done, op_idx_done + 1, context.current_time,
                                              context.current_time)

    # --- Funcție pentru dispatching ---
    def _dispatch_operations():
        """
        Selectează și programează operațiile cu cea mai bună prioritate pe toate mașinile libere,
        rezolvând conflictele pentru operații sau mașini.
        """
        all_potential_assignments: List[
            Tuple[float, int, int, int, float]] = []  # (priority, machine_id, job_sim_id, op_idx, ptime)

        MW_val = 0.0
        WIP_val = 0.0
        TUF_val = 0.0

        # 1. Colectează toți candidații posibili (mașină, operație, prioritate)
        for machine in machines:
            if not machine.busy and machine.broken_until <= context.current_time + 0:
                # Recalculate for the current machine
                # These variables are correctly calculated here, but initialized above
                MW_val = (context.current_time + 1.0) - machine.idle_since
                WIP_val = sum(1 for m_wip in machines if m_wip.busy)
                TUF_val = max(0.0, machine.broken_until - context.current_time)

                ready_ops_copy = list(context.ready_ops)  # Copie pentru a evita modificarea în timpul iterării
                for job_sim_id_cand, op_idx_cand in ready_ops_copy:
                    if job_sim_id_cand in context.cancelled_sim_ids:
                        continue

                    job_cand_obj = get_job_object_from_sim(job_sim_id_cand)
                    if not job_cand_obj or op_idx_cand >= job_cand_obj.num_operations:
                        continue

                    # Calculează timpii de disponibilitate pentru operația candidată
                    base_ready_cand = context.job_internal_pred_finish_time.get(
                        (job_sim_id_cand, op_idx_cand), job_cand_obj.arrival_time
                    )
                    etpc_min_cand_val = context.etpc_min_start_for_hind.get((job_sim_id_cand, op_idx_cand), 0.0)
                    op_eff_ready_t_cand = max(base_ready_cand, etpc_min_cand_val)
                    context.effective_ready_time[(job_sim_id_cand, op_idx_cand)] = op_eff_ready_t_cand

                    # Dacă operația nu este încă gata pentru a începe în acest pas de timp, sari peste
                    if op_eff_ready_t_cand > context.current_time + 1.0 - 0:
                        continue

                    operation_cand_obj = job_cand_obj.get_operation(op_idx_cand)
                    if not operation_cand_obj:
                        continue

                    # Obține timpul de procesare pe mașina curentă
                    ptime_cand_on_machine = operation_cand_obj.get_processing_time(machine.id)

                    # Dacă timpul de procesare este valid (nu None și pozitiv)
                    if ptime_cand_on_machine is not None and ptime_cand_on_machine > 0:
                        PT_val = ptime_cand_on_machine
                        RO_val = float(job_cand_obj.num_operations - (op_idx_cand + 1))
                        TQ_val = max(0.0, (context.current_time + 1.0) - op_eff_ready_t_cand)
                        RPT_val = compute_rpt_sim(job_sim_id_cand, op_idx_cand)
                        WJ_val = job_cand_obj.weight
                        DD_val = job_cand_obj.due_date
                        SLK_val = DD_val - (context.current_time + 1.0) - RPT_val if DD_val != float('inf') else float(
                            'inf')

                        job_internal_ready_time_cand = context.job_internal_pred_finish_time.get(
                            (job_sim_id_cand, op_idx_cand), job_cand_obj.arrival_time
                        )
                        ETPC_D_val = max(0.0, op_eff_ready_t_cand - job_internal_ready_time_cand)
                        N_ETPC_S_val = float(len(context.etpc_hind_map.get((job_sim_id_cand, op_idx_cand), [])))

                        try:
                            # Apelarea regulii de dispatch compilate pentru a obține prioritatea
                            priority = dispatch_rule_callable(
                                PT_val, RO_val, MW_val, TQ_val, WIP_val, RPT_val,
                                TUF_val, DD_val, SLK_val, WJ_val,
                                ETPC_D_val, N_ETPC_S_val
                            )
                        except Exception as e_dispatch_rule:
                            print(
                                f"Eroare în regula de dispatch pentru operația ({job_sim_id_cand}, {op_idx_cand}) pe mașina {machine.id}: {e_dispatch_rule}")
                            priority = float('inf')  # Prioritate foarte mare în caz de eroare

                        all_potential_assignments.append(
                            (priority, machine.id, job_sim_id_cand, op_idx_cand, ptime_cand_on_machine))

        # 2. Sortează toți candidații după prioritate (cea mai mică prioritate este cea mai bună)
        # Pentru tie-breaking: prioritizează mașinile cu ID mai mic și apoi operațiile cu ID de job mai mic, apoi op_idx mai mic.
        all_potential_assignments.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

        scheduled_machines_this_step = set()
        scheduled_ops_this_step = set()  # (job_sim_id, op_idx)

        # 3. Alocă operații pe mașini, respectând prioritățile și evitând conflictele
        for assignment in all_potential_assignments:
            _prio, chosen_machine_id, chosen_job_sim_id, chosen_op_idx, chosen_pt_val = assignment

            # Verifică dacă mașina este încă liberă și operația este încă disponibilă
            if machines[chosen_machine_id].busy or chosen_machine_id in scheduled_machines_this_step:
                continue
            if (chosen_job_sim_id, chosen_op_idx) not in context.ready_ops or \
                    (chosen_job_sim_id, chosen_op_idx) in scheduled_ops_this_step:
                continue

            # Dacă ajungem aici, înseamnă că aceasta este cea mai bună alocare validă non-conflictuală
            chosen_machine = machines[chosen_machine_id]
            chosen_machine.busy = True
            chosen_machine.job_id = chosen_job_sim_id
            chosen_machine.op_idx = chosen_op_idx
            chosen_machine.time_remaining = chosen_pt_val
            chosen_machine.start_time = context.current_time + 1

            # Marchează mașina și operația ca fiind programate pentru acest pas de timp
            scheduled_machines_this_step.add(chosen_machine_id)
            scheduled_ops_this_step.add((chosen_job_sim_id, chosen_op_idx))

            # Elimină operația din setul general de operații gata
            context.ready_ops.remove((chosen_job_sim_id, chosen_op_idx))

    # --- Funcție pentru verificarea terminării simulării ---
    def _check_simulation_termination() -> bool:
        """Verifică dacă simularea ar trebui să se termine."""
        if context.completed_total_ops >= context.current_total_ops:
            all_active_jobs_completed_in_schedule = True
            for sim_id_check, job_obj_check in current_jobs_sim_map.items():
                if sim_id_check not in context.cancelled_sim_ids and job_obj_check.num_operations > 0:
                    last_op_idx_check = job_obj_check.num_operations - 1
                    found_last_op_in_schedule = any(
                        sched_j == sim_id_check and sched_o == last_op_idx_check
                        for sched_j, sched_o, _, _, _ in context.schedule_output
                    )
                    if not found_last_op_in_schedule:
                        all_active_jobs_completed_in_schedule = False
                        break

            future_add_job_events = any(
                isinstance(event_timeline_sim[i], AddJobDynamicEvent) and event_timeline_sim[
                    i].event_time > context.current_time - 0
                for i in range(context.event_timeline_idx, len(event_timeline_sim))
            )

            if all_active_jobs_completed_in_schedule and not context.ready_ops and not future_add_job_events:
                return True
        return False

    # --- Bucla principală a simulării ---
    while context.current_time < SIMULATION_MAX_TIME_LIMIT:
        _process_dynamic_events()
        _update_machine_state()
        _dispatch_operations()

        if _check_simulation_termination():
            break

        context.current_time += 1

    # --- Calculul makespan-ului final ---
    final_makespan = 0.0
    active_job_present_at_end = any(
        sim_id_mk not in context.cancelled_sim_ids and job_obj_mk.num_operations > 0
        for sim_id_mk, job_obj_mk in current_jobs_sim_map.items()
    )

    if not context.schedule_output and active_job_present_at_end:
        final_makespan = SIMULATION_MAX_TIME_LIMIT if context.current_time >= SIMULATION_MAX_TIME_LIMIT - 0 else max_time
    else:
        for sim_id_mk in current_jobs_sim_map.keys():
            if sim_id_mk not in context.cancelled_sim_ids:
                final_makespan = max(final_makespan, context.job_end_time_sim.get(sim_id_mk, 0.0))

    if final_makespan == 0.0 and active_job_present_at_end and context.current_time >= SIMULATION_MAX_TIME_LIMIT - 0:
        final_makespan = SIMULATION_MAX_TIME_LIMIT

    return final_makespan, context.schedule_output
