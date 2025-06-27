# utils.py
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import csv
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union

from data_reader import FJSPInstance, BreakdownEvent, Job
from evaluator import get_job_completion_times_from_schedule  # Necesită pentru calc_job_related_metrics

# --- CONSTANTE ȘI FUNCȚII AUXILIARE PENTRU ORAR ---
TUPLE_FIELDS = {"job": 0, "op": 1, "machine": 2, "start": 3, "end": 4}


def field(op_tuple: Tuple, name: str) -> Any:
    """
    Returnează valoarea unui câmp specific dintr-un tuplu de operație (ex: "job", "start").
    """
    if not isinstance(op_tuple, tuple) or len(op_tuple) < max(TUPLE_FIELDS.values()) + 1:
        return None
    return op_tuple[TUPLE_FIELDS[name]]


# --- FUNCȚII UTILITARE GENERALE ---
def sanitize_filename_str(name: str, max_len: int = 100) -> str:
    """
    Sanitizează un șir de caractere pentru a fi utilizabil în numele de fișiere.
    Elimină caracterele nepermise și înlocuiește spațiile cu underscore-uri.
    """
    name = str(name)
    name = re.sub(r'[^\w\s-]', '', name)  # Elimină caracterele non-alfanumerice (cu excepția spațiilor și cratimelor)
    name = re.sub(r'[\s_]+', '_', name)  # Înlocuiește multiple spații/underscore-uri cu un singur underscore
    name = name.strip('_')  # Elimină underscore-urile de la început/sfârșit
    return name[:max_len]  # Trunchiază la lungimea maximă


def calculate_std_dev(data: List[float]) -> float:
    """
    Calculează deviația standard a unei liste de numere. Returnează 0.0 dacă lista este goală sau are un singur element.
    """
    if not data or len(data) < 2:
        return 0.0
    return float(np.std(data))


def safe_avg_list(lst: List[Union[float, int]]) -> float:
    """
    Calculează media unei liste de numere. Returnează 0.0 dacă lista este goală pentru a evita erori.
    """
    return sum(lst) / len(lst) if lst else 0.0


# --- FUNCȚII DE CALCUL AL METRICILOR ---
def get_per_machine_operation_counts(sched: List[Tuple], num_total_machines: int) -> Dict[int, int]:
    """
    Calculează numărul de operații alocate fiecărei mașini dintr-un orar.
    """
    op_counts = {m: 0 for m in range(num_total_machines)}
    if not sched:
        return op_counts
    for op_tuple in sched:
        machine_idx = field(op_tuple, "machine")
        if machine_idx is not None and 0 <= machine_idx < num_total_machines:
            op_counts[machine_idx] += 1
    return op_counts


def calc_machine_metrics(sched: List[Tuple], num_total_machines: int, schedule_makespan: float) -> \
        Tuple[float, float, float, float, float, List[float], List[float], Dict[int, int]]:
    """
    Calculează diverse metrici legate de mașini (timp de inactivitate, utilizare, etc.) dintr-un orar.

    Returnează:
        - total_idle_time_all_machines: Timpul total de inactivitate al tuturor mașinilor.
        - avg_idle_time: Timpul mediu de inactivitate per mașină.
        - std_dev_idle_time: Deviația standard a timpului de inactivitate per mașină.
        - total_busy_time_all_machines: Timpul total de ocupare al tuturor mașinilor.
        - avg_busy_utilization: Utilizarea medie a mașinilor (procent).
        - idle_times_per_machine: Listă cu timpul de inactivitate pentru fiecare mașină.
        - busy_times_per_machine: Listă cu timpul de ocupare pentru fiecare mașină.
        - ops_counts_per_machine: Dicționar cu numărul de operații per mașină.
    """
    if num_total_machines == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, [], [], {}

    ops_by_m = defaultdict(list)
    max_overall_finish_time = schedule_makespan
    # Dacă orarul este gol și makespan-ul este aproape de zero, evităm împărțirea la zero.
    if not sched and abs(max_overall_finish_time) < 1e-9:
        max_overall_finish_time = 1.0

    for op_tuple in sched:
        machine_idx = field(op_tuple, "machine")
        op_start = field(op_tuple, "start")
        op_end = field(op_tuple, "end")
        if machine_idx is None or op_start is None or op_end is None:
            continue
        ops_by_m[machine_idx].append((op_start, op_end))

    idle_times_per_machine: List[float] = [0.0] * num_total_machines
    busy_times_per_machine: List[float] = [0.0] * num_total_machines
    ops_counts_per_machine: Dict[int, int] = {m: 0 for m in range(num_total_machines)}

    for m_idx in range(num_total_machines):
        ops_list = sorted(ops_by_m.get(m_idx, []), key=lambda x: x[0])
        ops_counts_per_machine[m_idx] = len(ops_list)

        machine_idle = 0.0
        machine_busy = 0.0
        last_op_end_time = 0.0

        if not ops_list:
            machine_idle = max_overall_finish_time  # Mașina este inactivă pe toată durata makespan-ului
        else:
            # Timpul de inactivitate înainte de prima operație
            machine_idle += max(0.0, ops_list[0][0] - 0.0)
            last_op_end_time = ops_list[0][1]
            machine_busy += (ops_list[0][1] - ops_list[0][0])

            # Timpul de inactivitate între operații
            for i in range(1, len(ops_list)):
                machine_idle += max(0.0, ops_list[i][0] - last_op_end_time)
                machine_busy += (ops_list[i][1] - ops_list[i][0])
                last_op_end_time = ops_list[i][1]

            # Timpul de inactivitate după ultima operație până la makespan
            machine_idle += max(0.0, max_overall_finish_time - last_op_end_time)

        idle_times_per_machine[m_idx] = machine_idle
        busy_times_per_machine[m_idx] = machine_busy

    total_idle_time_all_machines = sum(idle_times_per_machine)
    total_busy_time_all_machines = sum(busy_times_per_machine)

    avg_idle_time = total_idle_time_all_machines / num_total_machines if num_total_machines > 0 else 0.0
    std_dev_idle_time = calculate_std_dev(idle_times_per_machine)

    avg_busy_utilization = (total_busy_time_all_machines / (max_overall_finish_time * num_total_machines)) \
        if max_overall_finish_time > 1e-9 and num_total_machines > 0 else 0.0

    return total_idle_time_all_machines, avg_idle_time, std_dev_idle_time, \
        total_busy_time_all_machines, avg_busy_utilization, \
        idle_times_per_machine, busy_times_per_machine, ops_counts_per_machine


def calc_job_related_metrics(sched: List[Tuple], instance: FJSPInstance, schedule_makespan: float) -> \
        Dict[str, Any]:
    """
    Calculează diverse metrici legate de job-uri (timp de flux, timp de așteptare, tardiness, etc.) dintr-un orar.
    """
    metrics = {
        "total_flow_time": 0.0, "avg_flow_time": 0.0, "std_flow_time": 0.0,
        "total_wait_time": 0.0, "avg_wait_time": 0.0, "std_wait_time": 0.0,
        "total_weighted_tardiness": 0.0, "avg_tardiness": 0.0, "std_tardiness": 0.0,
        "num_tardy_jobs": 0, "num_completed_jobs": 0,
        "per_job_flow_time": [], "per_job_wait_time": [], "per_job_tardiness": [],
        "per_job_completion_time": [], "per_job_is_tardy": [], "per_job_sim_id": []
    }
    if not instance.jobs_defined_in_file:
        return metrics

    job_completion_times = get_job_completion_times_from_schedule(sched)
    ops_by_j_for_wait = defaultdict(list)
    for op_tuple in sched:
        job_sim_id_ops = field(op_tuple, "job")
        if job_sim_id_ops is None:
            continue
        ops_by_j_for_wait[job_sim_id_ops].append(
            (field(op_tuple, "op"), field(op_tuple, "start"), field(op_tuple, "end"))
        )

    num_relevant_for_avg_tardiness = 0
    for job_obj in instance.jobs_defined_in_file:
        if hasattr(job_obj, 'is_cancelled_sim') and job_obj.is_cancelled_sim:
            continue
        if job_obj.num_operations == 0:
            continue  # Ignoră job-urile fără operații reale

        metrics["per_job_sim_id"].append(job_obj.sim_id)

        completion_time: Optional[float] = None
        is_completed = False
        if job_obj.sim_id in job_completion_times:
            completion_time = job_completion_times[job_obj.sim_id]
            is_completed = True
            metrics["num_completed_jobs"] += 1
        else:
            # Dacă un job nu este completat, asumăm că se termină la makespan pentru calculele de flux/tardiness
            completion_time = schedule_makespan

        metrics["per_job_completion_time"].append(completion_time if is_completed else -1)

        flow_time = completion_time - job_obj.arrival_time
        metrics["per_job_flow_time"].append(max(0.0, flow_time))

        current_job_total_wait = 0.0
        if is_completed:
            job_ops_in_schedule = sorted(ops_by_j_for_wait.get(job_obj.sim_id, []), key=lambda x: x[0])
            prev_op_end_or_arrival = job_obj.arrival_time
            if job_ops_in_schedule:
                for _op_idx, op_start, op_end in job_ops_in_schedule:
                    if op_start is None or op_end is None:
                        continue
                    current_job_total_wait += max(0.0, op_start - prev_op_end_or_arrival)
                    prev_op_end_or_arrival = op_end
        metrics["per_job_wait_time"].append(current_job_total_wait)

        tardiness = 0.0
        is_tardy_flag = 0
        if job_obj.due_date != float('inf'):  # Doar dacă există o dată de scadență finită
            num_relevant_for_avg_tardiness += 1
            tardiness = max(0.0, completion_time - job_obj.due_date)
            if tardiness > 1e-9:  # Considerăm "tardy" dacă tardiness-ul este semnificativ
                metrics["num_tardy_jobs"] += 1
                is_tardy_flag = 1
        metrics["per_job_tardiness"].append(tardiness)
        metrics["per_job_is_tardy"].append(is_tardy_flag)
        metrics["total_weighted_tardiness"] += job_obj.weight * tardiness

    if metrics["per_job_flow_time"]:
        metrics["total_flow_time"] = sum(metrics["per_job_flow_time"])
        metrics["avg_flow_time"] = metrics["total_flow_time"] / len(metrics["per_job_flow_time"]) if len(
            metrics["per_job_flow_time"]) > 0 else 0.0
        metrics["std_flow_time"] = calculate_std_dev(metrics["per_job_flow_time"])

    if metrics["per_job_wait_time"]:
        metrics["total_wait_time"] = sum(metrics["per_job_wait_time"])
        metrics["avg_wait_time"] = metrics["total_wait_time"] / len(metrics["per_job_wait_time"]) if len(
            metrics["per_job_wait_time"]) > 0 else 0.0
        metrics["std_wait_time"] = calculate_std_dev(metrics["per_job_wait_time"])

    if num_relevant_for_avg_tardiness > 0:
        finite_due_date_tardiness = [t for t, job_id in zip(metrics["per_job_tardiness"], metrics["per_job_sim_id"])
                                     if instance.get_job_by_sim_id(job_id) and instance.get_job_by_sim_id(
                job_id).due_date != float('inf')]
        metrics["avg_tardiness"] = sum(
            finite_due_date_tardiness) / num_relevant_for_avg_tardiness if num_relevant_for_avg_tardiness > 0 else 0.0
        metrics["std_tardiness"] = calculate_std_dev(finite_due_date_tardiness)

    return metrics


# --- FUNCȚII DE GENERARE RAPOARTE CSV ---
def generate_per_machine_metrics_csv(
        idle_times_per_machine: List[float], busy_times_per_machine: List[float],
        ops_counts_per_machine: Dict[int, int], num_total_machines: int,
        schedule_makespan: float, file_path: Path):
    """
    Generează un fișier CSV cu metrici per mașină (timp inactiv, ocupat, utilizare, număr operații).
    """
    header = ["MachineID", "IdleTime", "BusyTime", "Utilization", "OpsCount"]
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for m_idx in range(num_total_machines):
                idle = idle_times_per_machine[m_idx] if m_idx < len(idle_times_per_machine) else schedule_makespan
                busy = busy_times_per_machine[m_idx] if m_idx < len(busy_times_per_machine) else 0.0
                util = (busy / schedule_makespan) if schedule_makespan > 1e-9 else 0.0
                ops_count = ops_counts_per_machine.get(m_idx, 0)
                writer.writerow([f"M{m_idx}", f"{idle:.2f}", f"{busy:.2f}", f"{util:.3f}", ops_count])
    except IOError as e:
        print(f"Error writing machine metrics CSV to {file_path}: {e}")


def generate_per_job_metrics_csv(
        instance: FJSPInstance, job_metrics_dict: Dict[str, Any], file_path: Path):
    """
    Generează un fișier CSV cu metrici per job (timp de finalizare, timp de flux, tardiness, etc.).
    """
    header = ["JobSimID", "OriginalID", "ArrivalTime", "Weight", "DueDate",
              "CompletionTime", "FlowTime", "WaitingTime", "Tardiness", "IsTardy"]
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            num_jobs_in_metrics = len(job_metrics_dict.get("per_job_sim_id", []))
            for i in range(num_jobs_in_metrics):
                sim_id = job_metrics_dict["per_job_sim_id"][i]
                job_obj = instance.get_job_by_sim_id(sim_id)
                if not job_obj: continue
                row = [
                    sim_id,
                    job_obj.original_id_from_json if job_obj.original_id_from_json is not None else f"SimID_{sim_id}",
                    f"{job_obj.arrival_time:.2f}", f"{job_obj.weight:.2f}",
                    f"{job_obj.due_date:.2f}" if job_obj.due_date != float('inf') else "inf",
                    f"{job_metrics_dict['per_job_completion_time'][i]:.2f}" if
                    job_metrics_dict['per_job_completion_time'][i] != -1 else "N/A",
                    f"{job_metrics_dict['per_job_flow_time'][i]:.2f}",
                    f"{job_metrics_dict['per_job_wait_time'][i]:.2f}",
                    f"{job_metrics_dict['per_job_tardiness'][i]:.2f}",
                    job_metrics_dict['per_job_is_tardy'][i]
                ]
                writer.writerow(row)
    except IOError as e:
        print(f"Error writing job metrics CSV to {file_path}: {e}")