# main.py
from __future__ import annotations

import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union
import re
import numpy as np
import csv
import json
import datetime
import itertools

from deap import gp

from data_reader import load_instances_from_directory, FJSPInstance, BreakdownEvent, Job
from evaluator import run_genetic_program, calculate_total_weighted_tardiness, evaluate_individual, \
    get_job_completion_times_from_schedule
from ganttPlot import plot_gantt
from gpSetup import create_toolbox
from simpleTree import simplify_individual, tree_str, infix_str

# ---------------------------------------------------------------------------
# CONFIG (Parametri Generali ai Experimentului)
# ---------------------------------------------------------------------------
TRAIN_DIR = Path("dynamic_data/extended/training_sets_small")
TEST_DIR = Path("dynamic_data/extended/test_sets")
TEST_DIR_SMALL = Path("dynamic_data/extended/test_sets_small")
POP_SIZE = 60
N_GENERATIONS = 40
N_WORKERS = 6  # Numărul de worker-i pentru create_toolbox
MAX_HOF = 5  # Păstrăm doar cei mai buni 5 indivizi per rulare pentru raportare detaliată

BASE_OUTPUT_DIR = Path("rezultate/genetic_experiments")  # Directorul de bază pentru toate experimentele

# ----------------------------------------------------------
# TUPLE_FIELDS și field() (rămân neschimbate)
# ----------------------------------------------------------
TUPLE_FIELDS = {"job": 0, "op": 1, "machine": 2, "start": 3, "end": 4}


def field(op_tuple: Tuple, name: str) -> Any:
    if not isinstance(op_tuple, tuple) or len(op_tuple) < max(TUPLE_FIELDS.values()) + 1:
        return None
    return op_tuple[TUPLE_FIELDS[name]]


# ---------------------------------------------------------------------------
#  UTILITARE (rămân neschimbate)
# ---------------------------------------------------------------------------
def sanitize_filename_str(name: str, max_len: int = 100) -> str:
    name = str(name)
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s_]+', '_', name)
    name = name.strip('_')
    return name[:max_len]


def calculate_std_dev(data: List[float]) -> float:
    if not data or len(data) < 2: return 0.0
    return float(np.std(data))


# ---------------------------------------------------------------------------
#  FUNCTII DE METRICA (rămân neschimbate)
# ---------------------------------------------------------------------------
def get_per_machine_operation_counts(sched: List[Tuple], num_total_machines: int) -> Dict[int, int]:
    op_counts = {m: 0 for m in range(num_total_machines)}
    if not sched: return op_counts
    for op_tuple in sched:
        machine_idx = field(op_tuple, "machine")
        if machine_idx is not None and 0 <= machine_idx < num_total_machines:
            op_counts[machine_idx] += 1
    return op_counts


def calc_machine_metrics(sched: List[Tuple], num_total_machines: int, schedule_makespan: float) -> \
        Tuple[float, float, float, float, float, List[float], List[float], Dict[int, int]]:
    if num_total_machines == 0: return 0.0, 0.0, 0.0, 0.0, 0.0, [], [], {}
    ops_by_m = defaultdict(list)
    max_overall_finish_time = schedule_makespan
    if not sched and abs(max_overall_finish_time) < 1e-9:
        max_overall_finish_time = 1.0
    for op_tuple in sched:
        machine_idx = field(op_tuple, "machine")
        op_start = field(op_tuple, "start")
        op_end = field(op_tuple, "end")
        if machine_idx is None or op_start is None or op_end is None: continue
        ops_by_m[machine_idx].append((op_start, op_end))
    idle_times_per_machine: List[float] = [0.0] * num_total_machines
    busy_times_per_machine: List[float] = [0.0] * num_total_machines
    ops_counts_per_machine: Dict[int, int] = {m: 0 for m in range(num_total_machines)}
    for m_idx in range(num_total_machines):
        ops_list = sorted(ops_by_m.get(m_idx, []), key=lambda x: x[0])
        ops_counts_per_machine[m_idx] = len(ops_list)
        machine_idle = 0.0;
        machine_busy = 0.0;
        last_op_end_time = 0.0
        if not ops_list:
            machine_idle = max_overall_finish_time
        else:
            machine_idle += max(0.0, ops_list[0][0] - 0.0)
            last_op_end_time = ops_list[0][1]
            machine_busy += (ops_list[0][1] - ops_list[0][0])
            for i in range(1, len(ops_list)):
                machine_idle += max(0.0, ops_list[i][0] - last_op_end_time)
                machine_busy += (ops_list[i][1] - ops_list[i][0])
                last_op_end_time = ops_list[i][1]
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
    metrics = {
        "total_flow_time": 0.0, "avg_flow_time": 0.0, "std_flow_time": 0.0,
        "total_wait_time": 0.0, "avg_wait_time": 0.0, "std_wait_time": 0.0,
        "total_weighted_tardiness": 0.0, "avg_tardiness": 0.0, "std_tardiness": 0.0,
        "num_tardy_jobs": 0, "num_completed_jobs": 0,
        "per_job_flow_time": [], "per_job_wait_time": [], "per_job_tardiness": [],
        "per_job_completion_time": [], "per_job_is_tardy": [], "per_job_sim_id": []
    }
    if not instance.jobs_defined_in_file: return metrics
    job_completion_times = get_job_completion_times_from_schedule(sched)
    ops_by_j_for_wait = defaultdict(list)
    for op_tuple in sched:
        job_sim_id_ops = field(op_tuple, "job")
        if job_sim_id_ops is None: continue
        ops_by_j_for_wait[job_sim_id_ops].append(
            (field(op_tuple, "op"), field(op_tuple, "start"), field(op_tuple, "end"))
        )
    num_relevant_for_avg_tardiness = 0
    for job_obj in instance.jobs_defined_in_file:
        if hasattr(job_obj, 'is_cancelled_sim') and job_obj.is_cancelled_sim: continue
        if job_obj.num_operations == 0: continue
        metrics["per_job_sim_id"].append(job_obj.sim_id)
        completion_time: Optional[float] = None;
        is_completed = False
        if job_obj.sim_id in job_completion_times:
            completion_time = job_completion_times[job_obj.sim_id]
            is_completed = True;
            metrics["num_completed_jobs"] += 1
        else:
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
                    if op_start is None or op_end is None: continue
                    current_job_total_wait += max(0.0, op_start - prev_op_end_or_arrival)
                    prev_op_end_or_arrival = op_end
        metrics["per_job_wait_time"].append(current_job_total_wait)
        tardiness = 0.0;
        is_tardy_flag = 0
        if job_obj.due_date != float('inf'):
            num_relevant_for_avg_tardiness += 1
            tardiness = max(0.0, completion_time - job_obj.due_date)
            if tardiness > 1e-9: metrics["num_tardy_jobs"] += 1; is_tardy_flag = 1
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


def generate_per_machine_metrics_csv(
        idle_times_per_machine: List[float], busy_times_per_machine: List[float],
        ops_counts_per_machine: Dict[int, int], num_total_machines: int,
        schedule_makespan: float, file_path: Path):
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


def evaluate_and_save_results(
        instances_for_testing: List[FJSPInstance],
        best_individuals: List[gp.PrimitiveTree],
        toolbox,
        output_base_dir: Path,
        label: str = ""
):
    output_base_dir.mkdir(parents=True, exist_ok=True)
    for rank, individual_tree_original in enumerate(best_individuals, 1):
        individual_fitness_train = individual_tree_original.fitness.values[
            0] if individual_tree_original.fitness.valid else float("inf")
        simplified_individual_tree = simplify_individual(individual_tree_original, toolbox.pset)
        ind_size_original = len(individual_tree_original)
        ind_depth_original = individual_tree_original.height
        ind_size_simplified = len(simplified_individual_tree)
        ind_depth_simplified = simplified_individual_tree.height
        formula_infix_str = infix_str(simplified_individual_tree)
        formula_str_sanitized = sanitize_filename_str(formula_infix_str)
        individual_output_dir = output_base_dir / f"Indiv_{rank}"
        individual_output_dir.mkdir(parents=True, exist_ok=True)
        summary_results_file_path = individual_output_dir / "summary_overall_metrics.txt"
        per_instance_details_dir = individual_output_dir / "per_instance_details"

        print(
            f"\n--- Testing Individual {rank}/{len(best_individuals)} ({label} Train Fitness={individual_fitness_train:.4f}) ---")
        print(f"Output directory: {individual_output_dir}")
        print(f"Original Tree: {str(individual_tree_original)}")
        print(f"Simplified Formula: {formula_infix_str}")

        all_instance_makespans: List[float] = []
        all_instance_total_weighted_tardiness: List[float] = []
        all_instance_avg_machine_idles: List[float] = []
        all_instance_std_dev_machine_idles: List[float] = []
        all_instance_avg_job_waits: List[float] = []
        all_instance_std_dev_job_waits: List[float] = []
        all_instance_avg_job_tardiness: List[float] = []
        all_instance_std_dev_job_tardiness: List[float] = []
        all_instance_num_tardy_jobs: List[int] = []
        all_instance_evaluation_times: List[float] = []
        all_instance_avg_busy_util: List[float] = []

        with open(summary_results_file_path, "w", encoding="utf-8") as outf_summary:
            outf_summary.write(f"=== Individual {rank} (Rank in HoF) ===\n")
            outf_summary.write(f"Training_Fitness: {individual_fitness_train:.4f}\n")
            outf_summary.write(f"Original_Size: {ind_size_original}, Original_Depth: {ind_depth_original}\n")
            outf_summary.write(f"Original_Tree (string): {str(individual_tree_original)}\n")
            outf_summary.write(f"Simplified_Size: {ind_size_simplified}, Simplified_Depth: {ind_depth_simplified}\n")
            outf_summary.write(f"Simplified_Formula (infix): {formula_infix_str}\n")
            outf_summary.write(f"Simplified_Tree_ASCII: \n{tree_str(simplified_individual_tree)}\n\n")
            outf_summary.write("Per-Instance Summary Results:\n")
            header = (f"{'Instance':<45} {'MS':<7} {'TWT':<7} {'AvgIdle':<9} {'StdIdle':<9} "
                      f"{'AvgWait':<9} {'StdWait':<9} {'AvgTard':<9} {'StdTard':<9} "
                      f"{'#Tardy':<6} {'AvgBusy%':<9} {'EvalTime(s)':<10}\n")
            outf_summary.write(header)
            outf_summary.write("-" * (len(header) - 1) + "\n")

            for fjsp_instance_obj in instances_for_testing:
                instance_file_stem = Path(fjsp_instance_obj.file_name).stem
                current_instance_output_dir = per_instance_details_dir / sanitize_filename_str(instance_file_stem, 50)
                current_instance_output_dir.mkdir(parents=True, exist_ok=True)

                eval_priority_func = toolbox.compile(expr=simplified_individual_tree)
                time_eval_start = time.perf_counter()
                makespan, schedule_tuples = evaluate_individual(
                    individual=eval_priority_func,
                    fjsp_instance=fjsp_instance_obj,
                    toolbox=toolbox
                )
                eval_elapsed_time = time.perf_counter() - time_eval_start

                _total_idle, avg_idle, std_idle, _total_busy, avg_busy_util, \
                    idle_list_per_m, busy_list_per_m, ops_counts_m = \
                    calc_machine_metrics(schedule_tuples, fjsp_instance_obj.num_machines, makespan)

                job_metrics = calc_job_related_metrics(schedule_tuples, fjsp_instance_obj, makespan)

                all_instance_makespans.append(makespan)
                all_instance_total_weighted_tardiness.append(job_metrics["total_weighted_tardiness"])
                all_instance_avg_machine_idles.append(avg_idle)
                all_instance_std_dev_machine_idles.append(std_idle)
                all_instance_avg_job_waits.append(job_metrics["avg_wait_time"])
                all_instance_std_dev_job_waits.append(job_metrics["std_wait_time"])
                all_instance_avg_job_tardiness.append(job_metrics["avg_tardiness"])
                all_instance_std_dev_job_tardiness.append(job_metrics["std_tardiness"])
                all_instance_num_tardy_jobs.append(job_metrics["num_tardy_jobs"])
                all_instance_evaluation_times.append(eval_elapsed_time)
                all_instance_avg_busy_util.append(avg_busy_util)

                outf_summary.write(f"{instance_file_stem:<45} "
                                   f"{makespan:<7.2f} {job_metrics['total_weighted_tardiness']:<7.2f} "
                                   f"{avg_idle:<9.2f} {std_idle:<9.2f} "
                                   f"{job_metrics['avg_wait_time']:<9.2f} {job_metrics['std_wait_time']:<9.2f} "
                                   f"{job_metrics['avg_tardiness']:<9.2f} {job_metrics['std_tardiness']:<9.2f} "
                                   f"{job_metrics['num_tardy_jobs']:<6} {avg_busy_util * 100:<8.2f}% "
                                   f"{eval_elapsed_time:<10.3f}\n")

                print(
                    f"  Inst: {instance_file_stem:<40} MS={makespan:<7.2f} TWT={job_metrics['total_weighted_tardiness']:<7.2f} "
                    f"AvgIdle={avg_idle:<6.2f} AvgWait={job_metrics['avg_wait_time']:<6.2f} #Tardy={job_metrics['num_tardy_jobs']:<3} "
                    f"AvgBusy%={avg_busy_util * 100:<5.2f}")

                generate_per_machine_metrics_csv(
                    idle_list_per_m, busy_list_per_m, ops_counts_m,
                    fjsp_instance_obj.num_machines, makespan,
                    current_instance_output_dir / "machine_metrics.csv"
                )
                generate_per_job_metrics_csv(
                    fjsp_instance_obj, job_metrics,
                    current_instance_output_dir / "job_metrics.csv"
                )

                breakdowns_for_plot = defaultdict(list)
                for event_obj in fjsp_instance_obj.dynamic_event_timeline:
                    if isinstance(event_obj, BreakdownEvent):
                        breakdowns_for_plot[event_obj.machine_id].append(
                            (event_obj.event_time, event_obj.end_time)
                        )
                gantt_file_name = f"{instance_file_stem}_gantt.png"
                plot_gantt(
                    schedule_tuples,
                    num_machines=fjsp_instance_obj.num_machines,
                    breakdowns=breakdowns_for_plot,
                    title=f"{instance_file_stem} - Ind.{rank} (MS={makespan:.2f}, TWT={job_metrics['total_weighted_tardiness']:.2f})",
                    save_path=str(current_instance_output_dir / gantt_file_name)
                )

            num_test_cases = len(instances_for_testing)

            def safe_avg_list(lst: List[Union[float, int]]) -> float:
                return sum(lst) / len(lst) if lst else 0.0

            outf_summary.write("\n--- OVERALL AVERAGES for this Individual (across test instances) ---\n")
            outf_summary.write(
                f"Average_MS                     : {safe_avg_list(all_instance_makespans):.2f}\n")
            outf_summary.write(
                f"Average_TWT                    : {safe_avg_list(all_instance_total_weighted_tardiness):.2f}\n")
            outf_summary.write(
                f"Average_Avg_Machine_Idle       : {safe_avg_list(all_instance_avg_machine_idles):.2f}\n")
            outf_summary.write(
                f"Average_Std_Machine_Idle       : {safe_avg_list(all_instance_std_dev_machine_idles):.2f}\n")
            outf_summary.write(
                f"Average_Avg_Job_Wait           : {safe_avg_list(all_instance_avg_job_waits):.2f}\n")
            outf_summary.write(
                f"Average_Std_Job_Wait           : {safe_avg_list(all_instance_std_dev_job_waits):.2f}\n")
            outf_summary.write(
                f"Average_Avg_Tardiness          : {safe_avg_list(all_instance_avg_job_tardiness):.2f}\n")
            outf_summary.write(
                f"Average_Std_Tardiness          : {safe_avg_list(all_instance_std_dev_job_tardiness):.2f}\n")
            outf_summary.write(
                f"Average_Num_Tardy_Jobs         : {safe_avg_list(all_instance_num_tardy_jobs):.2f}\n")
            outf_summary.write(
                f"Average_Avg_Machine_Utilization: {safe_avg_list(all_instance_avg_busy_util) * 100:.2f}%\n")
            outf_summary.write(
                f"Average_Eval_Time              : {safe_avg_list(all_instance_evaluation_times):.3f}s\n")

            print("  ------- OVERALL AVERAGES for this Individual (across test instances) -------")
            print(f"  Avg Test MS                     = {safe_avg_list(all_instance_makespans):.2f}")
            print(f"  Avg Test TWT                    = {safe_avg_list(all_instance_total_weighted_tardiness):.2f}")
            print(f"  Avg Test Avg_Machine_Idle       = {safe_avg_list(all_instance_avg_machine_idles):.2f}")
            print(f"  Avg Test Std_Machine_Idle       = {safe_avg_list(all_instance_std_dev_machine_idles):.2f}")
            print(f"  Avg Test Avg_Job_Wait           = {safe_avg_list(all_instance_avg_job_waits):.2f}")
            print(f"  Avg Test Std_Job_Wait           = {safe_avg_list(all_instance_std_dev_job_waits):.2f}")
            print(f"  Avg Test Avg_Tardiness          = {safe_avg_list(all_instance_avg_job_tardiness):.2f}")
            print(f"  Avg Test Std_Tardiness          = {safe_avg_list(all_instance_std_dev_job_tardiness):.2f}")
            print(f"  Avg Test Num_Tardy_Jobs         = {safe_avg_list(all_instance_num_tardy_jobs):.2f}")
            print(f"  Avg Test Avg_Machine_Utilization= {safe_avg_list(all_instance_avg_busy_util) * 100:.2f}%")
            print(f"  Avg Eval Time                   = {safe_avg_list(all_instance_evaluation_times):.3f}s")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    global_start = time.time()

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR_SMALL.mkdir(parents=True, exist_ok=True)

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Directorul de bază pentru toate rezultatele experimentelor

    print("--- Loading Training Instances ---")
    train_insts: List[FJSPInstance] = load_instances_from_directory(str(TRAIN_DIR))
    print(f"Loaded {len(train_insts)} training instances.\n")

    print("--- Loading Test Instances ---")
    test_insts: List[FJSPInstance] = load_instances_from_directory(str(TEST_DIR))
    print(f"Loaded {len(test_insts)} test instances.\n")

    test_insts_small = load_instances_from_directory(str(TEST_DIR_SMALL))
    print(f"Loaded {len(test_insts_small)} small test instances.\n")

    if not train_insts:
        print("No training instances loaded. Exiting.")
        return

    # --- Parameter Grid pentru Experimente ---
    param_grid = {
        "alpha": [0,0.2, 0.5,0.8],  # Parametru pentru funcția de fitness
        "selection_strategy": ["tournament", "roulette", "best"],  # Strategii de selecție
        "crossover_strategy": ["one_point"],  # Strategii de încrucișare
        "mutation_strategy": ["uniform", "node_replacement"]  # Strategii de mutație
    }
    num_runs_per_config = 1  # Numărul de rulări pentru fiecare set de parametri

    # Stochează rezultatele tuturor rulărilor pentru raportul final
    all_experiment_results: List[Dict[str, Any]] = []

    # Pentru a urmări cel mai bun individ găsit în toate rulările și configurațiile
    best_overall_individual_data = {
        "fitness": float('inf'),  # Inițializăm cu o fitness foarte mare pentru probleme de minimizare
        "individual_str": "",
        "config": "",
        "run": -1
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_output_root = BASE_OUTPUT_DIR
    experiment_output_root.mkdir(parents=True, exist_ok=True)

    # Fișierul pentru raportul sumar
    report_file_path = experiment_output_root / "experiment_summary_report.txt"

    print(f"\n=== Starting GP Experiments ({num_runs_per_config} runs per config) ===")
    print(f"Results will be saved in: {experiment_output_root}")

    config_counter = 0

    # Iterăm prin toate combinațiile de parametri
    for alpha_val, sel_strat, cx_strat, mut_strat in itertools.product(
            param_grid["alpha"],
            param_grid["selection_strategy"],
            param_grid["crossover_strategy"],
            param_grid["mutation_strategy"]
    ):
        config_counter += 1
        # Generăm un nume unic și descriptiv pentru configurația curentă
        config_name = (
            f"alpha_{str(alpha_val).replace('.', '_')}_"  # Înlocuim '.' cu '_' pentru nume de fișiere valide
            f"sel_{sel_strat}_"
            f"cx_{cx_strat}_"
            f"mut_{mut_strat}"
        )

        # Creăm un director pentru această configurație specifică
        current_config_output_dir = experiment_output_root / config_name
        current_config_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Running Configuration {config_counter}: {config_name} ---")

        # Rulăm de 10 ori (sau `num_runs_per_config`) pentru fiecare configurație
        for run_idx in range(num_runs_per_config):
            run_start_time = time.time()
            print(f"  Starting Run {run_idx + 1}/{num_runs_per_config} for this config...")

            # Creăm un toolbox nou pentru fiecare rulare pentru a evita efectele secundare între rulări GP
            toolbox = create_toolbox(np=N_WORKERS)

            # Apelăm funcția run_genetic_program cu parametrii curentului set de parametri
            hof_current_run = run_genetic_program(
                train_instances=train_insts,
                toolbox=toolbox,
                ngen=N_GENERATIONS,
                pop_size=POP_SIZE,
                halloffame_size=MAX_HOF,  # Păstrăm MAX_HOF indivizi în Hall of Fame
                alpha=alpha_val,  # Valoarea alpha din configurația curentă
                selection_strategy=sel_strat,
                selection_tournsize=3,
                crossover_strategy=cx_strat,
                mutation_strategy=mut_strat,
                cxpb=0.5,  # Probabilitate de crossover (fixă, poate fi parametrizată)
                mutpb=0.3,  # Probabilitate de mutație (fixă, poate fi parametrizată)
                MAX_DEPTH=7  # Adâncime maximă a arborelui (fixă, poate fi parametrizată)
            )

            # Obținem cel mai bun individ din această rulare
            if hof_current_run:
                # Obținem cel mai bun individ (de obicei primul din Hall of Fame, dacă MAX_HOF=1)
                best_ind_this_run = hof_current_run[0]
                current_run_fitness = best_ind_this_run.fitness.values[0] if best_ind_this_run.fitness.valid else float(
                    'inf')

                # Simplificăm arborele pentru o reprezentare mai clară
                simplified_best_ind = simplify_individual(best_ind_this_run, toolbox.pset)
                simplified_best_ind_str = infix_str(simplified_best_ind)

                run_duration_s = time.time() - run_start_time
                print(
                    f"  Run {run_idx + 1} finished. Best Fitness: {current_run_fitness:.4f} (Duration: {run_duration_s:.2f}s)")

                # Stocăm detaliile rulării pentru raportul final
                run_details = {
                    "config_name": config_name,
                    "alpha": alpha_val,
                    "selection_strategy": sel_strat,
                    "crossover_strategy": cx_strat,
                    "mutation_strategy": mut_strat,
                    "run_index": run_idx + 1,
                    "best_fitness": current_run_fitness,
                    "best_individual_original_str": str(best_ind_this_run),
                    "best_individual_simplified_str": simplified_best_ind_str,
                    "run_duration_s": run_duration_s
                }
                all_experiment_results.append(run_details)

                # Actualizăm cel mai bun individ general dacă rularea curentă este mai bună
                if current_run_fitness < best_overall_individual_data["fitness"]:
                    best_overall_individual_data["fitness"] = current_run_fitness
                    best_overall_individual_data["individual_str"] = simplified_best_ind_str
                    best_overall_individual_data["config"] = config_name
                    best_overall_individual_data["run"] = run_idx + 1

                # Salvarea rezultatelor detaliate pentru această rulare
                # Creăm un director specific pentru această rulare (în interiorul directorului de configurație)
                run_output_dir = current_config_output_dir / f"run_{run_idx + 1}"
                run_output_dir.mkdir(parents=True, exist_ok=True)

                for rank, individual_to_save in enumerate(hof_current_run):
                    # Asigură-te că individul are un fitness valid înainte de a-l procesa
                    if not individual_to_save.fitness.valid:
                        print(f"  Skipping individual {rank + 1} from HoF: no valid fitness.")
                        continue

                    # Creăm un sub-director pentru fiecare individ din HoF
                    individual_output_sub_dir = run_output_dir / f"HoF_Indiv_{rank + 1}"
                    individual_output_sub_dir.mkdir(parents=True, exist_ok=True)

                    print(
                        f"  Saving results for HoF Individual {rank + 1} (Fitness: {individual_to_save.fitness.values[0]:.4f})...")

                    # Salvăm rezultatele pe setul de test mare pentru individul curent din HoF
                    evaluate_and_save_results(
                        instances_for_testing=test_insts,
                        best_individuals=[individual_to_save],  # Trimitem doar individul curent
                        toolbox=toolbox,
                        output_base_dir=individual_output_sub_dir / "TestSet_Big",
                        label=f"{config_name}_Run{run_idx + 1}_HoF_Indiv{rank + 1}_TestSet_Big"
                    )

                    # Salvăm rezultatele pe setul de test mic pentru individul curent din HoF
                    evaluate_and_save_results(
                        instances_for_testing=test_insts_small,
                        best_individuals=[individual_to_save],  # Trimitem doar individul curent
                        toolbox=toolbox,
                        output_base_dir=individual_output_sub_dir / "TestSet_Small",
                        label=f"{config_name}_Run{run_idx + 1}_HoF_Indiv{rank + 1}_TestSet_Small"
                    )
            else:
                print(f"  Run {run_idx + 1} did not find any valid individual in Hall of Fame.")

    # --- Generare Raport Sumar Final ---
    print("\n\n=== EXPERIMENT SUMMARY REPORT ===")
    with open(report_file_path, "w", encoding="utf-8") as f_report:
        f_report.write(f"Experiment Run: {timestamp}\n")
        f_report.write(f"Total Configurations Tested: {config_counter}\n")
        f_report.write(f"Runs per Configuration: {num_runs_per_config}\n")
        f_report.write(f"Parameters Grid: {json.dumps(param_grid, indent=2)}\n\n")  # Salvează gridul complet

        f_report.write("--- Best Individual from Each Run (Sorted by Fitness) ---\n")
        # Sortăm rezultatele pentru o mai bună lizibilitate în raport
        all_experiment_results.sort(key=lambda x: x["best_fitness"])

        for result in all_experiment_results:
            f_report.write(f"Config: {result['config_name']}, Run: {result['run_index']}\n")
            f_report.write(f"  Alpha: {result['alpha']}, Sel: {result['selection_strategy']}, "
                           f"CX: {result['crossover_strategy']}, Mut: {result['mutation_strategy']}\n")
            f_report.write(f"  Best Fitness: {result['best_fitness']:.4f}\n")
            f_report.write(f"  Simplified Individual: {result['best_individual_simplified_str']}\n")
            f_report.write(f"  Run Duration: {result['run_duration_s']:.2f}s\n\n")

        f_report.write("\n--- Overall Best Individual Found Across All Runs ---\n")
        if best_overall_individual_data["run"] != -1:  # Verificăm dacă a fost găsit un individ valid
            f_report.write(f"Best Overall Fitness: {best_overall_individual_data['fitness']:.4f}\n")
            f_report.write(f"Came from Config: {best_overall_individual_data['config']}\n")
            f_report.write(f"From Run Index: {best_overall_individual_data['run']}\n")
            f_report.write(f"Simplified Individual Formula: {best_overall_individual_data['individual_str']}\n")
        else:
            f_report.write("No valid best individual found across all runs.\n")

    print(f"\nFull experiment results and reports saved in: {experiment_output_root}")
    print(f"Summary report file: {report_file_path}")

    total_elapsed_time = time.time() - global_start
    print(f"Total experiment execution time: {total_elapsed_time:.1f}s")


if __name__ == "__main__":
    main()