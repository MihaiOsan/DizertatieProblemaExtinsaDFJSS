# main.py
from __future__ import annotations

import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union
import re
import numpy as np
import csv

from deap import gp

from data_reader import load_instances_from_directory, FJSPInstance, BreakdownEvent, Job
from evaluator import run_genetic_program, calculate_total_weighted_tardiness, evaluate_individual, \
    get_job_completion_times_from_schedule
from ganttPlot import plot_gantt
from gpSetup import create_toolbox
from simpleTree import simplify_individual, tree_str, infix_str

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TRAIN_DIR = Path("dynamic_data/extended/test_sets")
TEST_DIR = Path("dynamic_data/extended/test_sets")
TEST_DIR_SMALL = Path("dynamic_data/extended/test_sets_small")
POP_SIZE = 50
N_GENERATIONS = 25
N_WORKERS = 5
MAX_HOF = 5

BASE_OUTPUT_DIR = Path("rezultate/genetic")



# ----------------------------------------------------------
# TUPLE_FIELDS si field()
# ----------------------------------------------------------
TUPLE_FIELDS = {"job": 0, "op": 1, "machine": 2, "start": 3, "end": 4}


def field(op_tuple: Tuple, name: str) -> Any:
    if not isinstance(op_tuple, tuple) or len(op_tuple) < max(TUPLE_FIELDS.values()) + 1:
        return None
    return op_tuple[TUPLE_FIELDS[name]]


# ---------------------------------------------------------------------------
#  UTILITARE
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
#  FUNCTII DE METRICA
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
        individual_fitness_train = individual_tree_original.fitness.values[0] if individual_tree_original.fitness.valid else float("inf")
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
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("--- Loading Training Instances ---")
    train_insts: List[FJSPInstance] = load_instances_from_directory(str(TRAIN_DIR))
    print(f"Loaded {len(train_insts)} training instances.\n")
    print("--- Loading Test Instances ---")
    test_insts: List[FJSPInstance] = load_instances_from_directory(str(TEST_DIR))
    print(f"Loaded {len(test_insts)} test instances.\n")
    if not train_insts:
        print("No training instances loaded. Exiting.");
        return
    toolbox = create_toolbox(np=N_WORKERS)
    print("\n=== GP TRAINING ===")
    hof = run_genetic_program(
        train_instances=train_insts, toolbox=toolbox,
        ngen=N_GENERATIONS, pop_size=POP_SIZE,
        halloffame_size=MAX_HOF, alpha=0.2
    )
    best_individuals: List[gp.PrimitiveTree] = list(hof)[:MAX_HOF]

    # ---- Test Set Mare ----
    evaluate_and_save_results(
        instances_for_testing=test_insts,
        best_individuals=best_individuals,
        toolbox=toolbox,
        output_base_dir=BASE_OUTPUT_DIR / "TestSet_Big",
        label="TestSet_Big"
    )

    # ---- Test Set Mic ----
    test_insts_small = load_instances_from_directory(str(TEST_DIR_SMALL))
    print(f"Loaded {len(test_insts_small)} small test instances.\n")
    evaluate_and_save_results(
        instances_for_testing=test_insts_small,
        best_individuals=best_individuals,
        toolbox=toolbox,
        output_base_dir=BASE_OUTPUT_DIR / "TestSet_Small",
        label="TestSet_Small"
    )

    total_elapsed_time = time.time() - global_start
    print(f"\nResults written to per-individual files in '{BASE_OUTPUT_DIR}'.")
    print(f"Total execution time: {total_elapsed_time:.1f}s")



if __name__ == "__main__":
    main()
