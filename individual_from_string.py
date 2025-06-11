from __future__ import annotations
import time
from utils import *
from deap import gp
from data_reader import load_instances_from_directory, FJSPInstance, BreakdownEvent, Job
from evaluator import evaluate_individual, get_job_completion_times_from_schedule
from ganttPlot import plot_gantt
from gpSetup import create_toolbox
from simpleTree import simplify_individual, tree_str, infix_str

# --- TUPLE_FIELDS și field() (copiate din main.py) ---
TUPLE_FIELDS = {"job": 0, "op": 1, "machine": 2, "start": 3, "end": 4}


# --- FUNCTIA PRINCIPALA DE EVALUARE SI SALVARE (adaptată pentru a testa indivizi recreați) ---
def evaluate_and_save_results(
        instances_for_testing: List[FJSPInstance],
        individual_tree: gp.PrimitiveTree,
        toolbox,
        output_base_dir: Path,
        label: str = ""
):
    """
    Evaluates a single genetic program (individual_tree) on a set of test instances
    and saves detailed results including metrics, CSVs, and Gantt charts.
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    # Verificăm dacă atributul 'fitness' există și este valid (opțional, pentru afișare)
    individual_fitness_train = "N/A"
    if hasattr(individual_tree, 'fitness') and individual_tree.fitness.valid:
        # Formatăm valoarea fitness-ului dacă este disponibilă
        individual_fitness_train = f"{individual_tree.fitness.values[0]:.4f}"

    simplified_individual_tree = simplify_individual(individual_tree, toolbox.pset)
    ind_size_original = len(individual_tree)
    ind_depth_original = individual_tree.height
    ind_size_simplified = len(simplified_individual_tree)
    ind_depth_simplified = simplified_individual_tree.height
    formula_infix_str = infix_str(simplified_individual_tree)
    formula_str_sanitized = sanitize_filename_str(formula_infix_str)

    summary_results_file_path = output_base_dir / "summary_overall_metrics.txt"
    per_instance_details_dir = output_base_dir / "per_instance_details"
    per_instance_details_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Testing Individual ({label} Initial Fitness={individual_fitness_train}) ---")
    print(f"Output directory: {output_base_dir}")
    print(f"Original Tree: {str(individual_tree)}")
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
        outf_summary.write(f"=== Individual under test ===\n")
        outf_summary.write(f"Initial_Fitness: {individual_fitness_train}\n")
        outf_summary.write(f"Original_Size: {ind_size_original}, Original_Depth: {ind_depth_original}\n")
        outf_summary.write(f"Original_Tree (string): {str(individual_tree)}\n")
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

            actual_makespan, schedule_tuples = evaluate_individual(
                individual=eval_priority_func,
                fjsp_instance=fjsp_instance_obj,
                toolbox=toolbox
            )
            eval_elapsed_time = time.perf_counter() - time_eval_start

            _total_idle, avg_idle, std_idle, _total_busy, avg_busy_util, \
                idle_list_per_m, busy_list_per_m, ops_counts_m = \
                calc_machine_metrics(schedule_tuples, fjsp_instance_obj.num_machines, actual_makespan)

            job_metrics = calc_job_related_metrics(schedule_tuples, fjsp_instance_obj, actual_makespan)

            all_instance_makespans.append(actual_makespan)
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
                               f"{actual_makespan:<7.2f} {job_metrics['total_weighted_tardiness']:<7.2f} "
                               f"{avg_idle:<9.2f} {std_idle:<9.2f} "
                               f"{job_metrics['avg_wait_time']:<9.2f} {job_metrics['std_wait_time']:<9.2f} "
                               f"{job_metrics['avg_tardiness']:<9.2f} {job_metrics['std_tardiness']:<9.2f} "
                               f"{job_metrics['num_tardy_jobs']:<6} {avg_busy_util * 100:<8.2f}% "
                               f"{eval_elapsed_time:<10.3f}\n")

            print(
                f" Inst: {instance_file_stem:<40} MS={actual_makespan:<7.2f} TWT={job_metrics['total_weighted_tardiness']:<7.2f} "
                f"AvgIdle={avg_idle:<6.2f} AvgWait={job_metrics['avg_wait_time']:<6.2f} #Tardy={job_metrics['num_tardy_jobs']:<3} "
                f"AvgBusy%={avg_busy_util * 100:<5.2f}%")

            generate_per_machine_metrics_csv(
                idle_list_per_m, busy_list_per_m, ops_counts_m,
                fjsp_instance_obj.num_machines, actual_makespan,
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
                title=f"{instance_file_stem} - (MS={actual_makespan:.2f}, TWT={job_metrics['total_weighted_tardiness']:.2f})",
                save_path=str(current_instance_output_dir / gantt_file_name)
            )

        num_test_cases = len(instances_for_testing)

        outf_summary.write("\n--- OVERALL AVERAGES for this Individual (across test instances) ---\n")
        outf_summary.write(f"Average_MS : {safe_avg_list(all_instance_makespans):.2f}\n")
        outf_summary.write(f"Average_TWT : {safe_avg_list(all_instance_total_weighted_tardiness):.2f}\n")
        outf_summary.write(f"Average_Avg_Machine_Idle : {safe_avg_list(all_instance_avg_machine_idles):.2f}\n")
        outf_summary.write(f"Average_Std_Machine_Idle : {safe_avg_list(all_instance_std_dev_machine_idles):.2f}\n")
        outf_summary.write(f"Average_Avg_Job_Wait : {safe_avg_list(all_instance_avg_job_waits):.2f}\n")
        outf_summary.write(f"Average_Std_Job_Wait : {safe_avg_list(all_instance_std_dev_job_waits):.2f}\n")
        outf_summary.write(f"Average_Avg_Tardiness : {safe_avg_list(all_instance_avg_job_tardiness):.2f}\n")
        outf_summary.write(f"Average_Std_Tardiness : {safe_avg_list(all_instance_std_dev_job_tardiness):.2f}\n")
        outf_summary.write(f"Average_Num_Tardy_Jobs : {safe_avg_list(all_instance_num_tardy_jobs):.2f}\n")
        outf_summary.write(f"Average_Avg_Machine_Utilization: {safe_avg_list(all_instance_avg_busy_util) * 100:.2f}%\n")
        outf_summary.write(f"Average_Eval_Time : {safe_avg_list(all_instance_evaluation_times):.3f}s\n")

    print("\n ------- OVERALL AVERAGES for this Individual (across test instances) -------")
    print(f" Avg Test MS = {safe_avg_list(all_instance_makespans):.2f}")
    print(f" Avg Test TWT = {safe_avg_list(all_instance_total_weighted_tardiness):.2f}")
    print(f" Avg Test Avg_Machine_Idle = {safe_avg_list(all_instance_avg_machine_idles):.2f}")
    print(f" Avg Test Std_Machine_Idle = {safe_avg_list(all_instance_std_dev_machine_idles):.2f}")
    print(f" Avg Test Avg_Job_Wait = {safe_avg_list(all_instance_avg_job_waits):.2f}")
    print(f" Avg Test Std_Job_Wait = {safe_avg_list(all_instance_std_dev_job_waits):.2f}")
    print(f" Avg Test Avg_Tardiness = {safe_avg_list(all_instance_avg_job_tardiness):.2f}")
    print(f" Avg Test Std_Tardiness = {safe_avg_list(all_instance_std_dev_job_tardiness):.2f}")
    print(f" Avg Test Num_Tardy_Jobs = {safe_avg_list(all_instance_num_tardy_jobs):.2f}")
    print(f" Avg Test Avg_Machine_Utilization= {safe_avg_list(all_instance_avg_busy_util) * 100:.2f}%")
    print(f" Avg Eval Time = {safe_avg_list(all_instance_evaluation_times):.3f}s")


# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    #lista cu indivizii pentru teste
    individuals_to_test = [
        "max(mul(SLK, mul(1, PT)), 1)"
    ]

    # Configurează căile către directoarele de test
    TEST_DIR = Path("dynamic_data/extended/test_sets")
    TEST_DIR_SMALL = Path("dynamic_data/extended/test_sets_small")

    # Directorul de ieșire
    OUTPUT_BASE_DIR = Path("rezultate/individual_tests")
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Încarcă instanțele de test
    print("--- Loading Test Instances ---")
    test_insts: List[FJSPInstance] = load_instances_from_directory(str(TEST_DIR))
    print(f"Loaded {len(test_insts)} training instances.\n")

    test_insts_small: List[FJSPInstance] = load_instances_from_directory(str(TEST_DIR_SMALL))
    print(f"Loaded {len(test_insts_small)} small test instances.\n")

    if not test_insts and not test_insts_small:
        print("No test instances loaded. Exiting.")
        exit()

    # Recrează Toolbox-ul cu același PrimitiveSet folosit la antrenament.
    # `create_toolbox` din `gpSetup.py` ar trebui să se ocupe de asta.
    N_WORKERS = 6
    toolbox = create_toolbox(np=N_WORKERS)

    # Iterăm prin șirurile indivizilor și îi testăm
    for i, individual_string in enumerate(individuals_to_test):
        try:
            print(f"\n=== Testing Individual {i + 1}/{len(individuals_to_test)}: {individual_string} ===")

            # Recreează individul din șirul de caractere ca un simplu PrimitiveTree.
            # nu mai depinde de atributul `fitness`.
            recreated_individual = gp.PrimitiveTree.from_string(individual_string, toolbox.pset)

            # Director de ieșire specific pentru acest individ testat
            individual_output_dir = OUTPUT_BASE_DIR / f"Tested_Indiv_{i + 1}_{sanitize_filename_str(individual_string, 50)}"
            individual_output_dir.mkdir(parents=True, exist_ok=True)

            # Testează individul recreat pe seturile de date

            # Test pe setul de test mare
            evaluate_and_save_results(
                instances_for_testing=test_insts,
                individual_tree=recreated_individual,
                toolbox=toolbox,
                output_base_dir=individual_output_dir / "TestSet_Big",
                label=f"Manual_Test_Big_{i + 1}"
            )

            # Test pe setul de test mic
            evaluate_and_save_results(
                instances_for_testing=test_insts_small,
                individual_tree=recreated_individual,
                toolbox=toolbox,
                output_base_dir=individual_output_dir / "TestSet_Small",
                label=f"Manual_Test_Small_{i + 1}"
            )

        except Exception as e:
            print(f"ERROR testing individual '{individual_string}': {e}")
            print("  Asigură-te că `gpSetup.py` definește corect toate primitivele/terminalele utilizate în string.")
            print("  Verifică și dacă stringul conține caractere speciale sau este malformat.")
            # Continuăm la următorul individ chiar dacă apare o eroare pentru acesta.

    print("\nTestarea individuală a fost finalizată.")