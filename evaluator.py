import copy
import operator

from deap import tools, algorithms, gp, base
import random as rd
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

from data_reader import FJSPInstance, Job, Operation, ETPCConstraint, BaseEvent, BreakdownEvent, AddJobDynamicEvent, \
    CancelJobEvent

from scheduler import evaluate_individual

# --- Functii Utilitare pentru Metrici ---
TUPLE_FIELDS_EVAL = {"job": 0, "op": 1, "machine": 2, "start": 3, "end": 4}


def field_eval(op_tuple: Tuple, name: str) -> Any:
    if not isinstance(op_tuple, tuple) or len(op_tuple) < max(TUPLE_FIELDS_EVAL.values()) + 1:
        return None
    return op_tuple[TUPLE_FIELDS_EVAL[name]]


def get_job_completion_times_from_schedule(
        schedule_tuples: List[Tuple]
) -> Dict[int, float]:
    job_completion_times: Dict[int, float] = defaultdict(float)
    for op_tuple in schedule_tuples:
        job_sim_id = field_eval(op_tuple, "job")
        op_end_time = field_eval(op_tuple, "end")
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
        if job_obj.is_cancelled_sim:
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


def calculate_mean_waiting_time(
        schedule_tuples: List[Tuple],
        fjsp_instance: FJSPInstance
) -> float:
    if not schedule_tuples: return 0.0
    ops_by_job_sim_id: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    scheduled_job_sim_ids = set()
    for op_tuple in schedule_tuples:
        job_sim_id = field_eval(op_tuple, "job")
        if job_sim_id is None: continue
        ops_by_job_sim_id[job_sim_id].append(
            (field_eval(op_tuple, "op"),
             float(field_eval(op_tuple, "start")),
             float(field_eval(op_tuple, "end")))
        )
        scheduled_job_sim_ids.add(job_sim_id)
    total_wait_time = 0.0
    num_jobs_with_wait_calculated = 0
    for job_sim_id in scheduled_job_sim_ids:
        job_object = fjsp_instance.get_job_by_sim_id(job_sim_id)
        if not job_object or job_object.is_cancelled_sim: continue
        job_ops_in_schedule = ops_by_job_sim_id.get(job_sim_id, [])
        if not job_ops_in_schedule: continue
        job_ops_in_schedule.sort(key=lambda x: x[0])
        current_job_arrival_time = job_object.arrival_time
        prev_op_finish_time_or_arrival = current_job_arrival_time
        job_wait_time_for_this_job = 0.0
        for _op_idx, op_start, op_end in job_ops_in_schedule:
            job_wait_time_for_this_job += max(0.0, op_start - prev_op_finish_time_or_arrival)
            prev_op_finish_time_or_arrival = op_end
        total_wait_time += job_wait_time_for_this_job
        num_jobs_with_wait_calculated += 1
    return total_wait_time / num_jobs_with_wait_calculated if num_jobs_with_wait_calculated > 0 else 0.0


def multi_instance_fitness(individual: gp.PrimitiveTree,
                           fjsp_instances: List[FJSPInstance],
                           toolbox: base.Toolbox,
                           alpha: float = 0.7
                           ) -> Tuple[float,]:
    if individual is None:
        return (float('inf'),)
    alpha = max(0.0, min(1.0, alpha))
    weight_for_twt = 1.0 - alpha
    try:
        priority_func = toolbox.compile(expr=individual)
    except Exception as e:
        return (float('inf'),)
    total_combined_score = 0.0
    num_valid_instances_evaluated = 0
    for instance_obj in fjsp_instances:
        try:
            makespan, schedule_tuples = evaluate_individual(
                individual=priority_func,
                fjsp_instance=instance_obj,
                toolbox=toolbox
            )
        except Exception as e_eval:
            makespan = float('inf');
            schedule_tuples = []
        twt = calculate_total_weighted_tardiness(schedule_tuples, instance_obj, makespan)
        score_for_instance = float('inf')
        if makespan != float('inf'):
            if twt == float('inf') and weight_for_twt > 1e-9:
                score_for_instance = float('inf')
            else:
                score_for_instance = alpha * makespan + weight_for_twt * twt
        if score_for_instance != float('inf'):
            total_combined_score += score_for_instance
            num_valid_instances_evaluated += 1
    if num_valid_instances_evaluated == 0:
        return (float('inf'),)
    return (total_combined_score / num_valid_instances_evaluated,)


def run_genetic_program(train_instances: List[FJSPInstance],
                        toolbox: base.Toolbox,
                        ngen: int = 10,
                        pop_size: int = 20,
                        alpha: float = 0.7,
                        halloffame_size: int = 1,
                        cxpb: float = 0.5,
                        mutpb: float = 0.3,
                        MAX_DEPTH = 7
                        ):
    toolbox.register("evaluate",
                     multi_instance_fitness,
                     fjsp_instances=train_instances,
                     toolbox=toolbox,
                     alpha=alpha)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    if not hasattr(toolbox, 'expr') or not hasattr(toolbox, 'pset'):
        raise AttributeError("Toolbox not fully configured for mutation. Missing 'expr' or 'pset'.")
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=toolbox.pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(halloffame_size)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else float('inf'))
    stats_size = tools.Statistics(len)
    stats_best_ind_obj = tools.Statistics(key=lambda ind: ind)

    safe_avg = lambda x: sum(xi for xi in x if xi != float('inf')) / len([xi for xi in x if xi != float('inf')]) if len(
        [xi for xi in x if xi != float('inf')]) > 0 else 0.0
    safe_min = lambda x: min(xi for xi in x if xi != float('inf')) if any(xi != float('inf') for xi in x) else float(
        'inf')
    safe_max = lambda x: max(xi for xi in x if xi != float('inf')) if any(xi != float('inf') for xi in x) else float(
        '-inf')

    def safe_std(x_list):
        finite_vals = [xi for xi in x_list if xi != float('inf')]
        if len(finite_vals) < 2: return 0.0
        mean_val = sum(finite_vals) / len(finite_vals)
        return (sum((xi - mean_val) ** 2 for xi in finite_vals) / len(finite_vals)) ** 0.5

    stats_fit.register("avg", safe_avg)
    stats_fit.register("std", safe_std)
    stats_fit.register("min", safe_min)
    stats_fit.register("max", safe_max)

    stats_size.register("avg", lambda x: sum(x) / len(x) if len(x) > 0 else 0.0)
    stats_size.register("min", min)
    stats_size.register("max", max)

    # `min` va alege individul bazat pe `ind.fitness`
    stats_best_ind_obj.register("best", lambda pop_list: min(pop_list, key=lambda
        ind: ind.fitness if ind.fitness.valid else float('inf')))

    mstats = tools.MultiStatistics(fitness=stats_fit)

    final_pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=cxpb, mutpb=mutpb,
        ngen=ngen,
        stats=mstats,
        halloffame=hof,
        verbose=True
    )

    print("\n--- Best Individual per Generation (from Logbook) ---")
    if logbook:
        print(f"Gen\t{'MinFitness':<15}\tBest Individual Tree of Generation")
        print("-" * 80)
        for gen_data in logbook:
            gen_num = gen_data["gen"]
            min_fitness_val = gen_data.get("fitness", {}).get("min", float('inf'))
            best_ind_tree_of_gen = None
            if "best_individual" in gen_data and "best" in gen_data["best_individual"]:
                best_ind_tree_of_gen = gen_data["best_individual"]["best"]
            print(
                f"{gen_num}\t{min_fitness_val:<15.4f}\t{str(best_ind_tree_of_gen) if best_ind_tree_of_gen else 'N/A'}")
    else:
        print("Logbook is empty or not generated.")

    print("\nGenetic program finished.")
    return hof


def run_genetic_program_subsample(instances: List[FJSPInstance],
                                  toolbox: base.Toolbox,
                                  ngen=10, pop_size=20,
                                  chunk_size=5,
                                  subset_rate=0.5,
                                  alpha=0.7,
                                  cxpb=0.5, mutpb=0.3,
                                  halloffame_size=5,
                                  pop_init=None):
    if pop_init is None:
        pop = toolbox.population(n=pop_size)
    else:
        pop = pop_init

    hof = tools.HallOfFame(halloffame_size)
    num_chunks = (ngen + chunk_size - 1) // chunk_size
    gens_done = 0

    stats_fit_sub = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else float('inf'))
    stats_best_ind_sub = tools.Statistics(key=lambda ind: ind)

    safe_avg_sub = lambda x: sum(xi for xi in x if xi != float('inf')) / len(
        [xi for xi in x if xi != float('inf')]) if len([xi for xi in x if xi != float('inf')]) > 0 else 0.0
    safe_min_sub = lambda x: min(xi for xi in x if xi != float('inf')) if any(
        xi != float('inf') for xi in x) else float('inf')

    stats_fit_sub.register("avg", safe_avg_sub)
    stats_fit_sub.register("min", safe_min_sub)
    # --- MODIFICARE: Specificam `key` pentru functia `min` aplicata pe indivizi ---
    stats_best_ind_sub.register("best", lambda pop_list: min(pop_list, key=lambda
        ind: ind.fitness if ind.fitness.valid else float('inf')))

    mstats_sub = tools.MultiStatistics(fitness=stats_fit_sub)

    for chunk_idx in range(num_chunks):
        gens_left = ngen - gens_done
        gens_here = min(chunk_size, gens_left)
        if gens_here <= 0: break

        print(
            f"=== Chunk {chunk_idx + 1}/{num_chunks}, Generations in chunk: {gens_here} (Total done: {gens_done}) ===")

        if subset_rate < 1.0 and len(instances) > 1:
            k = max(1, int(len(instances) * subset_rate))
            chosen_insts_subset = rd.sample(instances, k)
        else:
            chosen_insts_subset = instances

        toolbox.register("evaluate", multi_instance_fitness,
                         fjsp_instances=chosen_insts_subset, toolbox=toolbox, alpha=alpha)

        if not hasattr(toolbox, 'expr') or not hasattr(toolbox, 'pset'):
            raise AttributeError("Toolbox not fully configured for mutation in subsample. Missing 'expr' or 'pset'.")
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=toolbox.pset)

        pop, logbook_chunk = algorithms.eaSimple(
            pop, toolbox,
            cxpb=cxpb, mutpb=mutpb,
            ngen=gens_here,
            stats=mstats_sub,
            halloffame=hof,
            verbose=True
        )

        print(f"\n--- Best Individual per Generation (Chunk {chunk_idx + 1}) ---")
        if logbook_chunk:
            print(f"Gen\t{'MinFitness':<15}\tBest Individual Tree (Chunk)")
            print("-" * 80)
            for gen_data_chunk in logbook_chunk:
                gen_num_chunk = gen_data_chunk["gen"]
                min_fitness_val_chunk = gen_data_chunk.get("fitness", {}).get("min", float('inf'))
                best_ind_tree_of_gen_chunk = None
                if "best_ind_of_chunk" in gen_data_chunk and "best" in gen_data_chunk["best_ind_of_chunk"]:
                    best_ind_tree_of_gen_chunk = gen_data_chunk["best_ind_of_chunk"]["best"]
                print(
                    f"{gens_done + gen_num_chunk}\t{min_fitness_val_chunk:<15.4f}\t{str(best_ind_tree_of_gen_chunk) if best_ind_tree_of_gen_chunk else 'N/A'}")
        else:
            print("Chunk logbook is empty or not generated.")
        gens_done += gens_here
    return hof
