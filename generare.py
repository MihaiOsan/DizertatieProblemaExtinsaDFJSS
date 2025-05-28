import random
import json
import os

def generate_breakdowns_for_machine(
        machine_id,
        max_time_horizon,
        mean_time_to_failure,
        mean_repair_time,
        seed=None
):
    """
    Generează evenimente de breakdown pentru o mașină (machine_id),
    până la un orizont de timp max_time_horizon.
    - Poisson pentru defecțiuni (MTTF).
    - Durată reparație ~ exponentială (MTTR).
    """
    breakdowns = []
    if seed is not None:
        random.seed(seed + machine_id)

    t = 0.0
    while True:
        delta_fail = random.expovariate(1.0 / mean_time_to_failure)
        t += delta_fail
        t = int(t)
        if t > max_time_horizon:
            break
        repair = random.expovariate(1.0 / mean_repair_time)
        repair = int(repair) + 1
        breakdowns.append({
            "start_time": t,
            "repair_time": repair
        })
        t += repair

    return breakdowns


def generate_flex_dataset_with_breakdowns(
        num_jobs,
        num_machines,
        machine_util,
        ec_percent,
        min_num_ops,
        max_num_ops,
        min_num_candidate_machines,
        max_num_candidate_machines,
        min_proc_time,
        max_proc_time,
        allowance_factors,
        # breakdown params
        max_time_horizon,
        mean_time_to_failure,
        mean_repair_time,
        seed=None,
        etpc_min_lapse=5,
        etpc_max_lapse=15,
        # cancelled jobs params
        cancelled_job_frac=0.0,
        cancel_delay_range=(10, 1000)
):
    """
    Generare set de date 'flexible job-shop' cu:
      - ETPC (Extended Technical Precedence Constraints)
      - Evenimente de tip 'breakdown' la mașini
      - Evenimente de anulare joburi ('cancelled jobs'), doar după sosire
    """
    random.seed(seed)

    # 1) Calculăm 'tv'
    avg_num_ops = (min_num_ops + max_num_ops) / 2
    avg_p = (min_proc_time + max_proc_time) / 2
    tv = (avg_num_ops * avg_p) / (num_machines * machine_util)

    # 2) Generăm arrival_times ~ Exp(1/tv)
    arrivals = [0]*num_machines
    current_arrival = 0.0
    for j_idx in range(num_jobs-num_machines):
        delta = int(random.expovariate(1.0 / tv))
        current_arrival += delta
        arrivals.append(current_arrival)

    # 3) Generăm weights (4:2:1)
    n4 = int(0.2 * num_jobs)
    n2 = int(0.6 * num_jobs)
    n1 = num_jobs - (n4 + n2)
    weights_list = [4] * n4 + [2] * n2 + [1] * n1
    random.shuffle(weights_list)

    # 4) Generăm joburile (flexibile + timpi diferiți)
    jobs_data = []
    for j_idx in range(num_jobs):
        ops_count = random.randint(min_num_ops, max_num_ops)
        total_proc_time_est = 0.0
        operations = []

        for _op in range(ops_count):
            num_candidates = random.randint(min_num_candidate_machines,
                                            max_num_candidate_machines)
            candidate_machines = random.sample(range(num_machines), num_candidates)

            machine_dict = {}
            for mc_id in candidate_machines:
                ptime = random.randint(min_proc_time, max_proc_time)
                machine_dict[mc_id] = ptime

            machine_dict_sorted = dict(sorted(machine_dict.items(), key=lambda x: x[0]))
            best_p = min(machine_dict_sorted.values())
            total_proc_time_est += best_p

            operations.append({
                "candidate_machines": machine_dict_sorted
            })

        AF = random.choice(allowance_factors)
        dd = arrivals[j_idx] + AF * total_proc_time_est
        dd = int(dd)

        jobs_data.append({
            "id": j_idx,
            "arrival_time": arrivals[j_idx],
            "weight": weights_list[j_idx],
            "due_date": dd,
            "operations": operations
        })

    # 5bis) Generăm evenimente de tip "cancelled_jobs" doar DUPĂ arrival_time + delay
    cancelled_jobs_list = []
    cancel_job_ids = []
    if cancelled_job_frac > 0:
        num_cancelled = max(1, int(cancelled_job_frac * num_jobs))
        cancel_job_ids = random.sample(range(num_jobs), num_cancelled)
        for jid in cancel_job_ids:
            job_arrival = jobs_data[jid]['arrival_time']
            min_cancel_time = int(job_arrival + cancel_delay_range[0])
            max_cancel_time = int(job_arrival + cancel_delay_range[1])
            # Dacă nu există interval valid, sărim peste anulare
            if min_cancel_time < max_cancel_time:
                cancel_time = random.randint(min_cancel_time, max_cancel_time)
                cancelled_jobs_list.append({
                    "job_id": jid,
                    "time": cancel_time
                })
            else:
                pass  # Nu-l adăugăm dacă nu e posibil

    # 5) Generăm ETPC constraints, fără joburile anulate
    num_pairs = max(1, round(ec_percent * num_jobs))
    used_pairs = set()
    etpc_constraints = []

    for _ in range(num_pairs):
        for _try in range(1000):  # safety loop
            j1 = random.randint(0, num_jobs - 1)
            j2 = random.randint(0, num_jobs - 1)
            # Excludem joburile anulate
            if (j1 != j2 and
                (j1, j2) not in used_pairs and (j2, j1) not in used_pairs and
                (not cancel_job_ids or (j1 not in cancel_job_ids and j2 not in cancel_job_ids)) and arrivals[j1] < jobs_data[j2]['due_date']
            ):
                used_pairs.add((j1, j2))
                fore_op_count = len(jobs_data[j1]["operations"])
                hind_op_count = len(jobs_data[j2]["operations"])
                fo = random.randint(0, fore_op_count - 1)
                ho = random.randint(0, hind_op_count - 1)
                lapse = random.randint(etpc_min_lapse, etpc_max_lapse)

                etpc_constraints.append({
                    "fore_job": j1,
                    "fore_op_idx": fo,
                    "hind_job": j2,
                    "hind_op_idx": ho,
                    "time_lapse": lapse
                })
                break
        else:
            print("[WARN] Nu s-au putut genera suficiente perechi ETPC fără joburi anulate.")

    # 6) Generăm breakdown-urile
    machine_breakdowns = {}
    for m_id in range(num_machines):
        b_events = generate_breakdowns_for_machine(
            m_id,
            max_time_horizon,
            mean_time_to_failure,
            mean_repair_time,
            seed=seed
        )
        machine_breakdowns[m_id] = b_events

    # 7) Adaugăm dynamic_events dacă e cazul
    dynamic_events = {}
    if cancelled_jobs_list:
        dynamic_events["cancelled_jobs"] = cancelled_jobs_list

    dataset = {
        "jobs": jobs_data,
        "etpc_constraints": etpc_constraints,
        "machine_breakdowns": machine_breakdowns,
        "params": {
            "num_jobs": num_jobs,
            "num_machines": num_machines,
            "machine_util": machine_util,
            "ec_percent": ec_percent,
            "ops_range": (min_num_ops, max_num_ops),
            "candidate_machines_range": (min_num_candidate_machines, max_num_candidate_machines),
            "proc_time_range": (min_proc_time, max_proc_time),
            "allowance_factors": allowance_factors,
            "etpc_lapse_range": (etpc_min_lapse, etpc_max_lapse),
            "mean_time_to_failure": mean_time_to_failure,
            "mean_repair_time": mean_repair_time,
            "max_time_horizon": max_time_horizon,
            "seed": seed,
            "cancelled_job_frac": cancelled_job_frac,
            "cancel_delay_range": cancel_delay_range
        }
    }

    if dynamic_events:
        dataset["dynamic_events"] = dynamic_events

    return dataset


if __name__ == "__main__":
    os.makedirs("dynamic_data/extended/training_sets", exist_ok=True)
    os.makedirs("dynamic_data/extended/test_sets", exist_ok=True)

    train_instances_no = 4
    test_instances_no = 2

    num_jobs_train = 350
    num_jobs_test = 150
    num_machines_range = (10, 13, 16)

    ops_range_train = (5, 15)
    ops_range_test = (6, 12)

    candidate_mc_train = (3, 6)
    candidate_mc_test = (3, 6)

    proc_range_train = (25, 60)
    proc_range_test = (25, 75)

    allowance_factors = [2, 6, 8]

    max_time_horizon = 8500.0
    mean_time_to_failure = 200.0
    mean_repair_time = 20.0

    train_util_list = [0.70, 0.85, 0.95]
    train_etpc_list = [0.05, 0.10, 0.15]

    test_util_list = [0.75, 0.95]
    test_etpc_list = [0.08, 0.15]

    cancelled_job_frac = 0.05   # 10% din joburi anulate
    cancel_delay_range = (250, 1000) # Anulare la 10-1000 timp după sosire

    # Generez seturi TRAINING
    train_id = 0
    for vers in range(train_instances_no):
        for util in train_util_list:
            for ec in train_etpc_list:
                for num_machines in num_machines_range:
                    seed_val = 1000 + train_id
                    ds = generate_flex_dataset_with_breakdowns(
                        num_jobs=num_jobs_train,
                        num_machines=num_machines,
                        machine_util=util,
                        ec_percent=ec,
                        min_num_ops=ops_range_train[0],
                        max_num_ops=ops_range_train[1],
                        min_num_candidate_machines=candidate_mc_train[0],
                        max_num_candidate_machines=candidate_mc_train[1],
                        min_proc_time=proc_range_train[0],
                        max_proc_time=proc_range_train[1],
                        allowance_factors=allowance_factors,
                        max_time_horizon=max_time_horizon,
                        mean_time_to_failure=mean_time_to_failure,
                        mean_repair_time=mean_repair_time,
                        seed=seed_val,
                        etpc_min_lapse=5,
                        etpc_max_lapse=15,
                        cancelled_job_frac=cancelled_job_frac,
                        cancel_delay_range=cancel_delay_range
                    )
                    fname = f"dynamic_data/extended/training_sets/train_flex_events_{train_id}_util{util}_ec{ec}_nm{num_machines}_v{vers}.json"
                    with open(fname, "w") as f:
                        json.dump(ds, f, indent=2)

                    print(f"[TRAIN FLEX+BREAK+CANCEL] Generated: {fname}")
                    train_id += 1

    # Generez seturi TEST
    test_id = 0
    for vers in range(test_instances_no):
        for util in test_util_list:
            for ec in test_etpc_list:
                for num_machines in num_machines_range:
                    seed_val = 2000 + test_id
                    ds_test = generate_flex_dataset_with_breakdowns(
                        num_jobs=num_jobs_test,
                        num_machines=num_machines,
                        machine_util=util,
                        ec_percent=ec,
                        min_num_ops=ops_range_test[0],
                        max_num_ops=ops_range_test[1],
                        min_num_candidate_machines=candidate_mc_test[0],
                        max_num_candidate_machines=candidate_mc_test[1],
                        min_proc_time=proc_range_test[0],
                        max_proc_time=proc_range_test[1],
                        allowance_factors=allowance_factors,
                        max_time_horizon=max_time_horizon,
                        mean_time_to_failure=mean_time_to_failure,
                        mean_repair_time=mean_repair_time,
                        seed=seed_val,
                        etpc_min_lapse=5,
                        etpc_max_lapse=10,
                        cancelled_job_frac=cancelled_job_frac,
                        cancel_delay_range=cancel_delay_range
                    )
                    fname_test = f"dynamic_data/extended/test_sets/test_flex_events_{test_id}_util{util}_ec{ec}_nm{num_machines}_v{vers}.json"
                    with open(fname_test, "w") as f:
                        json.dump(ds_test, f, indent=2)

                    print(f"[TEST FLEX+BREAK+CANCEL] Generated: {fname_test}")
                    test_id += 1

    # Generez seturi TEST
    test_id = 0
    for vers in range(1):
        for util in test_util_list:
            for ec in test_etpc_list:
                for num_machines in num_machines_range:
                    seed_val = 3000 + test_id
                    ds_test = generate_flex_dataset_with_breakdowns(
                        num_jobs=25,
                        num_machines=num_machines,
                        machine_util=util,
                        ec_percent=ec,
                        min_num_ops=ops_range_test[0],
                        max_num_ops=ops_range_test[1],
                        min_num_candidate_machines=candidate_mc_test[0],
                        max_num_candidate_machines=candidate_mc_test[1],
                        min_proc_time=proc_range_test[0],
                        max_proc_time=proc_range_test[1],
                        allowance_factors=allowance_factors,
                        max_time_horizon=max_time_horizon,
                        mean_time_to_failure=mean_time_to_failure,
                        mean_repair_time=mean_repair_time,
                        seed=seed_val,
                        etpc_min_lapse=5,
                        etpc_max_lapse=10,
                        cancelled_job_frac=cancelled_job_frac,
                        cancel_delay_range=cancel_delay_range
                    )
                    fname_test = f"dynamic_data/extended/test_sets_small/test_small_flex_events_{test_id}_util{util}_ec{ec}_nm{num_machines}_v{vers}.json"
                    with open(fname_test, "w") as f:
                        json.dump(ds_test, f, indent=2)

                    print(f"[TEST FLEX+BREAK+CANCEL] Generated: {fname_test}")
                    test_id += 1
