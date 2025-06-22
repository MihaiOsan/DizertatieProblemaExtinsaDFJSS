import json
import os

from generare import generate_flex_dataset_with_breakdowns

if __name__ == "__main__":
    os.makedirs("/Users/mihaiosan/PycharmProjects/DizertatieProblemaExtinsaDFJSS/dynamic_data/fan21/training_sets", exist_ok=True)
    os.makedirs("/Users/mihaiosan/PycharmProjects/DizertatieProblemaExtinsaDFJSS/dynamic_data/fan21/test_sets", exist_ok=True)

    train_instances_no = 3
    test_instances_no = 5

    # Ex. de parametri generali
    num_jobs_train = 100
    num_jobs_test = 500
    num_machines_range = [10]

    ops_range_train = (1, 10)
    ops_range_test = (3, 6)

    candidate_mc_train = [1]
    candidate_mc_test = [1]

    proc_range_train = (5, 20)
    proc_range_test = (5, 15)

    allowance_factors = [2, 6, 8]

    # Definiții breakdown
    max_time_horizon = 0
    # Timp mediu până la defectare (MTTF)
    mean_time_to_failure = 200.0
    # Timp mediu de reparație (MTTR)
    mean_repair_time = 20.0

    # Utilizări + ETPC la training
    train_util_list = [0.70, 0.85, 0.95]
    train_etpc_list = [0.03, 0.05, 0.08]

    # Utilizări + ETPC la test
    test_util_list = [0.70, 0.95]
    test_etpc_list = [0.03, 0.08]

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
                    max_num_candidate_machines=candidate_mc_train[0],
                    min_proc_time=proc_range_train[0],
                    max_proc_time=proc_range_train[1],
                    allowance_factors=allowance_factors,
                    max_time_horizon=max_time_horizon,
                    mean_time_to_failure=mean_time_to_failure,
                    mean_repair_time=mean_repair_time,
                    seed=seed_val,
                    etpc_min_lapse=5,
                    etpc_max_lapse=15
                    )
                    fname = f"/Users/mihaiosan/PycharmProjects/DizertatieProblemaExtinsaDFJSS/dynamic_data/fan21/training_sets/train_fan21_{train_id}_util{util}_ec{ec}_nm{num_machines}_v{vers}.json"
                    with open(fname, "w") as f:
                        json.dump(ds, f, indent=2)

                    print(f"[TRAIN FAN21] Generated: {fname}")
                    train_id += 1
'''
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
                    max_num_candidate_machines=candidate_mc_test[0],
                    min_proc_time=proc_range_test[0],
                    max_proc_time=proc_range_test[1],
                    allowance_factors=allowance_factors,
                    max_time_horizon=max_time_horizon,
                    mean_time_to_failure=mean_time_to_failure,
                    mean_repair_time=mean_repair_time,
                    seed=seed_val,
                    etpc_min_lapse=5,
                    etpc_max_lapse=10
                )
                    fname_test = f"/Users/mihaiosan/PycharmProjects/DizertatieProblemaExtinsaDFJSS/dynamic_data/fan21/test_sets/test_fan21_{test_id}_util{util}_ec{ec}_nm{num_machines}_v{vers}.json"
                    with open(fname_test, "w") as f:
                        json.dump(ds_test, f, indent=2)

                    print(f"[TEST FAN21] Generated: {fname_test}")
                    test_id += 1
'''