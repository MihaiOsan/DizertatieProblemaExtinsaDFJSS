# data_reader.py

import os
import json
import math
from typing import List, Tuple, Dict, Any, Optional


# --- Definirea Claselor Entitate ---

class Operation:
    def __init__(self, op_idx_in_job: int, job_sim_id: int, job_original_id: Any = None):
        self.op_idx_in_job: int = op_idx_in_job
        self.job_sim_id: int = job_sim_id
        self.job_original_id: Any = job_original_id
        self.alternatives: List[Tuple[int, float]] = []  # (machine_id, processing_time)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.assigned_machine: Optional[int] = None

    def add_alternative(self, machine_id: int, processing_time: float):
        self.alternatives.append((machine_id, float(processing_time)))
        self.alternatives.sort(key=lambda x: x[0])

    def get_processing_time(self, machine_id: int) -> Optional[float]:
        for m_id, p_time in self.alternatives:
            if m_id == machine_id:
                return p_time
        return None

    def get_best_processing_time(self) -> float:
        if not self.alternatives: return float('inf')
        return min(p_time for _, p_time in self.alternatives)

    def __repr__(self):
        return f"Op(J_sim:{self.job_sim_id}, OpIdx:{self.op_idx_in_job}, #Alts:{len(self.alternatives)})"


class Job:
    def __init__(self, sim_id: int, original_json_index: Optional[int] = None,
                 original_id_from_json: Any = None,
                 arrival_time: float = 0.0, weight: float = 1.0,
                 due_date: float = float('inf')):
        self.sim_id: int = sim_id
        self.original_json_index: Optional[int] = original_json_index  # Indexul din lista 'jobs' a JSON-ului
        self.original_id_from_json: Any = original_id_from_json

        self.arrival_time: float = float(arrival_time)
        self.weight: float = float(weight)
        self.due_date: float = float(due_date)
        self.operations: List[Operation] = []

        self.is_initial_in_sim: bool = False
        self.completion_time_sim: Optional[float] = None
        self.is_cancelled_sim: bool = False
        self.current_op_idx_sim: int = 0  # Progresul in simulare

    def add_operation(self, op_idx_in_job: int) -> Operation:
        # job_sim_id si job_original_id vor fi setate de Operation folosind referinta la acest Job
        operation = Operation(op_idx_in_job, self.sim_id, self.original_id_from_json)
        self.operations.append(operation)
        return operation

    @property
    def num_operations(self) -> int:
        return len(self.operations)

    def get_operation(self, op_idx_in_job: int) -> Optional[Operation]:
        if 0 <= op_idx_in_job < len(self.operations):
            return self.operations[op_idx_in_job]
        return None

    def __repr__(self):
        return (f"Job(SimID:{self.sim_id}, OrigID:'{self.original_id_from_json}', "
                f"JsonIdx:{self.original_json_index}, Arr:{self.arrival_time:.2f}, Ops:{self.num_operations})")


class ETPCConstraint:
    def __init__(self, fore_job_orig_id_ref: Any, fore_op_idx: int,
                 hind_job_orig_id_ref: Any, hind_op_idx: int, time_lapse: float,
                 constraint_orig_idx: Optional[int] = None):
        self.fore_job_orig_id_ref: Any = fore_job_orig_id_ref
        self.fore_op_idx: int = fore_op_idx
        self.hind_job_orig_id_ref: Any = hind_job_orig_id_ref
        self.hind_op_idx: int = hind_op_idx
        self.time_lapse: float = float(time_lapse)
        self.constraint_orig_idx: Optional[int] = constraint_orig_idx

    def __repr__(self):
        return (f"ETPC(OrigIdx:{self.constraint_orig_idx} => {self.fore_job_orig_id_ref}.{self.fore_op_idx} -> "
                f"{self.hind_job_orig_id_ref}.{self.hind_op_idx} +{self.time_lapse:.2f})")


class BaseEvent:  # Clasa de baza pentru evenimente
    def __init__(self, event_time: float, event_type: str):
        self.event_time: float = float(event_time)
        self.type: str = event_type

    def __lt__(self, other):  # Pentru sortare
        if self.event_time != other.event_time:
            return self.event_time < other.event_time
        return self.type < other.type  # Alfabetic

    def __repr__(self):
        return f"Event(Time:{self.event_time:.2f}, Type:'{self.type}')"


class BreakdownEvent(BaseEvent):
    def __init__(self, machine_id: int, start_time: float, end_time: float):
        super().__init__(start_time, "machine_breakdown")
        self.machine_id: int = machine_id
        self.end_time: float = float(end_time)  # Timpul cand masina devine disponibila

    def __repr__(self):
        return f"Breakdown(M{self.machine_id}, Start:{self.event_time:.2f}, End:{self.end_time:.2f})"


class AddJobDynamicEvent(BaseEvent):  # Eveniment pentru joburi adaugate explicit in sectiunea dynamic_events
    def __init__(self, arrival_time: float, job_object: Job):
        super().__init__(arrival_time, "add_job_dynamic")
        self.job_object: Job = job_object

    def __repr__(self):
        return f"AddJobDynamic(Time:{self.event_time:.2f}, JobSimID:{self.job_object.sim_id})"


class CancelJobEvent(BaseEvent):
    def __init__(self, cancel_time: float, job_to_cancel_orig_id_ref: Any,
                 job_to_cancel_sim_id_mapped: Optional[int] = None):  # sim_id va fi mapat de parser
        super().__init__(cancel_time, "cancel_job")
        self.job_to_cancel_orig_id_ref: Any = job_to_cancel_orig_id_ref
        self.job_to_cancel_sim_id_mapped: Optional[int] = job_to_cancel_sim_id_mapped

    def __repr__(self):
        return (f"CancelJob(Time:{self.event_time:.2f}, "
                f"OrigID:'{self.job_to_cancel_orig_id_ref}', SimID_Mapped:{self.job_to_cancel_sim_id_mapped})")


class FJSPInstance:
    def __init__(self, file_name: str):
        self.file_name: str = file_name
        self.num_machines: int = 0

        # Toate joburile definite in fisier (inainte de a fi clasificate ca initiale sau dinamice)
        self.jobs_defined_in_file: List[Job] = []

        # Dupa apelul `finalize_setup`, acestea vor fi populate/disponibile:
        self.initial_job_sim_ids: List[int] = []
        self.dynamic_event_timeline: List[BaseEvent] = []

        self.etpc_constraints: List[ETPCConstraint] = []

        self.breakdown_events: List[BreakdownEvent] = []
        self.add_job_events: List[AddJobDynamicEvent] = []
        self.cancel_job_events: List[CancelJobEvent] = []

        # Harta pentru a gasi un Job obiect pe baza sim_id-ului sau
        self._sim_id_to_job_object_map: Dict[int, Job] = {}
        self._original_id_to_sim_id_map: Dict[Any, int] = {}
        self._next_sim_id_counter: int = 0

    def _assign_sim_id(self, job_object: Job) -> int:
        assigned_sim_id = self._next_sim_id_counter
        job_object.sim_id = assigned_sim_id  # Atribuim sim_id obiectului Job
        self._next_sim_id_counter += 1
        self._sim_id_to_job_object_map[assigned_sim_id] = job_object
        if job_object.original_id_from_json is not None:
            if job_object.original_id_from_json in self._original_id_to_sim_id_map:
                print(f"   Warning: Duplicate original_id_from_json '{job_object.original_id_from_json}' encountered "
                      f"while assigning sim_id {assigned_sim_id}. Previous was {self._original_id_to_sim_id_map[job_object.original_id_from_json]}.")
            self._original_id_to_sim_id_map[job_object.original_id_from_json] = assigned_sim_id
        return assigned_sim_id

    def add_job_from_definition(self, job_def: Dict, original_json_idx: int) -> Optional[Job]:
        """Creeaza si adauga un obiect Job din definitia JSON. Returneaza obiectul Job sau None."""
        original_id = job_def.get('id')
        job_id_info_for_log = original_id if original_id is not None else f"at json_idx {original_json_idx}"

        # Sim_id-ul va fi atribuit de _assign_sim_id
        new_job = Job(sim_id=-1,  # Placeholder, va fi setat de _assign_sim_id
                      original_json_index=original_json_idx,
                      original_id_from_json=original_id)

        self._assign_sim_id(new_job)  # Atribuie si inregistreaza sim_id

        try:
            arrival_time_raw = job_def.get("arrival_time", 0.0)
            new_job.arrival_time = math.ceil(float(arrival_time_raw))
            if new_job.arrival_time < 0: new_job.arrival_time = 0.0

            weight_raw = job_def.get('weight', 1.0)
            new_job.weight = float(weight_raw)

            due_date_raw = job_def.get('due_date', float('inf'))
            new_job.due_date = float(due_date_raw)

        except (ValueError, TypeError) as e:
            print(f"   Warning: Invalid metadata for job '{job_id_info_for_log}': {e}. Using defaults.")

        operations_json = job_def.get("operations", [])
        if not isinstance(operations_json, list):
            print(
                f"   Warning: 'operations' for job '{job_id_info_for_log}' is not a list. Job will have no operations.")
            operations_json = []

        for op_idx, op_def in enumerate(operations_json):
            operation_obj = new_job.add_operation(op_idx)  # Creeaza si adauga Op la Job
            if isinstance(op_def, dict) and isinstance(op_def.get("candidate_machines"), dict):
                for m_str, p_str in op_def["candidate_machines"].items():
                    try:
                        m_id = int(m_str)
                        p_time = float(p_str)
                        if p_time < 0: raise ValueError("Negative processing time")
                        operation_obj.add_alternative(m_id, p_time)
                    except (ValueError, TypeError) as e_op:
                        print(
                            f"   Warning: Invalid alternative {m_str}:{p_str} for J_sim:{new_job.sim_id}.Op:{op_idx}: {e_op}")
            else:
                print(
                    f"   Warning: Invalid operation definition for J_sim:{new_job.sim_id}.Op:{op_idx}. Skipping alternatives.")

        self.jobs_defined_in_file.append(new_job)
        return new_job

    def finalize_setup(self):
        """Sorteaza evenimentele, determina joburile initiale, si pregateste pentru simulare."""
        for job_obj in self.jobs_defined_in_file:
            for op_obj in job_obj.operations:
                op_obj.alternatives = [
                    (m, p) for m, p in op_obj.alternatives if 0 <= m < self.num_machines
                ]
                if not op_obj.alternatives and job_obj.num_operations > 0:  # Daca o operatie a ramas fara alternative valide
                    print(
                        f"   Warning: J_sim:{job_obj.sim_id}.Op:{op_obj.op_idx_in_job} has no valid alternatives after machine validation.")

        self.initial_job_sim_ids = []
        temp_add_job_events_from_defs = []

        # Sortam `jobs_defined_in_file` dupa arrival_time si apoi sim_id (care e secvential)
        # pentru a asigura o procesare consistenta daca parserul nu a garantat ordinea de sosire
        self.jobs_defined_in_file.sort(key=lambda j: (j.arrival_time, j.sim_id))

        for job_obj in self.jobs_defined_in_file:
            if job_obj.arrival_time <= 1e-9:  # Consideram initial
                job_obj.is_initial_in_sim = True
                self.initial_job_sim_ids.append(job_obj.sim_id)
            else:
                job_obj.is_initial_in_sim = False
                temp_add_job_events_from_defs.append(AddJobDynamicEvent(job_obj.arrival_time, job_obj))

        # Adaugam evenimentele de adaugare de job (cele din `jobs` lista JSON cu arrival > 0
        # si cele definite explicit in `dynamic_events` -> `added_jobs` din JSON)
        self.dynamic_event_timeline.extend(self.breakdown_events)
        self.dynamic_event_timeline.extend(
            self.add_job_events)
        self.dynamic_event_timeline.extend(
            temp_add_job_events_from_defs)  # Acestea sunt cele din JSON "jobs" cu arrival_time > 0
        self.dynamic_event_timeline.extend(self.cancel_job_events)

        self.dynamic_event_timeline.sort()  # Sorteaza toate evenimentele dupa timp

    def get_job_by_sim_id(self, sim_id: int) -> Optional[Job]:
        return self._sim_id_to_job_object_map.get(sim_id)

    def get_job_by_original_id(self, original_id: Any) -> Optional[Job]:
        if original_id is None: return None
        sim_id = self._original_id_to_sim_id_map.get(original_id)
        if sim_id is not None:
            return self.get_job_by_sim_id(sim_id)
        return None

    @property
    def num_total_defined_jobs(self) -> int:
        return len(self.jobs_defined_in_file)

    @property
    def num_initial_jobs_in_sim(self) -> int:
        return len(self.initial_job_sim_ids)

    def __repr__(self):
        return (f"FJSPInstance(File:'{self.file_name}', Machines:{self.num_machines}, "
                f"TotalJobsDefined:{self.num_total_defined_jobs}, InitialJobsInSim:{self.num_initial_jobs_in_sim}, "
                f"TotalEventsInTimeline:{len(self.dynamic_event_timeline)})")


# --- Funcții de Citire Refactorizate ---

def read_dynamic_fjsp_instance_json(file_path: str) -> Optional[FJSPInstance]:
    instance = FJSPInstance(os.path.basename(file_path))
    print(f"   Reading OOP FJSP instance (.json) from: {file_path}")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"   Error reading or parsing JSON file {file_path}: {e}")
        return None

    try:
        # 1. Procesam toate joburile definite in lista "jobs"
        all_job_definitions_json = data.get('jobs', [])
        if not isinstance(all_job_definitions_json, list):
            print("   Warning: 'jobs' field is not a list. No jobs will be loaded from main list.")
            all_job_definitions_json = []

        for idx, job_def in enumerate(all_job_definitions_json):
            if isinstance(job_def, dict):
                instance.add_job_from_definition(job_def, idx)
            else:
                print(f"   Warning: Invalid job definition at index {idx} in 'jobs' list. Skipping.")

        # 2. Determinam si setam num_machines (dupa ce toate joburile si breakdown-urile sunt schitate)
        max_machine_index = -1
        for job_obj in instance.jobs_defined_in_file:  # Joburile din lista principala "jobs"
            for op_obj in job_obj.operations:
                for m_id, _ in op_obj.alternatives:
                    max_machine_index = max(max_machine_index, m_id)

        machine_breakdowns_json = data.get('machine_breakdowns', {})
        if isinstance(machine_breakdowns_json, dict):
            for m_str in machine_breakdowns_json.keys():
                try:
                    max_machine_index = max(max_machine_index, int(m_str))
                except:
                    pass  # Ignoram chei ne-numerice

        instance.num_machines = max_machine_index + 1 if max_machine_index > -1 else 0
        if instance.num_machines == 0 and (instance.jobs_defined_in_file or machine_breakdowns_json):
            print(
                f"   Warning: num_machines is 0 but jobs/breakdowns exist. Max machine index found was {max_machine_index}.")

        # 3. Procesam breakdowns
        if isinstance(machine_breakdowns_json, dict):
            for m_str, bd_list_json in machine_breakdowns_json.items():
                try:
                    m_id_bd = int(m_str)
                    if not (0 <= m_id_bd < instance.num_machines):
                        print(
                            f"   Warning: Machine ID {m_id_bd} in breakdowns out of range (0-{instance.num_machines - 1}). Skipping.")
                        continue
                    if isinstance(bd_list_json, list):
                        for bd_item_json in bd_list_json:
                            if isinstance(bd_item_json, dict):
                                start_raw = bd_item_json.get("start_time")
                                duration_raw = bd_item_json.get("duration", bd_item_json.get("repair_time"))
                                try:
                                    start = math.ceil(float(start_raw))
                                    duration = math.ceil(float(duration_raw))
                                    if start < 0 or duration < 0: raise ValueError("negative times")
                                    instance.breakdown_events.append(BreakdownEvent(m_id_bd, start, start + duration))
                                except:
                                    print(
                                        f"   Warning: Invalid breakdown item {bd_item_json} for M{m_id_bd}. Skipping.")
                except:
                    print(f"   Warning: Invalid machine key '{m_str}' in breakdowns. Skipping.")

        # 4. Procesam ETPC
        etpc_json = data.get('etpc_constraints', [])
        if isinstance(etpc_json, list):
            for idx, etpc_item in enumerate(etpc_json):
                if isinstance(etpc_item, dict):
                    try:
                        fj_id = etpc_item['fore_job']
                        fo = int(etpc_item['fore_op_idx'])
                        hj_id = etpc_item['hind_job']
                        ho = int(etpc_item['hind_op_idx'])
                        tl = float(etpc_item['time_lapse'])
                        instance.etpc_constraints.append(ETPCConstraint(fj_id, fo, hj_id, ho, tl, idx))
                    except (KeyError, ValueError, TypeError) as e_etpc:
                        print(f"   Warning: Invalid ETPC item {etpc_item}: {e_etpc}. Skipping.")

        # 5. Procesam evenimentele din sectiunea "dynamic_events" a JSON-ului
        json_dynamic_events_section = data.get('dynamic_events', {})
        if isinstance(json_dynamic_events_section, dict):
            # Cancelled Jobs
            for cj_idx, cj_def in enumerate(json_dynamic_events_section.get('cancelled_jobs', [])):
                if isinstance(cj_def, dict):
                    try:
                        cancel_t = math.ceil(float(cj_def['time']))
                        orig_id_ref = cj_def.get('job_id', cj_def.get('job_id_to_cancel'))
                        if orig_id_ref is None: raise KeyError("Missing job reference for cancel")

                        instance.cancel_job_events.append(CancelJobEvent(cancel_t, orig_id_ref))
                    except (KeyError, ValueError, TypeError) as e_cj:
                        print(f"   Warning: Invalid CancelJob event {cj_def}: {e_cj}. Skipping.")

            # Added Jobs (cele definite explicit in dynamic_events.added_jobs)
            for aj_idx, aj_def in enumerate(json_dynamic_events_section.get('added_jobs', [])):
                if isinstance(aj_def, dict):

                    placeholder_orig_idx = -(aj_idx + 1)  # Index negativ pentru a indica sursa
                    added_job_obj = instance.add_job_from_definition(aj_def, placeholder_orig_idx)
                    if added_job_obj:
                        pass  # add_job_from_definition le adauga la jobs_defined_in_file

        # 6. Finalizam setup-ul instantei
        instance.finalize_setup()

        return instance

    except Exception as e_global:
        print(f"   CRITICAL Global Error processing JSON instance {file_path}: {e_global}")
        import traceback
        traceback.print_exc()
        return None


def read_dynamic_fjsp_instance_txt(file_path: str) -> Optional[FJSPInstance]:
    instance = FJSPInstance(os.path.basename(file_path))
    print(f"   Reading OOP FJSP instance (.txt) from: {file_path}")

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"   Error reading TXT file {file_path}: {e}")
        return None

    if not lines: print(f"   Error: File {file_path} is empty."); return None

    try:
        header_parts = lines[0].split()
        if len(header_parts) < 2: raise ValueError("TXT Header invalid.")
        num_jobs_header, num_machines_header = map(int, header_parts[:2])
        instance.num_machines = num_machines_header
        if num_jobs_header < 0 or instance.num_machines <= 0:
            raise ValueError("Invalid num_jobs/num_machines in TXT header.")

        # Simulam un job_def_original_idx pentru joburile din TXT
        current_txt_job_idx = 0

        parsing_section = "jobs"  # jobs, dynamic_events, breakdowns, added_jobs, cancelled_jobs

        for line_idx, line_str in enumerate(lines[1:], start=1):
            line_str = line_str.strip()
            if not line_str or line_str.startswith("#"): continue

            if line_str.startswith("Dynamic Events"): parsing_section = "dynamic_events"; continue
            if "Machine Breakdowns" in line_str: parsing_section = "breakdowns"; continue
            if "Added Jobs" in line_str: parsing_section = "added_jobs"; continue
            if "Cancelled Jobs" in line_str: parsing_section = "cancelled_jobs"; continue

            if parsing_section == "jobs":
                if current_txt_job_idx >= num_jobs_header:
                    print(
                        f"   Warning (TXT): More job lines than specified in header. Line {line_idx + 1}: '{line_str}'")
                    # Consideram ca trecem la evenimente sau e o eroare
                    parsing_section = "dynamic_events"
                    continue

                parts = list(map(int, line_str.split()))
                num_ops_txt, part_idx = parts[0], 1

                # Cream un job nou. ID-ul original nu e explicit in TXT, folosim indexul.
                # Arrival time e 0 pentru aceste joburi. Weight/Due Date sunt default.
                job_obj = Job(sim_id=-1,
                              original_json_index=current_txt_job_idx,
                              original_id_from_json=f"TXT_J{current_txt_job_idx}",
                              arrival_time=0.0)
                instance._assign_sim_id(job_obj)

                for op_create_idx in range(num_ops_txt):
                    operation_obj = job_obj.add_operation(op_create_idx)
                    num_alts_txt = parts[part_idx];
                    part_idx += 1
                    for _ in range(num_alts_txt):
                        m_txt, p_txt = parts[part_idx], parts[part_idx + 1];
                        part_idx += 2
                        if not (0 <= m_txt < instance.num_machines):
                            print(
                                f"   Warning (TXT): Invalid machine {m_txt} for J{current_txt_job_idx}.Op{op_create_idx}. Skipping alt.")
                            continue
                        if p_txt < 0:
                            print(
                                f"   Warning (TXT): Negative proc time {p_txt} for J{current_txt_job_idx}.Op{op_create_idx}. Skipping alt.")
                            continue
                        operation_obj.add_alternative(m_txt, float(p_txt))
                instance.jobs_defined_in_file.append(job_obj)
                current_txt_job_idx += 1

            elif parsing_section == "breakdowns":
                m_bd, s_bd, e_bd = map(int, line_str.split())
                if not (0 <= m_bd < instance.num_machines):
                    print(f"   Warning (TXT): Invalid machine {m_bd} for breakdown. Skipping.")
                    continue
                if s_bd < 0 or e_bd < s_bd:
                    print(f"   Warning (TXT): Invalid breakdown interval [{s_bd},{e_bd}]. Skipping.")
                    continue
                instance.breakdown_events.append(BreakdownEvent(m_bd, float(s_bd), float(e_bd)))

            elif parsing_section == "added_jobs":  # Joburi adaugate definite in sectiunea dynamic events a TXT
                time_part_str, job_part_str = line_str.split(":", 1)
                arrival_time_aj = float(time_part_str.strip())
                if arrival_time_aj <= 0:  # Joburile adaugate trebuie sa aiba timp de sosire > 0
                    print(f"   Warning (TXT): Added job with non-positive arrival time {arrival_time_aj}. Skipping.")
                    continue

                parts_aj = list(map(int, job_part_str.split()))
                num_ops_aj_txt, part_idx_aj = parts_aj[0], 1

                # Cream un job nou pentru acest eveniment. ID original sintetic.
                added_job_orig_id = f"TXT_AJ_t{int(arrival_time_aj)}_{len(instance.jobs_defined_in_file)}"
                # Folosim un index mare/negativ pentru a-l deosebi de cele din header daca e nevoie
                added_job_json_idx = - (len(instance.jobs_defined_in_file) + 1000)

                job_obj_aj = Job(sim_id=-1,  # Va fi setat de _assign_sim_id
                                 original_json_index=added_job_json_idx,
                                 original_id_from_json=added_job_orig_id,
                                 arrival_time=arrival_time_aj)
                instance._assign_sim_id(job_obj_aj)

                for op_create_idx_aj in range(num_ops_aj_txt):
                    operation_obj_aj = job_obj_aj.add_operation(op_create_idx_aj)
                    num_alts_aj_txt = parts_aj[part_idx_aj];
                    part_idx_aj += 1
                    for _ in range(num_alts_aj_txt):
                        m_aj_txt, p_aj_txt = parts_aj[part_idx_aj], parts_aj[part_idx_aj + 1];
                        part_idx_aj += 2
                        if not (0 <= m_aj_txt < instance.num_machines): continue  # Skip invalid
                        if p_aj_txt < 0: continue  # Skip invalid
                        operation_obj_aj.add_alternative(m_aj_txt, float(p_aj_txt))
                instance.jobs_defined_in_file.append(job_obj_aj)  # Va fi sortat si procesat in finalize_setup

            elif parsing_section == "cancelled_jobs":
                cancel_t_cj, job_orig_idx_to_cancel_cj = map(int, line_str.split())
                if not (0 <= job_orig_idx_to_cancel_cj < num_jobs_header):
                    print(
                        f"   Warning (TXT): Invalid job index {job_orig_idx_to_cancel_cj} for cancellation. Max is {num_jobs_header - 1}. Skipping.")
                    continue
                instance.cancel_job_events.append(
                    CancelJobEvent(float(cancel_t_cj), f"TXT_J{job_orig_idx_to_cancel_cj}"))

        instance.finalize_setup()
        return instance

    except Exception as e_global_txt:
        print(f"   CRITICAL Global Error processing TXT instance {file_path}: {e_global_txt}")
        import traceback
        traceback.print_exc()
        return None


def load_instances_from_directory(input_dir: str) -> List[FJSPInstance]:
    """
    Parcurge `input_dir` și încarcă toate instanțele FJSP .txt și .json.
    Returnează o listă de obiecte FJSPInstance.
    Omite fișierele care cauzează erori de parsare.
    """
    print(f"Reading OOP FJSP instances from directory: {input_dir}")
    all_fjsp_instances: List[FJSPInstance] = []
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found or is not a directory.")
        return []

    for root, _, files in os.walk(input_dir):
        files.sort()  # Procesare consistenta
        for fname in files:
            fpath = os.path.join(root, fname)
            instance_obj: Optional[FJSPInstance] = None

            print(f"\n--- Processing file (OOP): {fname} ---")
            if fname.endswith(".txt"):
                instance_obj = read_dynamic_fjsp_instance_txt(fpath)
            elif fname.endswith(".json"):
                instance_obj = read_dynamic_fjsp_instance_json(fpath)

            if instance_obj:
                # Validare simpla a obiectului instanta
                if instance_obj.num_machines >= 0:
                    all_fjsp_instances.append(instance_obj)
                    print(f"   Successfully loaded OOP instance: {instance_obj.file_name} - "
                          f"Initial Jobs in Sim: {instance_obj.num_initial_jobs_in_sim}, "
                          f"Total Defined: {instance_obj.num_total_defined_jobs}, "
                          f"Machines: {instance_obj.num_machines}")
                else:
                    print(f"   Skipping file {fname} due to invalid instance object state (e.g., num_machines < 0).")
            else:
                print(f"   Skipping file {fname} due to parsing errors (parser returned None).")
            print(f"--- Finished processing logic for file (OOP): {fname} ---")

    print(f"\nDone reading OOP FJSP instances. Stored {len(all_fjsp_instances)} valid instances.")
    return all_fjsp_instances