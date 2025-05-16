import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict  # Necesar pentru breakdowns_for_plot in __main__ example
from typing import List, Dict, Tuple, Any, Optional  # Pentru type hints
from pathlib import Path  # Adaugat pentru exemplul __main__

# Presupunem ca TUPLE_FIELDS si field sunt definite undeva accesibil
# Daca nu, le putem include aici sau importa.
# Pentru acest exemplu, le voi include aici pentru a face functia auto-suficienta.

TUPLE_FIELDS = {
    "job": 0,
    "op": 1,
    "machine": 2,
    "start": 3,
    "end": 4,
}


def field(op_tuple: Tuple, name: str) -> Any:
    """Returnează câmpul `name` dintr‑un tuplu de operație din schedule."""
    if not isinstance(op_tuple, tuple) or len(op_tuple) < max(TUPLE_FIELDS.values()) + 1:
        # print(f"Warning: field() received unexpected op_tuple format: {op_tuple}")
        return None  # Sau ridica o eroare mai specifica
    try:
        return op_tuple[TUPLE_FIELDS[name]]
    except KeyError:
        # print(f"Warning: field '{name}' not found in TUPLE_FIELDS.")
        return None
    except IndexError:
        # print(f"Warning: IndexError accessing field '{name}' in op_tuple: {op_tuple}")
        return None


def plot_gantt(
        schedule: List[Tuple[int, int, int, float, float]],
        num_machines: int,
        breakdowns: Dict[int, List[Tuple[float, float]]],
        title: str = "Gantt Chart",
        save_path: Optional[str] = None
):
    """
    Generează și afișează/salvează un grafic Gantt pentru planificare.

    Args:
        schedule: Lista de tupluri (job_sim_id, op_idx, machine_id, start_time, end_time).
        num_machines: Numărul total de mașini.
        breakdowns: Un dicționar {machine_id: [(bd_start, bd_end), ...]} pentru perioadele de inactivitate.
        title: Titlul graficului.
        save_path: Calea unde să fie salvat graficul. Dacă None, graficul este afișat.
    """
    fig, ax = plt.subplots(figsize=(14, 7))  # Dimensiune ajustata pentru lizibilitate

    # Determinam makespan-ul real din schedule pentru a seta corect axa X
    actual_makespan_plot = 0.0
    if schedule:
        valid_ends = [field(op, "end") for op in schedule if field(op, "end") is not None]
        if valid_ends:
            actual_makespan_plot = max(valid_ends)

    if abs(actual_makespan_plot) < 1e-9 and breakdowns:
        max_bd_end = 0.0
        for bd_list in breakdowns.values():
            for _, bd_e in bd_list:
                max_bd_end = max(max_bd_end, bd_e)
        actual_makespan_plot = max_bd_end

    if abs(actual_makespan_plot) < 1e-9:
        actual_makespan_plot = 10.0

        # 1) Plot breakdown-urile
    if breakdowns:
        for m_idx in range(num_machines):
            for (bd_start, bd_end) in breakdowns.get(m_idx, []):
                if bd_start < actual_makespan_plot:
                    bd_duration = bd_end - bd_start
                    if bd_duration > 1e-9:
                        ax.barh(m_idx, bd_duration, left=bd_start, height=0.9,
                                color="lightcoral", alpha=0.5, edgecolor="maroon", hatch='///')

    # 2) Plot operațiile
    cmap = None
    try:
        # --- MODIFICARE AICI ---
        cmap = plt.colormaps.get_cmap("tab20")  # Noua API asteapta doar numele
        # --- SFARSIT MODIFICARE ---
    except AttributeError:
        try:
            cmap = plt.cm.get_cmap("tab20", 20)  # API veche, lut=20 e ok pentru tab20
        except ValueError:
            cmap = plt.cm.get_cmap("viridis", 20)  # Fallback general

    if cmap is None:  # Fallback final daca totul esueaza
        cmap = plt.cm.get_cmap("viridis", 20)

    job_colors: Dict[int, Any] = {}
    assigned_color_indices = set()

    for op_tuple_plot in schedule:
        job_id = field(op_tuple_plot, "job")
        op_idx = field(op_tuple_plot, "op")
        machine_id = field(op_tuple_plot, "machine")
        start_time = field(op_tuple_plot, "start")
        end_time = field(op_tuple_plot, "end")

        if None in [job_id, op_idx, machine_id, start_time, end_time]:
            continue

        job_id = int(job_id)
        op_idx = int(op_idx)
        machine_id = int(machine_id)
        start_time = float(start_time)
        end_time = float(end_time)

        if job_id not in job_colors:
            color_idx_candidate = job_id % cmap.N
            while color_idx_candidate in assigned_color_indices and len(assigned_color_indices) < cmap.N:
                color_idx_candidate = (color_idx_candidate + 1) % cmap.N
            job_colors[job_id] = cmap(color_idx_candidate)
            assigned_color_indices.add(color_idx_candidate)

        duration = end_time - start_time
        if duration < 1e-9: continue

        ax.barh(machine_id, duration, left=start_time, color=job_colors[job_id],
                edgecolor="black", height=0.7, alpha=0.9)
        ax.text(start_time + duration / 2, machine_id, f"J{job_id}.{op_idx}",
                ha="center", va="center", color="black", fontsize=7, fontweight='bold')

    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Machine", fontsize=10)
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"M{i}" for i in range(num_machines)], fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold')

    if any(breakdowns.values()):
        breakdown_patch = mpatches.Patch(facecolor="lightcoral", alpha=0.5, edgecolor="maroon", hatch='///',
                                         label='Breakdown')
        ax.legend(handles=[breakdown_patch], fontsize=8, loc='best')

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=150)
            print(f"   Gantt chart saved to: {save_path}")
        except Exception as e:
            print(f"   Error saving Gantt chart to {save_path}: {e}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
