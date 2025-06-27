import concurrent.futures
import operator
import random
from deap import base, creator, tools, gp


def protected_div(a: float, b: float) -> float:
    """
    Diviziune protejată pentru a evita erorile de împărțire la zero.
    Returnează `a` dacă `b` este foarte aproape de zero, altfel `a / b`.
    """
    if abs(b) < 1e-9:
        if a > 1e-9: return 1e9  # Numar mare pozitiv
        if a < -1e-9: return -1e9  # Numar mare negativ
        return 0.0  # Daca si a e 0
    return a / b

def generate_random_value_for_erc():
    return round(random.uniform(-5, 5), 2)


def create_toolbox(np: int = 3) -> base.Toolbox:
    """
    Creează și returnează un obiect `toolbox` DEAP cu
    definirea primitivelor GP, a tipurilor de date și
    operatorilor de încrucișare/mutare, selecție etc.

    definește un PrimitiveSet cu 12 argumente pentru a include ETPC.

    :param np: Numărul de worker-i pentru ThreadPoolExecutor (evaluare paralelă).
    :return: Obiectul DEAP Toolbox configurat.
    """
    print("Creating GP Toolbox with 12 arguments for dispatch rule (including ETPC)...")

    # PrimitiveSet "MAIN" acum accepta 12 argumente
    pset = gp.PrimitiveSet("MAIN", 12)

    # Argumentele existente:
    pset.renameArguments(ARG0='PT')  # Processing Time
    pset.renameArguments(ARG1='RO')  # Remaining Operations in the same job
    pset.renameArguments(ARG2='MW')  # Machine Wait time
    pset.renameArguments(ARG3='TQ')  # Time in Queue
    pset.renameArguments(ARG4='WIP')  # Work In Progress
    pset.renameArguments(ARG5='RPT')  # Remaining Processing Time (job-level)
    pset.renameArguments(ARG6='TUF')  # Time Until Fixed (of machine's breakdown)
    pset.renameArguments(ARG7='DD')  # Due Date (al jobului candidat)
    pset.renameArguments(ARG8='SLK')  # Slack Time (al jobului candidat)
    pset.renameArguments(ARG9='WJ')  # Weight of Job

    # Noile argumente pentru ETPC:
    pset.renameArguments(ARG10='ETPC_D')  # ETPC Delay for the current operation
    pset.renameArguments(ARG11='N_ETPC_S')  # Number of ETPC Successors for the current operation

    # Primitive standard
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="protected_div")
    pset.addPrimitive(operator.neg, 1, name="neg")
    pset.addPrimitive(min, 2, name="min")
    pset.addPrimitive(max, 2, name="max")

    # Terminale constante
    pset.addTerminal(1)
    pset.addEphemeralConstant("ERC", generate_random_value_for_erc)

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    if np > 0:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=np if np > 0 else None)
        toolbox.register("map", executor.map)

    toolbox.pset = pset

    print("Toolbox created successfully with ETPC arguments.")
    return toolbox

