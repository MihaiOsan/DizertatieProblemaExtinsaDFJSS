import concurrent.futures
import operator
import random  # random este importat dar nu folosit direct aici, poate pentru viitor
from deap import base, creator, tools, gp


def protected_div(a: float, b: float) -> float:
    """
    Diviziune protejată pentru a evita erorile de împărțire la zero.
    Returnează `a` dacă `b` este foarte aproape de zero, altfel `a / b`.
    """
    # Daca b e 0, am putea returna o valoare mare daca a e pozitiv,
    # sau a insusi, sau 1.0. Alegerea depinde de cum vrem sa penalizam/interpretam.
    # Varianta initiala `else a` poate fi problematica daca `a` e mic si `b` e aproape de 0.
    # O valoare mare ar putea fi mai sigura pentru a evita prioritati neasteptat de mari.
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

    Acum definește un PrimitiveSet cu 12 argumente pentru a include ETPC.

    :param np: Numărul de worker-i pentru ThreadPoolExecutor (evaluare paralelă).
    :return: Obiectul DEAP Toolbox configurat.
    """
    print("Creating GP Toolbox with 12 arguments for dispatch rule (including ETPC)...")

    # PrimitiveSet "MAIN" acum accepta 12 argumente
    pset = gp.PrimitiveSet("MAIN", 12)

    # Argumentele existente:
    pset.renameArguments(ARG0='PT')  # Processing Time
    pset.renameArguments(ARG1='RO')  # Remaining Operations
    pset.renameArguments(ARG2='MW')  # Machine Wait time
    pset.renameArguments(ARG3='TQ')  # Time in Queue
    pset.renameArguments(ARG4='WIP')  # Work In Progress
    pset.renameArguments(ARG5='RPT')  # Remaining Processing Time (job-level)
    pset.renameArguments(ARG6='TUF')  # Time Until Finish (of machine's breakdown)
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
    pset.addTerminal(1.0, name="oneF")
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


if __name__ == '__main__':
    test_toolbox = create_toolbox(np=1)
    print("\nPrimitive Set Arguments:")
    for i in range(test_toolbox.pset.arity):  # Arity este numarul de argumente
        print(f"ARG{i}: {test_toolbox.pset.arguments[i]}")

    print("\nTerminals:")
    # Afisam terminalele de tip float si apoi celelalte tipuri daca exista
    if float in test_toolbox.pset.terminals:
        for term in test_toolbox.pset.terminals[float]:
            print(f"Float Terminal: {term.name if hasattr(term, 'name') else term.value}")
    for term_type in test_toolbox.pset.terminals:
        if term_type is not float:
            for term in test_toolbox.pset.terminals[term_type]:
                print(f"Type {term_type} Terminal: {term.name if hasattr(term, 'name') else term.value}")

    try:
        expr_example = gp.genFull(test_toolbox.pset, min_=2, max_=3)  # Adancime mai mare pentru mai multe argumente
        individual_example = creator.Individual(expr_example)
        print(f"\nExample Individual Tree: {str(individual_example)}")

        compiled_func = test_toolbox.compile(expr=individual_example)

        # Testeaza cu valori placeholder (12 argumente)
        # PT, RO, MW, TQ, WIP, RPT, TUF, DD, SLK, WJ, ETPC_D, N_ETPC_S
        result = compiled_func(10, 3, 5, 2, 20, 50, 0, 200, 100, 2, 5.0, 3)
        print(f"Compiled function result with 12 placeholder args: {result}")
    except Exception as e:
        print(f"Error during example individual test: {e}")
