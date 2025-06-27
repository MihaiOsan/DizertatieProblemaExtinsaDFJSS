# tests/test_tree_simplifier.py
"""Unit-tests pentru **treeSimplifier.py** (funcții: `simplify_individual`, `tree_str`, `infix_str`).
Comentarii scurte, clare.
"""

import pytest
from deap import gp

try:
    import simpleTree as ts
except ImportError:
    pytest.skip("Modulul simpleTree nu este importabil", allow_module_level=True)


def _pset():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.renameArguments(ARG0="x")

    pset.addPrimitive(lambda a, b: a + b, 2, name="add")
    pset.addPrimitive(lambda a, b: a - b, 2, name="sub")
    pset.addPrimitive(lambda a, b: a * b, 2, name="mul")
    pset.addPrimitive(lambda a, b: 1e9 if abs(b) < 1e-9 else a / b, 2, name="protected_div")
    pset.addPrimitive(lambda a: -a, 1, name="neg")
    pset.addPrimitive(max, 2, name="max")
    pset.addPrimitive(min, 2, name="min")

    pset.addTerminal(0.0)
    pset.addTerminal(1.0)
    return pset


def _compile(ind, pset):
    return gp.compile(expr=ind, pset=pset)

def test_add_zero():
    pset = _pset()
    ind = gp.PrimitiveTree.from_string("add(x, 0.0)", pset)
    simp = ts.simplify_individual(ind, pset)
    assert len(simp) < len(ind)
    assert _compile(simp, pset)(7) == 7


def test_mul_one():
    pset = _pset()
    ind = gp.PrimitiveTree.from_string("mul(1.0, x)", pset)
    simp = ts.simplify_individual(ind, pset)
    assert len(simp) < len(ind)
    assert _compile(simp, pset)(-3.5) == -3.5


def test_double_neg():
    pset = _pset()
    ind = gp.PrimitiveTree.from_string("neg(neg(x))", pset)
    simp = ts.simplify_individual(ind, pset)
    assert len(simp) == 1  # redus la un singur nod


def test_constant_fold():
    pset = _pset()
    ind = gp.PrimitiveTree.from_string("add(1.0, 1.0)", pset)
    simp = ts.simplify_individual(ind, pset)
    assert len(simp) == 1 and _compile(simp, pset)(0) == 2.0

# ------------------------------------------------------------------
# tree_str & infix_str ---------------------------------------------
# ------------------------------------------------------------------

def test_tree_str_contains_branches():
    pset = _pset()
    ind = gp.PrimitiveTree.from_string("sub(x, 1.0)", pset)
    txt = ts.tree_str(ind)
    assert "└─" in txt or "├─" in txt


def test_infix_minimal_parentheses():
    pset = _pset()
    ind = gp.PrimitiveTree.from_string("protected_div(add(x, 1.0), mul(1.0, x))", pset)
    inf = ts.infix_str(ind)
    assert "/" in inf and inf.count("(") <= 4  # paranteze minime
