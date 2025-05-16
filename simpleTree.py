"""Utility helpers for DEAP Genetic Programming individuals.

Features
========
* **simplify_individual** – încearcă:
  1. `deap.gp.simplify` (SymPy‑powered) – dacă funcționează.
  2. Peephole simplifier cu reguli algebrice + constant‑fold.
  În caz de eroare, întoarce individul original.

* **tree_str** – ascii‑tree pentru un `PrimitiveTree`.
"""
from __future__ import annotations

import operator as _op
from numbers import Number
from typing import Any, Sequence, Union, Dict, Tuple

from deap import gp

try:
    import sympy  # noqa: F401 – numai pentru gp.simplify

    _HAS_SYMPY = True
except ImportError:  # pragma: no cover
    _HAS_SYMPY = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_const(node) -> bool:
    return isinstance(node, Number)


def _const_val(node) -> Union[int, float]:
    return float(node)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Simplifier (recursive)
# ---------------------------------------------------------------------------

def _simplify_rec(name: str, args: Sequence[Any]):
    children = list(args)

    # algebraic rules ------------------------------------------------------
    if name == "add":
        children = [c for c in children if not (_is_const(c) and _const_val(c) == 0)]
        if not children:
            return 0.0
        if len(children) == 1:
            return children[0]

    elif name == "mul":
        if any(_is_const(c) and _const_val(c) == 0 for c in children):
            return 0.0
        children = [c for c in children if not (_is_const(c) and _const_val(c) == 1)]
        if not children:
            return 1.0
        if len(children) == 1:
            return children[0]

    elif name == "sub" and len(children) == 2 and children[0] == children[1]:
        return 0.0

    elif name in ("max", "min") and len(children) == 2 and children[0] == children[1]:
        return children[0]

    elif name == "neg":
        child = children[0]
        if isinstance(child, tuple) and child[0] == "neg":
            return child[1]
        if _is_const(child):
            return -_const_val(child)

    # constant folding -----------------------------------------------------
    if all(_is_const(c) for c in children):
        consts = [_const_val(c) for c in children]
        try:
            match name:
                case "add":
                    return sum(consts)
                case "sub":
                    return consts[0] - consts[1]
                case "mul":
                    return _op.mul(consts[0], consts[1])
                case "max":
                    return max(consts)
                case "min":
                    return min(consts)
                case "neg":
                    return -consts[0]
                case "protected_div":
                    return 1.0 if consts[1] == 0 else consts[0] / consts[1]
        except Exception:  # pragma: no cover
            pass

    if len(children) == 1:
        return children[0]

    return (name, *children)


# ---------------------------------------------------------------------------
# Tree <-> nested tuple conversion
# ---------------------------------------------------------------------------

def _to_nested(expr: gp.PrimitiveTree, idx: int = 0):
    """Convert DEAP tree to nested tuples usable by peephole simplifier.

    Returns: (nested_repr, next_index)
    """
    node = expr[idx]
    arity = getattr(node, "arity", 0)

    if arity == 0:  # terminal
        if hasattr(node, "value"):
            return node.value, idx + 1  # constant
        return node.name, idx + 1       # variable

    children = []
    next_idx = idx + 1
    for _ in range(arity):
        sub, next_idx = _to_nested(expr, next_idx)
        children.append(sub)

    return (node.name, *children), next_idx


def _from_nested(nested, pset):
    if not isinstance(nested, tuple):
        return str(nested)
    name, *children = nested
    child_strs = ( _from_nested(c, pset) for c in children )
    return f"{name}({', '.join(child_strs)})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simplify_individual(ind: gp.PrimitiveTree, pset):
    """Return a *simplified* PrimitiveTree, or the original one on failure."""

    # 1) SymPy path --------------------------------------------------------
    if _HAS_SYMPY:
        try:
            simp = gp.simplify(ind, pset)
            return simp if isinstance(simp, gp.PrimitiveTree) else gp.PrimitiveTree.from_string(str(simp), pset)
        except Exception:
            pass  # fall through

    # 2) Peephole path -----------------------------------------------------
    nested, _ = _to_nested(ind)

    def walk(expr):
        if not isinstance(expr, tuple):
            return expr
        name, *children = expr
        return _simplify_rec(name, [walk(c) for c in children])

    simplified_nested = walk(nested)
    raw_expr = _from_nested(simplified_nested, pset)

    try:
        return gp.PrimitiveTree.from_string(raw_expr, pset)
    except Exception:  # Any parser error → give back original
        return ind


# ---------------------------------------------------------------------------
# ASCII tree printer
# ---------------------------------------------------------------------------
def tree_str(expr: gp.PrimitiveTree) -> str:
    """Return a *Unicode* tree diagram using ├─/└─ branches."""

    def _label(node):
        if getattr(node, "arity", 0) == 0:
            return str(node.value) if hasattr(node, "value") else node.name
        return node.name

    lines: list[str] = []

    def rec(idx: int, prefix: str, is_last: bool):
        node = expr[idx]
        branch = "└─ " if is_last else "├─ "
        lines.append(f"{prefix}{branch}{_label(node)}")

        child_prefix = prefix + ("   " if is_last else "│  ")
        child_idx = idx + 1
        arity = getattr(node, "arity", 0)
        for i in range(arity):
            rec(child_idx, child_prefix, i == arity - 1)
            child_idx = expr.searchSubtree(child_idx).stop

    rec(0, "", True)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Infix pretty‑printer (mathematical formula)
# ---------------------------------------------------------------------------

# precedence mapping: lower number = binds weaker
_PRECEDENCE: Dict[str, int] = {
    "add": 1,
    "sub": 1,
    "mul": 2,
    "protected_div": 2,
    "neg": 3,
    "max": 0,
    "min": 0,
}

_SYMBOLS: Dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "protected_div": "/",
    "neg": "-",
}

def infix_str(expr: gp.PrimitiveTree) -> str:
    """Return an infix formula with **minimal but sufficient** parentheses."""

    def rec(idx: int) -> Tuple[str, int]:
        node = expr[idx]
        name = node.name
        arity = getattr(node, "arity", 0)

        # terminals -------------------------------------------------------
        if arity == 0:
            if hasattr(node, "value"):
                return str(node.value), 4  # high prec.
            return node.name, 4

        # unary neg -------------------------------------------------------
        if name == "neg":
            sub_str, sub_prec = rec(idx + 1)
            if sub_prec < _PRECEDENCE["neg"]:
                sub_str = f"({sub_str})"
            return f"-{sub_str}", _PRECEDENCE["neg"]

        # n‑ary / binary --------------------------------------------------
        child_idx = idx + 1
        parts: list[str] = []
        precs: list[int] = []
        for _ in range(arity):
            s, p = rec(child_idx)
            parts.append(s)
            precs.append(p)
            child_idx = expr.searchSubtree(child_idx).stop

        if name in ("add", "mul"):
            op = _SYMBOLS[name]
            cur_prec = _PRECEDENCE[name]
            wrapped = [f"({s})" if p < cur_prec else s for s, p in zip(parts, precs)]
            return f" {op} ".join(wrapped), cur_prec

        if name in ("sub", "protected_div"):
            op = _SYMBOLS[name]
            cur_prec = _PRECEDENCE[name]
            left = parts[0]
            if precs[0] < cur_prec:
                left = f"({left})"
            right = parts[1]
            # for non‑associative ops we need () even when equal precedence on RHS
            if precs[1] <= cur_prec:
                right = f"({right})"
            return f"{left} {op} {right}", cur_prec

        # functions max/min ---------------------------------------------
        return f"{name}({', '.join(parts)})", _PRECEDENCE[name]

    txt, _ = rec(0)
    return txt

