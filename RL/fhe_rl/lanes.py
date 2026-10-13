"""
Lane packing: turns a scalar-lane expression into the program the code generator really builds.

  (Vec lane0 lane1 ... laneN)   where lanes are scalar expressions

Lanes with the same operator STRUCTURE (variable names and constants ignored) are packed together:
every leaf position of a structure becomes ONE ciphertext holding that variable of all those lanes,
and the structure's operations are done once. Different structures are separate groups, whose
results are added together at the end. Vec nodes that contain only variables/constants are left
untouched (they are already 1 ciphertext each).

pack_expr_str(expr_str, parse_sexpr) -> s-expression string that the unchanged estimator can use.
"""
from expr import Const, Op, Var


def _shape(n):
    if isinstance(n, Const):
        return "K"
    if isinstance(n, Var):
        return "V"
    return "(" + n.op + " " + " ".join(_shape(a) for a in n.args) + ")"


def _pack_vec(vec):
    groups = {}
    for lane in vec.args:
        groups.setdefault(_shape(lane), []).append(lane)
    ids = {}

    def build(nodes):
        first = nodes[0]
        if isinstance(first, Const):
            return str(first)
        if isinstance(first, Var):
            key = tuple(n.name for n in nodes)
            if key not in ids:
                ids[key] = len(ids)
            return f"c{ids[key]}"
        return "(" + first.op + " " + " ".join(
            build([n.args[i] for n in nodes]) for i in range(len(first.args))) + ")"

    roots = [build(g) for s, g in groups.items() if s != "K"]
    if not roots:
        return None
    out = roots[0]
    for r in roots[1:]:
        out = f"(+ {out} {r})"
    return out


def pack_expr_str(expr_str, parse_sexpr):
    tree = parse_sexpr(expr_str)

    def rec(n):
        if isinstance(n, (Const, Var)):
            return str(n)
        if n.op == "Vec" and any(isinstance(a, Op) for a in n.args):
            s = _pack_vec(n)
            return s if s is not None else str(n)
        return "(" + n.op + " " + " ".join(rec(a) for a in n.args) + ")"

    return rec(tree)