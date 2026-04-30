"""
Implement the old MILPs used to find the 2nd inertia-type bound for the distance-k
chromatic number 


@author: Valentin Michaux
"""

import csv
import time
import sys
import pathlib

from sage.all import *
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_graph(G, k: int = 2, eps: float = 0.001):
    """Solve MILP 8 for all ell=1,...,N-1 and return the best solution."""

    # The graph is relabeled in main(), but using an explicit vertex order here
    # makes the adjacency matrix construction robust.
    vertices = list(G.vertex_iterator())
    A = np.array(G.adjacency_matrix(vertices=vertices), dtype=float)

    N = A.shape[0]
    t0 = time.time()

    eigs = np.linalg.eigh(A)[0]
    eigs[np.abs(eigs) < 1e-10] = 0
    eigs = np.round(eigs, 10)

    uniq, mult = np.unique(eigs, return_counts=True)
    order = np.argsort(-uniq)

    theta = uniq[order]
    m = mult[order].astype(int)
    d = len(theta) - 1

    th_pow = [[theta[j] ** i for i in range(k + 1)] for j in range(d + 1)]

    best_obj = None
    best_ell = None

    for ell in range(1, N):
        mdl = gp.Model(f"MILP8_{G.name()}_ell{ell}")
        mdl.Params.OutputFlag = 0
        mdl.Params.IntegralityFocus = 1

        a = mdl.addVars(k + 1, lb=-GRB.INFINITY, name="a")
        b = mdl.addVars(d + 1, vtype=GRB.BINARY, name="b")
        c = mdl.addVars(d + 1, vtype=GRB.BINARY, name="c")

        mtb = gp.quicksum(int(m[j]) * b[j] for j in range(d + 1))
        mtc = gp.quicksum(int(m[j]) * c[j] for j in range(d + 1))

        mdl.setObjective(1.0 + (N - mtb) / float(ell), GRB.MAXIMIZE)

        mdl.addConstr(
            gp.quicksum(
                int(m[j])
                * gp.quicksum(a[i] * th_pow[j][i] for i in range(k + 1))
                for j in range(d + 1)
            ) == 0
        )

        for j in range(d + 1):
            ptheta = gp.quicksum(a[i] * th_pow[j][i] for i in range(k + 1))

            mdl.addConstr(ptheta - b[j] + eps <= 0)
            mdl.addConstr(ptheta - c[j] <= 0)
            mdl.addConstr(ptheta + (1 - c[j]) - eps >= 0)

        mdl.addConstr(mtc == ell)

        mdl.optimize()

        if mdl.status == GRB.OPTIMAL:
            obj = mdl.ObjVal
            if best_obj is None or obj > best_obj:
                best_obj = obj
                best_ell = ell

    run_time = time.time() - t0
    return best_obj, best_ell, run_time


def main(list_file="graph_list.txt", out_csv="milp8_results_k=3.csv"):
    list_path = pathlib.Path(list_file)

    if not list_path.exists():
        sys.exit(f"List file '{list_file}' not found.")

    results = []

    for raw_name in list_path.read_text().splitlines():
        name = raw_name.strip()

        if not name or name.startswith("#"):
            continue

        try:
            G = getattr(graphs, name)()

            # Important fix:
            # Some Sage graphs, for example TutteGraph, have mixed vertex labels
            # such as integers and tuples. Relabeling avoids sorting/comparison
            # errors when constructing the adjacency matrix.
            G.relabel(inplace=True)

            # Restore the original graph name after relabeling.
            G.name(name)

        except AttributeError:
            print(f"[skip] {name}: not in sage.graphs.*")
            continue

        print(f"[solve] {name:30s}", end=" … ", flush=True)

        obj, ell, secs = solve_graph(G)

        if obj is None:
            print("no OPTIMAL solution")
        else:
            print(f"obj = {obj:.6g}, ell = {ell}, t = {secs:.2f}s")

        results.append((name, obj, ell, secs))

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["graph", "objective", "ell", "time_seconds"])
        w.writerows(results)

    print(f"\nFinished ✔  →  {out_csv} written with {len(results)} rows.")


if __name__ == "__main__":
    main()