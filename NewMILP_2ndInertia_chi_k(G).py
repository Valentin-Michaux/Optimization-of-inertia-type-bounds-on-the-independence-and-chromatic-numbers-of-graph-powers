'''
Implement the new MILP for the second inertia type for the distance-k chromatic number


@author: Valentin Michaux
'''
import csv
import time
import sys
import pathlib

from sage.all import *
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_graph(G, k: int = 2, eps: float = 0.001):
    """Solve the single MILP10. Return (objective_value, ell, run_time_seconds)."""

    # Robust fix for graphs like TutteGraph:
    # use an explicit vertex ordering when constructing the adjacency matrix.
    vertices = list(G.vertex_iterator())
    A = np.array(G.adjacency_matrix(vertices=vertices), dtype=float)

    N = A.shape[0]
    t0 = time.time()

    # 1) eigen-analysis
    eigs = np.linalg.eigh(A)[0]
    eigs[np.abs(eigs) < 1e-10] = 0
    eigs = np.round(eigs, 10)

    uniq, mult = np.unique(eigs, return_counts=True)
    order = np.argsort(-uniq)

    theta = uniq[order]
    m = mult[order].astype(int)
    d = len(theta) - 1

    # 2) pre-compute eigenvalue powers
    th_pow = [[theta[j] ** i for i in range(k + 1)] for j in range(d + 1)]

    # 3) build MILP10
    mdl = gp.Model(f"MILP10_{G.name()}")
    mdl.Params.OutputFlag = 0
    mdl.Params.IntegralityFocus = 1

    a = mdl.addVars(k + 1, lb=-GRB.INFINITY, name="a")
    b = mdl.addVars(d + 1, vtype=GRB.BINARY, name="b")
    c = mdl.addVars(d + 1, vtype=GRB.BINARY, name="c")
    y = mdl.addVars(range(1, N), vtype=GRB.BINARY, name="y")
    t = mdl.addVar(lb=0.0, ub=float(N), vtype=GRB.CONTINUOUS, name="t")

    mtb = gp.quicksum(int(m[j]) * b[j] for j in range(d + 1))
    mtc = gp.quicksum(int(m[j]) * c[j] for j in range(d + 1))

    mdl.setObjective(1.0 + t, GRB.MAXIMIZE)

    # spectral equality: sum_j m_j p(theta_j) = 0
    mdl.addConstr(
        gp.quicksum(
            int(m[j])
            * gp.quicksum(a[i] * th_pow[j][i] for i in range(k + 1))
            for j in range(d + 1)
        ) == 0
    )

    # eigenvalue sign constraints
    for j in range(d + 1):
        ptheta = gp.quicksum(a[i] * th_pow[j][i] for i in range(k + 1))
        mdl.addConstr(ptheta - b[j] + eps <= 0)
        mdl.addConstr(ptheta - c[j] <= 0)

    # choose one ell
    mdl.addConstr(gp.quicksum(y[ell] for ell in range(1, N)) == 1)

    # m^T c = sum ell y_ell
    mdl.addConstr(mtc == gp.quicksum(ell * y[ell] for ell in range(1, N)))

    # big-M-free linearization for t
    for ell in range(1, N):
        mdl.addConstr(
            ell * t <= N - mtb + ell * N * (1 - y[ell])
        )

    mdl.optimize()

    run_time = time.time() - t0

    if mdl.status == GRB.OPTIMAL:
        chosen_ell = None
        for ell in range(1, N):
            if y[ell].X > 0.5:
                chosen_ell = ell
                break

        return mdl.ObjVal, chosen_ell, run_time

    return None, None, run_time


def main(list_file="graph_list.txt", out_csv="milp10_results.csv"):
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
            # Some Sage graphs, such as TutteGraph, have mixed vertex labels,
            # for example integers and tuples. Relabeling gives vertices
            # simple integer labels 0,1,2,...
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