"""
Implement the new MILP to compute the inertia-type bound for the k-independence number

@author: Valentin Michaux
"""

import csv
import sys
import time
import pathlib

from sage.all import *
import gurobipy as gp
from gurobipy import GRB
import numpy as np


# Values of k to run.
POWER_VALUES = [2, 3]

# Output file. Excel can open this directly.
OUT_CSV = "MILP3_results.csv"

# Big-M and epsilon values.
M = 1000
EPS = 1


def load_graphs_from_list(list_file="graph_list.txt"):
    """
    Load Sage graphs from graph_list.txt.

    Each line of graph_list.txt should contain a Sage graph constructor name,
    for example:

        PetersenGraph
        TutteGraph
        Balaban10Cage
    """
    list_path = pathlib.Path(list_file)

    if not list_path.exists():
        sys.exit(f"List file '{list_file}' not found.")

    loaded = []

    for raw_name in list_path.read_text().splitlines():
        name = raw_name.strip()

        if not name or name.startswith("#"):
            continue

        try:
            G = getattr(graphs, name)()

            # Fix for TutteGraph and other graphs with mixed vertex labels.
            # Sage sometimes has vertices like integers and tuples together.
            # Relabeling changes them to 0,1,2,...
            G.relabel(inplace=True)

            # Restore original graph name after relabeling.
            G.name(name)

            loaded.append(G)

        except AttributeError:
            print(f"[skip] {name}: not in sage.graphs.*")

    return loaded


def adjacency_matrix_numpy(G):
    """
    Return the adjacency matrix as a dense numpy array.

    The explicit vertex order avoids the TutteGraph mixed-label sorting problem.
    """
    vertices = list(G.vertex_iterator())
    A = G.adjacency_matrix(vertices=vertices)
    return np.array(A, dtype=float)


def matrix_power(A, i):
    """
    Compute A^i.
    """
    return np.linalg.matrix_power(A, i)


def solve_graph_milp3(G, k: int, M: float = M, eps: float = EPS):
    """
    Solve MILP 3 for one graph G and one value of k.

    Returns:
        objective, run_time_seconds, status
    """
    total_start = time.time()

    A = adjacency_matrix_numpy(G)
    n = A.shape[0]

    # Eigenvalues.
    eigs = np.linalg.eigh(A)[0]
    eigs[np.abs(eigs) < 1e-10] = 0
    eigs = np.round(eigs, 10)

    uniq, mult = np.unique(eigs, return_counts=True)

    # Descending order.
    order = np.argsort(-uniq)
    unique_eigenvalues = uniq[order]
    m = mult[order].astype(int)

    d = len(unique_eigenvalues) - 1

    # Pre-compute powers.
    A_powers = [
        matrix_power(A, i)
        for i in range(k + 1)
    ]

    eigenvalue_powers = [
        [
            unique_eigenvalues[j] ** i
            for i in range(k + 1)
        ]
        for j in range(d + 1)
    ]

    # MILP 3.
    model = gp.Model(f"MILP3_{G.name()}_k{k}")
    model.Params.OutputFlag = 0
    model.Params.IntegralityFocus = 1
    model.Params.IntFeasTol = 0.1

    a = model.addVars(k + 1, lb=-GRB.INFINITY, name="a")
    b = model.addVars(d + 1, vtype=GRB.BINARY, name="b")

    # Objective: minimize m^T b.
    model.setObjective(
        gp.quicksum(int(m[j]) * b[j] for j in range(d + 1)),
        GRB.MINIMIZE,
    )

    # Constraints:
    # sum_i a_i (A^i)_{vv} >= 0 for every vertex v.
    for v in range(n):
        model.addConstr(
            gp.quicksum(
                a[i] * A_powers[i][v, v]
                for i in range(k + 1)
            ) >= 0
        )

    # Eigenvalue constraints.
    for j in range(d + 1):
        model.addConstr(
            gp.quicksum(
                a[i] * eigenvalue_powers[j][i]
                for i in range(k + 1)
            ) - M * b[j] + eps <= 0
        )

    model.optimize()

    run_time = time.time() - total_start

    if model.status == GRB.OPTIMAL:
        return model.ObjVal, run_time, "OK"

    return None, run_time, f"STATUS_{model.status}"


def main(list_file="graph_list.txt", out_csv=OUT_CSV):
    rows = []

    graphs_to_solve = load_graphs_from_list(list_file)

    for G in graphs_to_solve:
        for k in POWER_VALUES:
            print(f"[solve] {G.name():30s} k={k}", end=" … ", flush=True)

            try:
                obj, secs, status = solve_graph_milp3(G, k)

                if obj is None:
                    print(f"{status}, t = {secs:.2f}s")
                else:
                    print(f"obj = {obj:.6g}, t = {secs:.2f}s")

                rows.append({
                    "graph": G.name(),
                    "k": k,
                    "objective": obj if obj is not None else "",
                    "time_seconds": secs,
                    "status": status,
                })

            except Exception as e:
                print(f"ERROR: {e}")

                rows.append({
                    "graph": G.name(),
                    "k": k,
                    "objective": "",
                    "time_seconds": "",
                    "status": f"ERROR: {e}",
                })

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "graph",
                "k",
                "objective",
                "time_seconds",
                "status",
            ],
        )

        w.writeheader()
        w.writerows(rows)

    print(f"\nFinished ✔  →  {out_csv} written with {len(rows)} rows.")


if __name__ == "__main__":
    main()