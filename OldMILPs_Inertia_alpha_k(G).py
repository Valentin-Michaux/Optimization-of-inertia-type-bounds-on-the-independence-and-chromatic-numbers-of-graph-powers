"""
Implement the old MILPs to compute the inertia-type bound for the k-independence number

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


# Choose the values of k you want.
# Use [2] if you only want k = 2.
# Use [2, 3] if you want both tables.
POWER_VALUES = [2, 3]


def load_graphs_from_list(list_file="graph_list.txt"):
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

            # Fix for TutteGraph and other Sage graphs with mixed vertex labels.
            G.relabel(inplace=True)

            # Restore original graph name.
            G.name(name)

            loaded.append(G)

        except AttributeError:
            print(f"[skip] {name}: not in sage.graphs.*")

    return loaded


def adjacency_matrix_numpy(G):
    """
    Return a dense numpy adjacency matrix using an explicit vertex order.
    This avoids the TutteGraph mixed-label sorting problem.
    """
    vertices = list(G.vertex_iterator())
    return np.array(G.adjacency_matrix(vertices=vertices), dtype=float)


def solve_graph_milp1(G, k: int = 2, M: float = 1000, eps: float = 1):
    """
    Solve MILP1 for graph G and power k.

    Returns:
        objective, best_vertex, run_time_seconds, status
    """
    total_start = time.time()

    A = adjacency_matrix_numpy(G)
    n = A.shape[0]

    eigs = np.linalg.eigh(A)[0]
    eigs[np.abs(eigs) < 1e-10] = 0
    eigs = np.round(eigs, 10)

    uniq, mult = np.unique(eigs, return_counts=True)
    order = np.argsort(-uniq)

    unique_eigenvalues = uniq[order]
    m = mult[order].astype(int)

    d = len(unique_eigenvalues) - 1

    A_powers = [
        np.linalg.matrix_power(A, i)
        for i in range(k + 1)
    ]

    eigenvalue_powers = [
        [
            unique_eigenvalues[j] ** i
            for i in range(k + 1)
        ]
        for j in range(d + 1)
    ]

    best_obj = None
    best_vertex = None

    for u in range(n):
        model = gp.Model(f"MILP1_{G.name()}_k{k}_vertex_{u}")
        model.Params.OutputFlag = 0
        model.Params.IntegralityFocus = 1
        model.Params.IntFeasTol = 0.1

        a = model.addVars(k + 1, lb=-GRB.INFINITY, name="a")
        b = model.addVars(d + 1, vtype=GRB.BINARY, name="b")

        model.setObjective(
            gp.quicksum(int(m[j]) * b[j] for j in range(d + 1)),
            GRB.MINIMIZE,
        )

        for v in range(n):
            if v != u:
                model.addConstr(
                    gp.quicksum(
                        a[i] * A_powers[i][v, v]
                        for i in range(k + 1)
                    ) >= 0
                )

        model.addConstr(
            gp.quicksum(
                a[i] * A_powers[i][u, u]
                for i in range(k + 1)
            ) == 0
        )

        for j in range(d + 1):
            model.addConstr(
                gp.quicksum(
                    a[i] * eigenvalue_powers[j][i]
                    for i in range(k + 1)
                ) - M * b[j] + eps <= 0
            )

        model.optimize()

        if model.status == GRB.OPTIMAL:
            obj = model.ObjVal

            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_vertex = u

    total_time = time.time() - total_start

    if best_obj is None:
        return None, None, total_time, "NO_OPTIMAL_SOLUTION"

    return best_obj, best_vertex, total_time, "OK"


def main(list_file="graph_list.txt", out_csv="MILP1_results.csv"):
    rows = []

    for G in load_graphs_from_list(list_file):
        for k in POWER_VALUES:
            print(f"[solve] {G.name():30s} k={k}", end=" … ", flush=True)

            try:
                obj, best_vertex, secs, status = solve_graph_milp1(G, k=k)

                if obj is None:
                    print(f"{status}, t = {secs:.2f}s")
                else:
                    print(
                        f"obj = {obj:.6g}, "
                        f"best_vertex = {best_vertex}, "
                        f"t = {secs:.2f}s"
                    )

                rows.append({
                    "graph": G.name(),
                    "k": k,
                    "objective": obj,
                    "best_vertex": best_vertex,
                    "time_seconds": secs,
                    "status": status,
                })

            except Exception as e:
                print(f"ERROR: {e}")

                rows.append({
                    "graph": G.name(),
                    "k": k,
                    "objective": "",
                    "best_vertex": "",
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
                "best_vertex",
                "time_seconds",
                "status",
            ],
        )

        w.writeheader()
        w.writerows(rows)

    print(f"\nFinished ✔  →  {out_csv} written with {len(rows)} rows.")


if __name__ == "__main__":
    main()