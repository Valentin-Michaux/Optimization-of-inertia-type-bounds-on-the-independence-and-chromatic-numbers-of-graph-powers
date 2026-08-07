"""
Compute the exact k-chromatic number chi_k

For each graph G and each k in POWER_VALUES, this script constructs the
k-th power graph G^k and computes

    chi_k(G) = chi(G^k).

@author: Valentin Michaux
"""

import csv
import sys
import time
import pathlib
from sage.all import *


# Choose the values of k you want.
# For example, [2, 3] computes the chromatic number of G^2 and G^3.
POWER_VALUES = [2, 3]

# Exact chromatic-number algorithm used by Sage.
# "MILP" is usually a good choice when a MILP solver such as Gurobi is available.
# You can also use "DLX" if desired.
CHROMATIC_ALGORITHM = "MILP"

# Set this to "Gurobi" if Gurobi is configured as a Sage MILP solver.
# Leaving it as None makes Sage use its default MILP solver.
CHROMATIC_SOLVER = None

# If True, also saves the 0/1 adjacency matrices B for each graph and k.
SAVE_MATRICES = False
MATRIX_DIR = "power_adjacency_matrices"


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

            # Important fix for graphs like TutteGraph:
            # Some Sage graphs have mixed vertex labels, e.g. integers and tuples.
            # Relabeling changes vertices to 0,1,2,... and avoids comparison errors.
            G.relabel(inplace=True)

            # Restore the original graph name after relabeling.
            G.name(name)

            loaded.append(G)

        except AttributeError:
            print(f"[skip] {name}: not in sage.graphs.*")

    return loaded


def power_adjacency_matrix_from_walks(G, k):
    """
    Return the 0/1 adjacency matrix B where

        B_ij = 1 iff (A^ell)_ij > 0 for at least one ell in {1,...,k},

    and

        B_ii = 0.

    This is the adjacency matrix of the k-th power graph G^k.
    """

    # Robust adjacency matrix construction.
    # Even though the graph is relabeled in load_graphs_from_list(),
    # we also give Sage an explicit vertex order here.
    vertices = list(G.vertex_iterator())
    A = G.adjacency_matrix(sparse=True, vertices=vertices)

    n = G.order()

    B = matrix(ZZ, n, n, sparse=True)

    A_power = A

    for ell in range(1, k + 1):
        for (i, j), val in A_power.dict().items():
            if i != j and val > 0:
                B[i, j] = 1

        if ell < k:
            A_power = A_power * A

    # Ensure the diagonal is zero.
    for i in range(n):
        B[i, i] = 0

    return B


def graph_from_adjacency_matrix(B, name=""):
    """
    Build a simple Sage graph from a 0/1 adjacency matrix B.
    """
    n = B.nrows()
    H = Graph()
    H.add_vertices(range(n))

    for (i, j), val in B.dict().items():
        if i < j and val != 0:
            H.add_edge(i, j)

    if name:
        H.name(name)

    return H


def exact_chromatic_number(H):
    """
    Compute the exact chromatic number of H.

    With CHROMATIC_ALGORITHM = "MILP", Sage solves an exact integer
    optimization problem for the chromatic number. If CHROMATIC_SOLVER
    is None, Sage uses its default MILP solver.
    """
    if CHROMATIC_ALGORITHM == "MILP":
        if CHROMATIC_SOLVER is None:
            return H.chromatic_number(algorithm="MILP")
        return H.chromatic_number(
            algorithm="MILP",
            solver=CHROMATIC_SOLVER,
        )

    return H.chromatic_number(algorithm=CHROMATIC_ALGORITHM)


def save_matrix_csv(B, filename):
    """
    Save a Sage matrix B as a CSV file.
    """
    path = pathlib.Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for i in range(B.nrows()):
            w.writerow([int(B[i, j]) for j in range(B.ncols())])


def solve_graph(G, power_values=POWER_VALUES):
    """
    For each k in power_values:
      1. build B where B_ij = 1 iff (A^ell)_ij > 0 for some 1 <= ell <= k,
      2. build the graph H with adjacency matrix B,
      3. compute chi(H), which equals chi_k(G).
    """
    results = {}

    for k in power_values:
        t0 = time.time()

        B = power_adjacency_matrix_from_walks(G, k)
        H = graph_from_adjacency_matrix(B, name=f"{G.name()}_power_{k}")

        chi = exact_chromatic_number(H)
        secs = time.time() - t0

        if SAVE_MATRICES:
            safe_name = G.name().replace(" ", "_").replace("/", "_")
            save_matrix_csv(
                B,
                f"{MATRIX_DIR}/{safe_name}_power_{k}.csv",
            )

        results[k] = {
            "power_graph_edges": H.size(),
            "chromatic_number": chi,
            "time_seconds": secs,
        }

    return results


def main(list_file="graph_list.txt", out_csv="power_graph_chromatic_results.csv"):
    rows = []

    for G in load_graphs_from_list(list_file):
        print(f"[solve] {G.name():30s}", end=" … ", flush=True)

        total_t0 = time.time()

        try:
            result = solve_graph(G)
            total_time = time.time() - total_t0

            print_parts = []
            for k in POWER_VALUES:
                print_parts.append(
                    f"chi_{k} = {result[k]['chromatic_number']}"
                )

            print(", ".join(print_parts) + f", t = {total_time:.2f}s")

            row = {
                "graph": G.name(),
                "vertices": G.order(),
                "edges_original": G.size(),
                "total_time_seconds": total_time,
                "status": "OK",
            }

            for k in POWER_VALUES:
                row[f"edges_power_{k}"] = result[k]["power_graph_edges"]
                row[f"chromatic_power_{k}"] = result[k]["chromatic_number"]
                row[f"time_power_{k}_seconds"] = result[k]["time_seconds"]

            rows.append(row)

        except Exception as e:
            total_time = time.time() - total_t0
            print(f"ERROR: {e}")

            row = {
                "graph": G.name(),
                "vertices": G.order(),
                "edges_original": G.size(),
                "total_time_seconds": total_time,
                "status": f"ERROR: {e}",
            }

            for k in POWER_VALUES:
                row[f"edges_power_{k}"] = ""
                row[f"chromatic_power_{k}"] = ""
                row[f"time_power_{k}_seconds"] = ""

            rows.append(row)

    headers = [
        "graph",
        "vertices",
        "edges_original",
    ]

    for k in POWER_VALUES:
        headers += [
            f"edges_power_{k}",
            f"chromatic_power_{k}",
            f"time_power_{k}_seconds",
        ]

    headers += [
        "total_time_seconds",
        "status",
    ]

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    print(f"\nFinished ✔  →  {out_csv} written with {len(rows)} rows.")


if __name__ == "__main__":
    main()
