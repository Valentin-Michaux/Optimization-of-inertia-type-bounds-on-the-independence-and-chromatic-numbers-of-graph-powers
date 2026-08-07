"""
Implement the paper's single normalized MILP for the inertia-type upper bound
on alpha_k(G):

    minimize    m^T b
    subject to  sum_i a_i (A^i)_vv >= 0          for every v,
                sum_i a_i theta_j^i - b_j + eps <= 0,
                b_j binary.

For k=2, the n diagonal constraints are replaced by the two exactly equivalent
minimum-/maximum-degree inequalities proved in the paper.

There is NO numerical big-M in this reformulated model.

Spectrum handling:
- multiplicities are obtained exactly from the characteristic polynomial;
- distinct eigenvalues are never rounded, clustered, or averaged;
- only exact zero is converted explicitly to 0.0 before passing values to Gurobi.

Run with SageMath and a licensed Gurobi installation.
"""

import csv
import pathlib
import sys
import time
from typing import Optional, Tuple

import gurobipy as gp
from gurobipy import GRB
from sage.all import AA, graphs

POWER_VALUES = [2, 3]
OUT_CSV = "MILP2_alpha_k_results.csv"

# Normalized counterpart of M=1000, epsilon=1.
EPSILON = 1.0e-3


def load_graphs_from_list(list_file: str = "graph_list.txt"):
    """Load Sage graph constructors listed one per line in list_file."""
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
            G.relabel(inplace=True)
            G.name(name)
            loaded.append(G)
        except AttributeError:
            print(f"[skip] {name}: not in sage.graphs.*")
    return loaded


def exact_adjacency_and_spectrum(G):
    """
    Return the exact Sage adjacency matrix, distinct eigenvalues (as doubles for
    Gurobi), and exact multiplicities.

    Distinct eigenvalues and multiplicities are obtained from the exact
    characteristic polynomial over Sage's Algebraic Real Field (AA).  We do
    NOT round, cluster, average, or otherwise group numerical eigenvalues.

    Only an eigenvalue that is exactly zero in AA is converted explicitly to
    0.0.  Every other algebraic eigenvalue is converted directly to a Python
    double for Gurobi.
    """
    vertices = list(G.vertex_iterator())
    A = G.adjacency_matrix(vertices=vertices)

    roots = list(A.charpoly().roots(ring=AA))
    roots.sort(key=lambda item: item[0], reverse=True)

    multiplicities = [int(mult) for _, mult in roots]
    if sum(multiplicities) != A.nrows():
        raise RuntimeError(
            "The exact characteristic polynomial did not return all eigenvalues."
        )

    theta = [0.0 if root == 0 else float(root) for root, _ in roots]

    # Never silently merge two distinct exact eigenvalues after conversion to
    # double precision.  If this ever happens, the Gurobi model needs a
    # higher-precision/certified numerical interface rather than clustering.
    for j in range(len(theta) - 1):
        if not theta[j] > theta[j + 1]:
            raise FloatingPointError(
                "Two distinct exact eigenvalues are not distinguishable in "
                "double precision; refusing to merge or round them."
            )

    return A, theta, multiplicities


def configure_gurobi(model: gp.Model) -> None:
    """Use strict solver tolerances because epsilon separates strict signs."""
    model.Params.OutputFlag = 0
    model.Params.IntegralityFocus = 1
    model.Params.IntFeasTol = 1.0e-9
    model.Params.FeasibilityTol = 1.0e-9
    model.Params.NumericFocus = 2


def add_closed_walk_constraints(model, a, A_powers, k: int, n: int) -> None:
    """Add the paper's diagonal constraints, including the exact k=2 reduction."""
    if k == 2:
        # (A^2)_vv is exactly the degree of v.
        degrees = [int(A_powers[2][v, v]) for v in range(n)]
        delta = min(degrees)
        Delta = max(degrees)
        model.addConstr(a[0] + delta * a[2] >= 0.0, name="diag_min_degree")
        model.addConstr(a[0] + Delta * a[2] >= 0.0, name="diag_max_degree")
        return

    for v in range(n):
        model.addConstr(
            gp.quicksum(
                a[i] * float(A_powers[i][v, v]) for i in range(k + 1)
            ) >= 0.0,
            name=f"diag_nonnegative_{v}",
        )


def solve_graph_milp2(
    G,
    k: int,
    eps: float = EPSILON,
) -> Tuple[Optional[float], float, str]:
    """Solve the paper's single normalized independence MILP."""
    if k < 1:
        raise ValueError("k must be a positive integer.")
    if not 0.0 < eps < 1.0:
        raise ValueError("The normalized formulation requires 0 < epsilon < 1.")

    total_start = time.perf_counter()

    A, theta, multiplicities = exact_adjacency_and_spectrum(G)
    n = A.nrows()
    d = len(theta) - 1

    A_powers = [A**i for i in range(k + 1)]
    theta_powers = [
        [theta_j**i for i in range(k + 1)]
        for theta_j in theta
    ]

    model = gp.Model(f"MILP2_{G.name()}_k{k}")
    configure_gurobi(model)

    a = model.addVars(k + 1, lb=-GRB.INFINITY, name="a")
    b = model.addVars(d + 1, vtype=GRB.BINARY, name="b")

    model.setObjective(
        gp.quicksum(multiplicities[j] * b[j] for j in range(d + 1)),
        GRB.MINIMIZE,
    )

    add_closed_walk_constraints(model, a, A_powers, k, n)

    # Normalized spectral constraints: p(theta_j) - b_j + epsilon <= 0.
    for j in range(d + 1):
        p_theta_j = gp.quicksum(
            a[i] * theta_powers[j][i] for i in range(k + 1)
        )
        model.addConstr(
            p_theta_j - b[j] + eps <= 0.0,
            name=f"spectral_{j}",
        )

    model.optimize()
    run_time = time.perf_counter() - total_start

    if model.Status == GRB.OPTIMAL:
        return float(model.ObjVal), run_time, "OK"

    return None, run_time, f"STATUS_{int(model.Status)}"


def main(
    list_file: str = "graph_list.txt",
    out_csv: str = OUT_CSV,
) -> None:
    rows = []

    for G in load_graphs_from_list(list_file):
        for k in POWER_VALUES:
            print(f"[solve] {G.name():30s} k={k}", end=" ... ", flush=True)
            try:
                obj, secs, status = solve_graph_milp2(G, k=k, eps=EPSILON)
                if obj is None:
                    print(f"{status}, t = {secs:.2f}s")
                else:
                    print(f"obj = {obj:.6g}, t = {secs:.2f}s")

                rows.append({
                    "graph": G.name(),
                    "k": k,
                    "epsilon": EPSILON,
                    "objective": "" if obj is None else obj,
                    "time_seconds": secs,
                    "status": status,
                })
            except Exception as exc:
                print(f"ERROR: {exc}")
                rows.append({
                    "graph": G.name(),
                    "k": k,
                    "epsilon": EPSILON,
                    "objective": "",
                    "time_seconds": "",
                    "status": f"ERROR: {exc}",
                })

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "graph", "k", "epsilon", "objective", "time_seconds", "status"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinished -> {out_csv} written with {len(rows)} rows.")


if __name__ == "__main__":
    main()
