"""
Implement the original vertex-dependent big-M MILPs for the inertia-type upper
bound on alpha_k(G).

For each distinguished vertex u:

    minimize    m^T b
    subject to  sum_i a_i (A^i)_vv >= 0          for v != u,
                sum_i a_i (A^i)_uu  = 0,
                sum_i a_i theta_j^i - M b_j + eps <= 0,
                b_j binary.

The final bound is the minimum objective value over all u in V(G).

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

BIG_M = 1000.0
EPSILON = 1.0


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


def solve_graph_milp1(
    G,
    k: int = 2,
    big_m: float = BIG_M,
    eps: float = EPSILON,
) -> Tuple[Optional[float], Optional[int], float, str]:
    """Solve the paper's original big-M MILP for every distinguished vertex."""
    if k < 1:
        raise ValueError("k must be a positive integer.")
    if big_m <= 0.0:
        raise ValueError("M must be positive.")
    if eps <= 0.0:
        raise ValueError("epsilon must be positive.")
    if eps >= big_m:
        raise ValueError("The big-M formulation requires epsilon < M.")

    total_start = time.perf_counter()

    A, theta, multiplicities = exact_adjacency_and_spectrum(G)
    n = A.nrows()
    d = len(theta) - 1

    # These matrices have exact integer entries.
    A_powers = [A**i for i in range(k + 1)]
    theta_powers = [
        [theta_j**i for i in range(k + 1)]
        for theta_j in theta
    ]

    best_obj: Optional[float] = None
    best_vertex: Optional[int] = None
    nonoptimal_statuses = []

    for u in range(n):
        model = gp.Model(f"MILP1_{G.name()}_k{k}_vertex_{u}")
        configure_gurobi(model)

        a = model.addVars(k + 1, lb=-GRB.INFINITY, name="a")
        b = model.addVars(d + 1, vtype=GRB.BINARY, name="b")

        model.setObjective(
            gp.quicksum(multiplicities[j] * b[j] for j in range(d + 1)),
            GRB.MINIMIZE,
        )

        for v in range(n):
            diagonal_value = gp.quicksum(
                a[i] * float(A_powers[i][v, v]) for i in range(k + 1)
            )
            if v == u:
                model.addConstr(diagonal_value == 0.0, name=f"diag_equal_{u}")
            else:
                model.addConstr(
                    diagonal_value >= 0.0,
                    name=f"diag_nonnegative_{v}",
                )

        # p(theta_j) - M b_j + epsilon <= 0.
        for j in range(d + 1):
            p_theta_j = gp.quicksum(
                a[i] * theta_powers[j][i] for i in range(k + 1)
            )
            model.addConstr(
                p_theta_j - big_m * b[j] + eps <= 0.0,
                name=f"spectral_{j}",
            )

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            obj = float(model.ObjVal)
            if best_obj is None or obj < best_obj - 1.0e-9:
                best_obj = obj
                best_vertex = u
        else:
            nonoptimal_statuses.append(int(model.Status))

    total_time = time.perf_counter() - total_start

    if best_obj is None:
        status_text = ",".join(map(str, sorted(set(nonoptimal_statuses)))) or "unknown"
        return None, None, total_time, f"NO_OPTIMAL_SOLUTION[{status_text}]"

    return best_obj, best_vertex, total_time, "OK"


def main(
    list_file: str = "graph_list.txt",
    out_csv: str = "MILP1_alpha_k_results.csv",
) -> None:
    rows = []

    for G in load_graphs_from_list(list_file):
        for k in POWER_VALUES:
            print(f"[solve] {G.name():30s} k={k}", end=" ... ", flush=True)
            try:
                obj, best_vertex, secs, status = solve_graph_milp1(
                    G, k=k, big_m=BIG_M, eps=EPSILON
                )
                if obj is None:
                    print(f"{status}, t = {secs:.2f}s")
                else:
                    print(
                        f"obj = {obj:.6g}, best_vertex = {best_vertex}, "
                        f"t = {secs:.2f}s"
                    )

                rows.append({
                    "graph": G.name(),
                    "k": k,
                    "M": BIG_M,
                    "epsilon": EPSILON,
                    "objective": "" if obj is None else obj,
                    "best_vertex": "" if best_vertex is None else best_vertex,
                    "time_seconds": secs,
                    "status": status,
                })
            except Exception as exc:
                print(f"ERROR: {exc}")
                rows.append({
                    "graph": G.name(),
                    "k": k,
                    "M": BIG_M,
                    "epsilon": EPSILON,
                    "objective": "",
                    "best_vertex": "",
                    "time_seconds": "",
                    "status": f"ERROR: {exc}",
                })

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "graph", "k", "M", "epsilon", "objective",
                "best_vertex", "time_seconds", "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinished -> {out_csv} written with {len(rows)} rows.")


if __name__ == "__main__":
    main()
