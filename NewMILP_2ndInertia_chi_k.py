"""
Implement the paper's unified normalized MILP for the second inertia-type lower
bound on the distance-k chromatic number:

    maximize    1 + t
    subject to  sum_j m_j p(theta_j) = 0,
                p(theta_j) - b_j + epsilon <= 0,
                p(theta_j) - c_j <= 0,
                p(theta_j) + (1-c_j) - epsilon >= 0,
                sum_ell y_ell = 1,
                m^T c = sum_ell ell y_ell,
                0 <= t <= n,
                ell t <= n - m^T b + ell n (1-y_ell),
                b_j, c_j, y_ell binary.

The third sign inequality is the reverse implication required to make
c_j=1 correspond to p(theta_j)>0 (up to epsilon separation).

There is NO numerical spectral big-M in this reformulated model.  The term
ell*n*(1-y_ell) belongs only to the exact selector linearization from the
displayed unified MILP.

The bound applies only to k-partially walk-regular graphs; this script checks
that hypothesis exactly before solving.

The script reports only the optimal value of the chromatic MILP. No
associated upper bound on alpha_k(G) is computed.

Spectrum handling:
- multiplicities are obtained exactly from the characteristic polynomial;
- distinct eigenvalues are never rounded, clustered, or averaged;
- only exact zero is converted explicitly to 0.0 before passing values to Gurobi.
"""

import csv
import pathlib
import sys
import time

import gurobipy as gp
from gurobipy import GRB
from sage.all import AA, graphs

POWER_VALUES = [2, 3]
EPSILON = 1.0e-3


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


class NotKPartiallyWalkRegular(ValueError):
    """Raised when the chromatic inertia bound does not apply to the graph."""


def is_k_partially_walk_regular(A, k: int) -> bool:
    """Check the defining diagonal condition exactly, using the integer matrix."""
    n = A.nrows()
    for ell in range(k + 1):
        A_ell = A**ell
        reference = A_ell[0, 0]
        if any(A_ell[v, v] != reference for v in range(1, n)):
            return False
    return True


def configure_gurobi(model: gp.Model) -> None:
    """Use strict solver tolerances because epsilon separates strict signs."""
    model.Params.OutputFlag = 0
    model.Params.IntegralityFocus = 1
    model.Params.IntFeasTol = 1.0e-9
    model.Params.FeasibilityTol = 1.0e-9
    model.Params.NumericFocus = 2




def solve_graph(G, k: int = 2, eps: float = EPSILON):
    """Solve the unified chromatic MILP and return its optimal objective value."""
    if k < 1:
        raise ValueError("k must be a positive integer.")
    if not 0.0 < eps < 1.0:
        raise ValueError("The normalized formulation requires 0 < epsilon < 1.")

    t0 = time.perf_counter()
    A, theta, m = exact_adjacency_and_spectrum(G)
    N = A.nrows()

    if not is_k_partially_walk_regular(A, k):
        raise NotKPartiallyWalkRegular(
            f"{G.name()} is not {k}-partially walk-regular."
        )

    d = len(theta) - 1
    th_pow = [
        [theta[j]**i for i in range(k + 1)]
        for j in range(d + 1)
    ]

    mdl = gp.Model(f"MILP10_{G.name()}_k{k}")
    configure_gurobi(mdl)

    a = mdl.addVars(k + 1, lb=-GRB.INFINITY, name="a")
    b = mdl.addVars(d + 1, vtype=GRB.BINARY, name="b")
    c = mdl.addVars(d + 1, vtype=GRB.BINARY, name="c")
    y = mdl.addVars(range(1, N), vtype=GRB.BINARY, name="y")
    t = mdl.addVar(lb=0.0, ub=float(N), vtype=GRB.CONTINUOUS, name="t")

    mtb = gp.quicksum(m[j] * b[j] for j in range(d + 1))
    mtc = gp.quicksum(m[j] * c[j] for j in range(d + 1))

    mdl.setObjective(1.0 + t, GRB.MAXIMIZE)

    mdl.addConstr(
        gp.quicksum(
            m[j] * gp.quicksum(
                a[i] * th_pow[j][i] for i in range(k + 1)
            )
            for j in range(d + 1)
        ) == 0.0,
        name="trace_zero",
    )

    for j in range(d + 1):
        ptheta = gp.quicksum(
            a[i] * th_pow[j][i] for i in range(k + 1)
        )

        mdl.addConstr(
            ptheta - b[j] + eps <= 0.0,
            name=f"nonnegative_indicator_{j}",
        )
        mdl.addConstr(
            ptheta - c[j] <= 0.0,
            name=f"positive_upper_{j}",
        )
        # Reverse implication: c_j=1 forces p(theta_j) >= epsilon.
        mdl.addConstr(
            ptheta + (1 - c[j]) - eps >= 0.0,
            name=f"positive_lower_{j}",
        )

    mdl.addConstr(
        gp.quicksum(y[ell] for ell in range(1, N)) == 1,
        name="choose_one_ell",
    )
    mdl.addConstr(
        mtc == gp.quicksum(ell * y[ell] for ell in range(1, N)),
        name="positive_multiplicity",
    )

    # Exact selector linearization from the displayed unified formulation.
    for ell in range(1, N):
        mdl.addConstr(
            ell * t <= N - mtb + ell * N * (1 - y[ell]),
            name=f"ratio_selector_{ell}",
        )

    # Maximize the second inertia-type chromatic bound.
    mdl.optimize()

    run_time = time.perf_counter() - t0

    if mdl.Status != GRB.OPTIMAL:
        return None, run_time

    # Optimal MILP value: 1 + t.
    chi_objective = float(mdl.ObjVal)
    return chi_objective, run_time


def main(
    list_file="graph_list.txt",
    out_csv="NewMILP_2ndInertia_chi_k_results.csv",
):
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
            G.relabel(inplace=True)
            G.name(name)
        except AttributeError:
            print(f"[skip] {name}: not in sage.graphs.*")
            continue

        for k in POWER_VALUES:
            print(f"[solve-new] {name:30s} k={k}", end=" ... ", flush=True)
            try:
                obj, secs = solve_graph(G, k=k, eps=EPSILON)
                status = "OK" if obj is not None else "NO_OPTIMAL_SOLUTION"

                if obj is None:
                    print(f"{status}, t = {secs:.2f}s")
                else:
                    print(f"obj = {obj:.6g}, t = {secs:.2f}s")

                results.append(
                    (name, k, EPSILON, obj, secs, status)
                )

            except NotKPartiallyWalkRegular:
                status = "NOT_K_PARTIALLY_WALK_REGULAR"
                print(status)
                results.append(
                    (name, k, EPSILON, "", "", status)
                )

            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append(
                    (name, k, EPSILON, "", "", f"ERROR: {exc}")
                )

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "graph", "k", "epsilon",
            "chi_objective",
            "time_seconds",
            "status",
        ])
        writer.writerows(results)

    print(f"\nFinished -> {out_csv} written with {len(results)} rows.")


if __name__ == "__main__":
    main()
