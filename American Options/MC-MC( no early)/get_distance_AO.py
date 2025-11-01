# get_distance_AO.py
import cvxpy as cp
import numpy as np
from joblib import Parallel, delayed

def solve_distance_AO(
    x0: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    solver=cp.OSQP,
    solver_kwargs: dict | None = None,
    return_x_star: bool = False
):
    """
    Distance from x0 to the polyhedron E = {y : G y >= h}.
    Returns ||x0 - Proj_E(x0)||.
    """
    d = x0.size
    y = cp.Variable(d)
    obj = cp.Minimize(0.5 * cp.sum_squares(y - x0))
    cons = [G @ y >= h]
    prob = cp.Problem(obj, cons)
    if solver_kwargs is None:
        solver_kwargs = dict()
    prob.solve(solver=solver, **solver_kwargs)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Projection infeasible or failed: {prob.status}")
    y_star = y.value
    dist = float(np.linalg.norm(y_star - x0))
    return (dist, y_star) if return_x_star else dist

def compute_distances_AO_parallel(
    X0: np.ndarray,       # shape (d, N)
    G: np.ndarray,
    h: np.ndarray,
    solver=cp.OSQP,
    solver_kwargs: dict | None = None,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Vectorized wrapper: returns distances for each column x0 in X0.
    """
    func = delayed(solve_distance_AO)
    results = Parallel(n_jobs=n_jobs)(
        func(X0[:, i], G, h, solver=solver, solver_kwargs=solver_kwargs)
        for i in range(X0.shape[1])
    )
    return np.array(results)
