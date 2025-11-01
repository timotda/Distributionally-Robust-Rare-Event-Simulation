import cvxpy as cp
import numpy as np
from joblib import Parallel, delayed

def solve_distance(
    x0: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    v: float,
    solver=cp.ECOS,
    solver_kwargs: dict = None,
    return_x_star: bool = False
) -> float:
    """
    Solve
        minimize   ½‖x - x0‖₂²
        s.t.       y ≥ A x + a
                   y ≥ B x + b
                   wᵀ y ≤ –nu

    for a single x0 ∈ R^d, and return the distance ‖x* – x0‖₂.

    Args:
      x0            (d,)     reference point
      A, B          (d,d)    problem matrices
      a, b, w       (d,)     vectors
      nu            float    scalar ν
      solver        cvxpy.Solver  (default OSQP)
      solver_kwargs dict     extra args passed to `problem.solve(...)`
                              (e.g. `{'verbose': False, 'osqp':{'parallel':False}}`)

    Returns:
      distance      float    ‖x* – x0‖₂
    """
    d = A.shape[1]
    m = A.shape[0]
    # decision variables
    x = cp.Variable(d)
    y = cp.Variable(m)

    # constraints
    constraints = [
        y >= A @ x + a,
        y >= B @ x + b,
        w.T @ y <= -v
    ]
    # objective
    objective = cp.Minimize(0.5 * cp.sum_squares(x - x0))

    # build & solve
    prob = cp.Problem(objective, constraints)
    if solver_kwargs is None:
        # ensure one thread per solve by default
        solver_kwargs = {'verbose': False}
      
    prob.solve(solver=solver, **solver_kwargs)
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"QP failed with status {prob.status}")
    # compute distance
    x_star = x.value
    distance = float(np.linalg.norm(x_star - x0))

    if return_x_star:
        return x_star,distance
    else:
        return distance


def compute_distances_parallel(
    X0: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    v: float,
    solver=cp.ECOS,
    solver_kwargs: dict = None,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Compute distances = [solve_distance(x0_i, ...)] for each row x0_i in X0,
    running up to |n_jobs| solves in parallel.

    Args:
      X0            (N,d)    array of N reference points
      A, B          (d,d)
      a, b, w       (d,)
      nu            float
      solver        cvxpy.Solver
      solver_kwargs dict      extra args to pass to solve_distance
      n_jobs        int       # of parallel workers (joblib style; -1 = all cores)

    Returns:
      distances     (N,)     array of distances
    """
    # wrap solve_distance with fixed problem data

    func = delayed(solve_distance)
    results = Parallel(n_jobs=n_jobs)(
        func(
            X0[:,i], A, B, a, b, w, v,
            solver=cp.ECOS,
            solver_kwargs=solver_kwargs
        )
        for i in range(X0.shape[1])
    )
    return np.array(results)

def debug_distances(X0, A, B, a, b, w, v):
    # solver settings:
    solver_kwargs = {
    "verbose":     True,
    "max_iter":    200_000,
    "eps_abs":     1e-8,     # keep absolute residual tight
    "eps_rel":     1e-4,     # allow a larger gap
    "warm_start":  True,
    "polish":      True      # post‑solve polishing
}

    for i in range(X0.shape[1]):
        x0 = X0[:, i]
        print(f"\n\n--- Sample #{i} x0 = {x0.round(4)}  ---")
        try:
            # re‑build the problem each time so you see fresh verbose logs
            d = solve_distance(
                x0, A, B, a, b, w, v,
                solver=cp.ECOS,
                solver_kwargs=solver_kwargs
            )
            print(f"  → solved, distance = {d:.6f}")
        except Exception as e:
            print(f"  → **Failed** on sample {i} !\nError: {e}")
            # if CVXPY has a prob object in scope, you can also print:
            # print("Last CVXPY status:", prob.status)
            break
