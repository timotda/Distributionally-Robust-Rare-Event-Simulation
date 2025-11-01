import cvxpy as cp
import numpy as np 
from joblib import Parallel, delayed

def solve_distance(x0, Q, r, s, v, solver=cp.SCS, return_x_star = False):
    d = Q.shape[0]

    x = cp.Variable(d)

    constraints = [
        cp.quad_form(x,Q) + r.T @ x + s <= -v
    ]

    objective = cp.Minimize(0.5 * cp.sum_squares(x - x0))

    prob = cp.Problem(objective, constraints)

    solver_kwargs = {'verbose': False}
    prob.solve(solver=solver, **solver_kwargs)

    x_star = x.value
    distance = float(np.linalg.norm(x_star - x0))

    if return_x_star:
        return x_star,distance
    else:
        return distance
    


def compute_distances_parallel(x0, Q, r, s, v, solver=cp.SCS, n_jobs = -1):
    func = delayed(solve_distance)
    results = Parallel(n_jobs=n_jobs)(
        func(
            x0[:,i],Q,r,s,v,
            solver=solver,
        )
        for i in range(x0.shape[1])
    )
    return np.array(results)

