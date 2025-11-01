import cvxpy as cp
import numpy as np
from joblib import Parallel, delayed


def solve_distance(x0, Q_costs, L_costs, F_costs, v,x_charm, solver=cp.SCS):
    d = Q_costs.shape[0]

    x = cp.Variable(d)
    
    constraints = [
        cp.quad_form(x - x_charm, Q_costs)
        + cp.norm1(L_costs.T @ (x - x_charm))
        + F_costs
        <= v
    ]
    objective = cp.Minimize(0.5 * cp.sum_squares(x - x0))

    prob = cp.Problem(objective, constraints)

    solver_kwargs = {'verbose': False}
    prob.solve(solver=solver, **solver_kwargs)

    x_star = x.value
    distance = float(np.linalg.norm(x_star - x0))

    return distance
    


def compute_distances_parallel(x0, Q_costs, L_costs, F_costs, v,x_charm, solver=cp.SCS, n_jobs = -1):
    func = delayed(solve_distance)
    results = Parallel(n_jobs=n_jobs)(
        func(
            x0[:,i],Q_costs,L_costs,F_costs,v,x_charm,
            solver=solver,
        )
        for i in range(x0.shape[1])
    )
    return np.array(results)

