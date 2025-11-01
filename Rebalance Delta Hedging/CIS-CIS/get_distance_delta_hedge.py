import cvxpy as cp
import numpy as np
from joblib import Parallel, delayed


def solve_distance(y0, mu, Lchol, Q_costs, L_costs, F_costs, v, x_charm, solver=cp.SCS, return_y_star=False):
    #    y ~ N(0,I), x = mu + Lchol @ y
    #    Distance = min_y 0.5 ||y - y0||^2  s.t.  cost(mu + L y) <= v   
    d = Q_costs.shape[0]
    y = cp.Variable(d)
    x = mu + Lchol @ y

    constraints = [
        cp.quad_form(x - x_charm, Q_costs)
        + cp.norm1(L_costs.T @ (x - x_charm))
        + F_costs
        <= v
    ]

    objective = cp.Minimize(0.5 * cp.sum_squares(y - y0))
     
    prob = cp.Problem(objective, constraints)

    solver_kwargs = {'verbose': False}
    prob.solve(solver=solver, **solver_kwargs)
    y_star = y.value
    distance = float(np.linalg.norm(y_star - y0))

    if return_y_star:
        return y_star,distance
    else:
        return distance
    


def compute_distances_parallel(y0, mu, Lchol, Q_costs, L_costs, F_costs, v, x_charm, solver=cp.SCS, n_jobs = -1):
    func = delayed(solve_distance)

    results = Parallel(n_jobs=n_jobs)(
        func(
            y0[:,i], mu, Lchol, Q_costs, L_costs, F_costs, v, x_charm,
            solver=solver,
        )
        for i in range(y0.shape[1])
    )
    return np.array(results)

