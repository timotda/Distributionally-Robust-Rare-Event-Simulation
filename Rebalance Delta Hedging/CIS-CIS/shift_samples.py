import numpy as np 
from get_distance_delta_hedge import compute_distances_parallel

def shift_samples(mu, Lchol, Q_costs, L_costs, F_costs, loss_threshold, R, QM, u0, N, rng,x_charm):
    d = mu.shape[0]
    if R == 0:
        # plain MC
        Y  = rng.standard_normal((d, N))
        Lw = np.ones(N)
        d2 = compute_distances_parallel(
            Y,
            mu, Lchol, Q_costs, L_costs, F_costs, loss_threshold,x_charm
        )**2
        return Y, Lw, d2

    if not (0 <= u0 < R**2):
        raise ValueError("Need 0 ≤ u0 < R²")
    rate = R - np.sqrt(u0)             # > 0

    # y1 ~ rate + Exp(rate); LR = phi(y1)/f(y1)
    y1 = rate + rng.exponential(scale=1.0/rate, size=N)
    Lw = np.exp(-0.5*y1**2 + rate*(y1 - rate)) / (rate * np.sqrt(2*np.pi))

    Y  = np.vstack([y1, rng.standard_normal((d-1, N))])
    Y  = QM @ Y                         # rotate in y-space
    X  = mu.reshape(-1,1) + Lchol @ Y   # map to x-space
 
    d2 = compute_distances_parallel(Y, mu, Lchol, Q_costs, L_costs, F_costs, loss_threshold,x_charm)**2
    return X, Lw, d2
