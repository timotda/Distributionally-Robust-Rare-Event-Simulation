import numpy as np
from scipy.optimize import brentq
from get_distance_delta_hedge import compute_distances_parallel

def get_radius_mc_delta(delta2,loss_threshold,mu,Lchol,Q_costs,L_costs,F_costs,N,rng,x_charm):
    d= Q_costs.shape[1]
    Z = Lchol@ rng.standard_normal((d, N)) + mu.reshape(-1,1)
    d2 = compute_distances_parallel(Z,Q_costs,L_costs,F_costs,loss_threshold,x_charm)**2

    idx        = np.argsort(d2)          # ascending
    d2_sorted  = d2[idx]
    cum_w      = np.cumsum(d2_sorted) / N # cumulative ĥ(u) values
    
    def hhat(u):
        # ❷ O(log N1) lookup instead of O(N1) scan
        k = np.searchsorted(d2_sorted, u, side='right')
        return cum_w[k-1] if k else 0.0

    
    umin, umax = 0.0, float(np.max(d2))

    fmin = hhat(umin) - delta2
    fmax = hhat(umax) - delta2
    breakpoint()
    if fmin * fmax > 0:
        raise ValueError(f"No root in [{umin}, {umax}] (hhat−δ² has same sign at endpoints).")

    # Solve via Brent’s method
    u = brentq(lambda u: hhat(u) - delta2, umin, umax)
     
    return u