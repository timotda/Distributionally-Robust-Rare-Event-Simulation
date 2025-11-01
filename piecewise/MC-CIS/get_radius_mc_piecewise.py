import numpy as np
from scipy.optimize import brentq
from get_distance_piecewise import compute_distances_parallel

def get_radius_mc_piecewise(delta2,loss_threshold,w,A,B,a,b,N,rng):
    d= A.shape[1]
    Z = rng.standard_normal((d, N))
    d2 = compute_distances_parallel(Z, A, B, a, b,w,loss_threshold)**2

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

    if fmin * fmax > 0:
        raise ValueError(f"No root in [{umin}, {umax}] (hhat−δ² has same sign at endpoints).")

    # Solve via Brent’s method
    u = brentq(lambda u: hhat(u) - delta2, umin, umax)
    var_H = np.var(d2 * (d2 <= u), ddof=1)
      
    return u, var_H