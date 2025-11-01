import numpy as np
from scipy.optimize import brentq
from get_distance_linear import get_distance_linear

def get_radius_mc_linear(delta2,A,b0, N, rng):
    """
    Estimate u_bar via naive MC and Brent's method for root finding.
    """
    # Draw nominal samples
    
    d = A.shape[1]
    Z = rng.standard_normal((d, N))
    w = -A.flatten()
    d2 = get_distance_linear(Z, w, b0)**2
    ##### modifying with this lower the time a lot : d2_sorted  = np.sort(d2)
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
    return u





  
    