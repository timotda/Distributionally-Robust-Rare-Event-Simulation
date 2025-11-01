import numpy as np
from scipy.optimize import brentq

def get_radius_cis(d2, L, delta2):
    N = len(d2)
    idx        = np.argsort(d2)          # ascending
    d2_sorted  = d2[idx]
    w_sorted   = d2_sorted * L[idx]      # pre-multiply once
    cum_w      = np.cumsum(w_sorted) / N # cumulative ĥ(u) values
   
    def hhat(u):
        #O(log N1) lookup instead of O(N1) scan
        k = np.searchsorted(d2_sorted, u, side='right')
        return cum_w[k-1] if k else 0.0

    
    umin, umax = 0.0, float(np.max(d2))

    fmin = hhat(umin) - delta2
    fmax = hhat(umax) - delta2
    
    if fmin * fmax > 0:
        raise ValueError(f"No root in [{umin}, {umax}] (hhat−δ² has same sign at endpoints).")

    # Solve via Brent’s method
    u = brentq(lambda u: hhat(u) - delta2, umin, umax)
    var_H = np.var(d2 * (d2 <= u) * L, ddof=1)
    return u, var_H 
