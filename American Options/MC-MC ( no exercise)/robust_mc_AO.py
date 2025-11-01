# robust_mc_AO.py
import numpy as np
from get_distance_AO import compute_distances_AO_parallel

def robust_mc_AO(u_bar, N, G, h, rng):
    d = G.shape[1]
    Z = rng.standard_normal((d, N))                 # nominal driver ~ N(0, I_d)
    d2 = compute_distances_AO_parallel(Z, G, h)**2
    hits = (d2 <= u_bar)

    p   = np.mean(hits)
    err = 1.96 * np.std(hits, ddof=1) / np.sqrt(N)  # 95% CI half-width
    var_p = np.var(hits, ddof=1)
    var_H = np.var(d2 * (d2 <= u_bar), ddof=1)      # same bookkeeping you use
    return p, err, var_p, var_H
