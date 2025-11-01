# get_radius_mc_AO.py
import numpy as np
from scipy.optimize import brentq
from get_distance_AO import compute_distances_AO_parallel

def get_radius_mc_AO(delta2, N, G, h, rng):
    d = G.shape[1]
    Z = rng.standard_normal((d, N))
    d2 = compute_distances_AO_parallel(Z, G, h)**2

    idx       = np.argsort(d2)
    d2_sorted = d2[idx]
    cum_w     = np.cumsum(d2_sorted) / N            # \hat h(u)

    def hhat(u):
        k = np.searchsorted(d2_sorted, u, side='right')
        return cum_w[k-1] if k else 0.0

    umin, umax = 0.0, float(np.max(d2))
    fmin = hhat(umin) - delta2
    fmax = hhat(umax) - delta2
    if fmin * fmax > 0:
        raise ValueError("No root for hhat(u)=delta^2 on [0, max d2].")

    return brentq(lambda u: hhat(u) - delta2, umin, umax)
