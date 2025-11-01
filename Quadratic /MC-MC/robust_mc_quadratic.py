import numpy as np
from get_distance_quadratic import compute_distances_parallel


def robust_mc_quadratic(u_bar,loss_threshold,Q,r,s,N,rng):
    

    d = Q.shape[1]
    Z = rng.standard_normal((d, N))
    d2 = compute_distances_parallel(Z,Q,r,s,loss_threshold)**2

    hits = (d2 <= u_bar)

    #  MC estimate and 95% error
    p = np.mean(hits)
    
    err = 1.96 * np.std(hits) / np.sqrt(N)
    
    var_p = np.var(hits, ddof=1)
    var_H = np.var(d2 * (d2 <= u_bar), ddof=1)
    return p, err, var_p, var_H