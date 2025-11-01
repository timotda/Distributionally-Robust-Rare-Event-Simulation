import numpy as np
from get_distance_piecewise import compute_distances_parallel
#Second Stage estimation
# return the worst case proba and the error

def robust_mc_piecewise(u_bar,loss_threshold,w,N,A,B,a,b,rng):
    

    d = A.shape[1]
    Z = rng.standard_normal((d, N))
    d2 = compute_distances_parallel(Z, A, B, a, b,w,loss_threshold)**2

    hits = (d2 <= u_bar)

    #  MC estimate and 95% error
    p = np.mean(hits)
    
    err = 1.96 * np.std(hits) / np.sqrt(N)

    var_p = np.var(hits, ddof=1)
    
    return p, err, var_p