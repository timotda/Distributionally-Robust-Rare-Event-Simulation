import numpy as np
from get_distance_linear import get_distance_linear
#Second Stage estimation
# return the worst case proba and the error

def robust_mc_linear(u_bar,A,b0,N,rng):
    

    d = A.shape[1]
    Z = rng.standard_normal((d, N))
    w = -A.flatten()
    d2 = get_distance_linear(Z, w, b0)**2

    hits = (d2 <= u_bar)
    
    #  MC estimate and 95% error
    p = np.mean(hits)
    err = 1.96 * np.std(hits) / np.sqrt(N)
    var_p = np.var(hits, ddof=1)
    return p, err , var_p
