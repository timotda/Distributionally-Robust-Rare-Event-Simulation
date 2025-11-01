import numpy as np

def robust_cis_piecewise(u_bar,d2,L):
    Y  = (d2 <= u_bar) * L            
    p  = np.mean(Y)
    err = 1.96 * np.std(Y) / np.sqrt(len(Y))
    var_p = np.var(Y, ddof=1)
    return p, err,var_p
