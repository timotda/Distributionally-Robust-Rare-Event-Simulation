import numpy as np

def robust_cis_delta(u_bar,d2,L):
    Y  = (d2 <= u_bar) * L            
    p  = np.mean(Y)
    err = 1.96 * np.std(Y) / np.sqrt(len(Y))
    var_p = np.var(Y, ddof=1)
    var_H = np.var(d2 * (d2 <= u_bar) * L, ddof=1)
    return p, err,var_p,var_H