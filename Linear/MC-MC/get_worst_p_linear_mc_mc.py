import numpy as np
from get_radius_mc_linear import get_radius_mc_linear
from robust_mc_linear import robust_mc_linear
from scipy.linalg import cholesky

# compute the two stage estimation to find the estimated worst case probability 
def get_worst_p_linear_mc_mc(delta, loss_threshold, mu, Sigma, w,alpha, N,rng):
    N1 = int(alpha*N); N2 = int((1-alpha)*N)
    Lc = cholesky(Sigma, lower=True)
    A    = (-w @ Lc).reshape(1,-1)
    b0   = float(loss_threshold + w.dot(mu))
    u_bar = get_radius_mc_linear(delta**2,A, b0, N1, rng)
    p, err, var_p, var_H = robust_mc_linear(u_bar,A,b0, N2,rng)
    return u_bar,p,err,var_H,var_p

