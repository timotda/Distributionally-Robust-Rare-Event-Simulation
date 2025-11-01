import numpy as np
from get_radius_mc_linear import get_radius_mc_linear
from robust_cis_linear import robust_cis_linear
from shifted_sample import shift_samples
from find_xstar import find_x_star
from scipy.linalg import cholesky

# compute the two stage estimation to find the estimated worst case probability 
def get_worst_p_linear_mc_cis(delta, loss_threshold, mu, Sigma, w,alpha, N,rng, u_opt =None):
    N1 = int(alpha*N); N2 = int((1-alpha)*N)
    Lc = cholesky(Sigma, lower=True)
    A    = (-w @ Lc).reshape(1,-1)
    b0   = float(loss_threshold + w.dot(mu))
    # Stage 1: compute the radius u_bar
    u_bar ,var_H= get_radius_mc_linear(delta**2,A, b0, N1, rng)
    # Stage 2: compute the worst-case probability using cis
    x_star, R, QM = find_x_star(A,b0) 
    ### USE THE U_bar from the first estimation with 10000 samples ###
    if u_opt is not None:
        u0 = u_opt
    else:
        u0 = R**2/2
    ### resample shifted ###
    X,L1,d2 = shift_samples(A,b0,R,QM,u_bar,N2,rng)
    ### compute worst case proba ###
    p_hat, err, var_p = robust_cis_linear(u_bar,d2,L1)
    return u_bar,p_hat, err, var_H,var_p
