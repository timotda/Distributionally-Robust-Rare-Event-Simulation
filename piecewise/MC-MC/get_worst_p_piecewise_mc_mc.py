import numpy as np
from scipy.linalg import cholesky
from get_radius_mc_piecewise import get_radius_mc_piecewise
from robust_mc_piecewise import robust_mc_piecewise
from scipy.stats import norm


def get_worst_p_piecewise_mc_mc(delta, loss_threshold, mu, Sigma, w,alpha, N,rng,A,B,a,b):
    N1 = int(alpha*N); N2 = int((1-alpha)*N)
    Lchol = cholesky(Sigma, lower=True)       
    A_tilde  = A @ Lchol
    a_tilde  = A @ mu.reshape(-1) + a.reshape(-1)
    B_tilde  = B @ Lchol
    b_tilde  = B @ mu.reshape(-1) + b.reshape(-1)
    u_bar = get_radius_mc_piecewise(delta**2,loss_threshold,w,A_tilde,B_tilde,a_tilde,b_tilde,N1,rng)
    p,err,var_p, var_H= robust_mc_piecewise(u_bar,loss_threshold,w,N2,A_tilde,B_tilde,a_tilde,b_tilde,rng)

    return u_bar,p,err,var_H,var_p


