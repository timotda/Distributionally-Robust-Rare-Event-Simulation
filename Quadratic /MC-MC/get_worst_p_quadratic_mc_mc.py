import numpy as np
from scipy.linalg import cholesky
from get_radius_mc_quadratic import get_radius_mc_quadratic
from robust_mc_quadratic import robust_mc_quadratic
from scipy.stats import norm


def get_worst_p_quadratic_mc_mc(delta, loss_threshold, mu, Sigma,alpha, N,rng,Q,r,s):
    N1 = int(alpha*N); N2 = int((1-alpha)*N)
    Lchol = cholesky(Sigma, lower=True)       
    Q_tilde = Lchol.T @ Q @ Lchol                    
    r_tilde = 2 * (mu @ Q @ Lchol) + (r @ Lchol)
    s_tilde = mu @ Q @ mu + (r @ mu) + s 

    u_bar = get_radius_mc_quadratic(delta**2,loss_threshold,Q_tilde,r_tilde,s_tilde,N1,rng)
    p,err,var_p, var_H= robust_mc_quadratic(u_bar,loss_threshold,Q_tilde,r_tilde,s_tilde,N2,rng)

    return u_bar,p,err,var_H,var_p

