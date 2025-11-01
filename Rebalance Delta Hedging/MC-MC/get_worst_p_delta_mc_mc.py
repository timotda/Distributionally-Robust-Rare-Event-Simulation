import numpy as np
from scipy.linalg import cholesky
from get_radius_mc_delta import get_radius_mc_delta
from robust_mc_delta import robust_mc_delta
from scipy.stats import norm


def get_worst_p_delta_hedged_mc_mc(delta, loss_threshold, mu, Sigma,alpha, N,rng,S0,gamma,linear_costs,quad_costs,fixed_costs,x_charm):
    N1 = int(alpha*N); N2 = int((1-alpha)*N)
    Lchol = cholesky(Sigma, lower=True)   
    Q_costs = 1/2 *np.diag(S0) @ np.diag(gamma) @ np.diag(quad_costs) @ np.diag(gamma) @ np.diag(S0) 
    L_costs = np.diag(linear_costs) @ np.diag(gamma) @ np.diag(S0)
    F_costs = fixed_costs

    u_bar = get_radius_mc_delta(delta**2,loss_threshold,mu,Lchol,Q_costs,L_costs,F_costs,N1,rng,x_charm)
    p,err,var_p, var_H= robust_mc_delta(u_bar,loss_threshold,mu,Lchol,Q_costs,L_costs,F_costs,N2,rng,x_charm)

    return u_bar,p,err,var_H,var_p


