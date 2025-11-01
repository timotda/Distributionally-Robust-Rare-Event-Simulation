import numpy as np
from scipy.linalg      import cholesky
from get_radius_cis_delta import get_radius_cis_delta
from robust_cis_delta    import robust_cis_delta
from find_xstar import find_x_star
from shift_samples import shift_samples



def get_worst_p_delta_cis_cis(delta, loss_threshold, mu, Sigma,alpha, N,rng,S0,gamma,linear_costs,quad_costs,fixed_costs,u_opt=None):
    N1 = int(alpha*N); N2 = int((1-alpha)*N)
    Lchol = cholesky(Sigma, lower=True)       
    Q_costs = 1/2 *np.diag(S0) @ np.diag(gamma) @ np.diag(quad_costs) @ np.diag(gamma) @ np.diag(S0) 
    L_costs = np.diag(linear_costs) @ np.diag(gamma) @ np.diag(S0)
    F_costs = fixed_costs
    breakpoint()
    # Stage 1: try CIS + root‐find
    #### FIND x_star ####
    x_star,R,QM= find_x_star(mu,Lchol,Q_costs,L_costs,F_costs,loss_threshold)

    if u_opt is not None:
        u0 = u_opt
    else :
        u0 = R**2 / 2
    ### Shift sample ###
    X,L1,d2 = shift_samples(mu,Lchol,Q_costs,L_costs,F_costs,loss_threshold,R,QM,u0,N1,rng)
    ### SOLVE h(u) = delta^2
    u_bar = get_radius_cis_delta(d2, L1, delta**2)
    ### resample shifted ###
    X,L1,d2 = shift_samples(mu,Lchol,Q_costs,L_costs,F_costs,loss_threshold,R,QM,u_bar,N2,rng)
    ### compute worst case proba ###
    p_hat, err, var_p , var_H= robust_cis_delta(u_bar,d2,L1)

    return u_bar,p_hat, err, var_H,var_p
