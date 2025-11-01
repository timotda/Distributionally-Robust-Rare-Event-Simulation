import numpy as np
from scipy.linalg      import cholesky
from get_radius_cis_quadratic import get_radius_cis_quadratic
from robust_cis_quadratic    import robust_cis_quadratic
from find_xstar import find_x_star
from shift_samples import shift_samples



def get_worst_p_quadratic_cis_cis(delta, loss_threshold, mu, Sigma,alpha, N,rng,Q,r,s,u_opt=None):
    N1 = int(alpha*N); N2 = int((1-alpha)*N)
    Lchol = cholesky(Sigma, lower=True)       
    Q_tilde = Lchol.T @ Q @ Lchol                    
    r_tilde = 2 * (mu @ Q @ Lchol) + (r @ Lchol)
    s_tilde = mu @ Q @ mu + (r @ mu) + s
    # Stage 1: try CIS + root‐find
    #### FIND x_star ####
    x_star,R,QM= find_x_star(Q_tilde,r_tilde,s_tilde,loss_threshold)

    if u_opt is not None:
        u0 = u_opt
        if u_opt > R**2:
            u0 = R**2 *0.99
    else :
        u0 = R**2 / 2
    ### Shift sample ###
    X,L1,d2 = shift_samples(Q_tilde,r_tilde,s_tilde,loss_threshold,R,QM,u0,N1,rng)
    ### SOLVE h(u) = delta^2
    u_bar = get_radius_cis_quadratic(d2, L1, delta**2)
    ### resample shifted ###
    X,L1,d2 = shift_samples(Q_tilde,r_tilde,s_tilde,loss_threshold,R,QM,u_bar,N2,rng)
    ### compute worst case proba ###
    p_hat, err, var_p , var_H= robust_cis_quadratic(u_bar,d2,L1)
    return u_bar,p_hat, err, var_H,var_p,R
