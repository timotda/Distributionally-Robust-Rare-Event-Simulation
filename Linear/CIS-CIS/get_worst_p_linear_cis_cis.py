import numpy as np
from scipy.linalg      import cholesky
from get_radius_cis_linear import get_radius_cis
from robust_cis_linear    import robust_cis_linear
from find_xstar import find_x_star
from shifted_sample import shift_samples



def get_worst_p_linear_cis_cis(delta, loss_threshold, mu, Sigma, w, alpha,N,rng,u_opt=None):
    N1 = int(alpha*N); N2 = int((1-alpha)*N)
    Lc   = cholesky(Sigma, lower=True)
    A    = (-w @ Lc).reshape(1,-1)
    b0   = float(loss_threshold + w.dot(mu))
  
    # Stage 1: try CIS + root‐find
    #### FIND x_star ####
    x_star, R, QM = find_x_star(A,b0)  

    ### USE THE U_bar from the first estimation with 10000 samples ###
    if u_opt is not None:
        u0 = u_opt
    else:
        u0 = R**2/2 
   
    ### Shift sample ###
    X,L1,d2 = shift_samples(A,b0,R,QM,u0,N1,rng)
    ### SOLVE h(u) = delta^2
    u_bar = get_radius_cis(d2, L1, delta**2)
    ### resample shifted ###
    X,L1,d2 = shift_samples(A,b0,R,QM,u_bar,N2,rng)
    ### compute worst case proba ###
    p_hat, err, var_p , var_H= robust_cis_linear(u_bar,d2,L1)
    return u_bar,p_hat, err, var_H,var_p
