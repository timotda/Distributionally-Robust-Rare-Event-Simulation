import numpy as np
from scipy.linalg      import cholesky
from get_radius_cis_piecewise import get_radius_cis
from robust_cis_piecewise    import robust_cis_piecewise
from find_xstar import find_x_star
from shift_samples import shift_samples



def get_worst_p_piecewise_cis_cis(delta, loss_threshold, mu, Sigma, w, alpha,N,rng,A,B,a,b,u_opt=None):
    N1 = int(alpha*N); N2 = int((1-alpha)*N)
    Lchol = cholesky(Sigma, lower=True)       
    A_tilde  = A @ Lchol
    a_tilde  = A @ mu.reshape(-1) + a.reshape(-1)
    B_tilde  = B @ Lchol
    b_tilde  = B @ mu.reshape(-1) + b.reshape(-1)
   
    
    # Stage 1: try CIS + root‐find
    #### FIND x_star ####
    x_star,R,QM= find_x_star(A_tilde,B_tilde,a_tilde,b_tilde,w,loss_threshold)
    breakpoint()
    if u_opt is not None:
        u0 = u_opt
    else :
        u0 = 0.9*R**2 
    ### Shift sample ###
    X,L1,d2 = shift_samples(A_tilde,B_tilde,a_tilde,b_tilde,loss_threshold,w,R,QM,u0,N1,rng)
    ### SOLVE h(u) = delta^2
    u_bar = get_radius_cis(d2, L1, delta**2)
    ### resample shifted ###
    X,L1,d2 = shift_samples(A_tilde,B_tilde,a_tilde,b_tilde,loss_threshold,w,R,QM,u_bar,N2,rng)
    ### compute worst case proba ###
    p_hat, err, var_p , var_H= robust_cis_piecewise(u_bar,d2,L1)

    return u_bar,p_hat, err, var_H,var_p

