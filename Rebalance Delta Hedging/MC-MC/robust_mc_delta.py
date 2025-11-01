import numpy as np
from get_distance_delta_hedge import compute_distances_parallel


def robust_mc_delta(u_bar,loss_threshold,mu,Lchol,Q_costs,L_costs,F_costs,N,rng,x_charm):
    

    d = Q_costs.shape[1]
    Z = Lchol@ rng.standard_normal((d, N)) + mu.reshape(-1,1)
    d2 = compute_distances_parallel(Z,Q_costs,L_costs,F_costs,loss_threshold,x_charm)**2

    hits = (d2 <= u_bar)

    #  MC estimate and 95% error
    p = np.mean(hits)
    
    err = 1.96 * np.std(hits) / np.sqrt(N)
    
    var_p = np.var(hits, ddof=1)
    var_H = np.var(d2 * (d2 <= u_bar), ddof=1)
    return p, err, var_p, var_H