import numpy as np 
from scipy.linalg import qr
from get_distance_AO import solve_distance_AO

def find_x_star(G,h):
 
    d = G.shape[1]

    R,x_star = solve_distance_AO(np.zeros(d),G,h,return_x_star=True)
    # build basis so e=x*/R is first column

    e = x_star / R
    M = np.eye(d); M[:,0] = e
    QM, _ = qr(M, mode='full')
    if QM[:,0].dot(e) < 0:
        QM[:,0] *= -1




    slack = G @ x_star - h
    tol   = 1e-8 * max(1.0, np.linalg.norm(G@x_star), np.linalg.norm(h))
    active = np.where(slack <= tol)[0]
    rank_active = np.linalg.matrix_rank(G[active, :])
    print("active size:", len(active), "rank:", rank_active)
    return x_star,R, QM