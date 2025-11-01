import numpy as np 
from scipy.linalg import qr
from get_distance_delta_hedge import solve_distance

def find_x_star(mu,Lchol,Q_costs,L_costs,F_costs,loss_threshold):
    d = Q_costs.shape[0]
    # solve at y0 = 0 (i.e., distance from origin in y-space)
    y_star, R = solve_distance(
        np.zeros(d), mu, Lchol, Q_costs, L_costs, F_costs, loss_threshold,
        return_y_star=True
    )
    e = y_star / R                    
    M = np.eye(d); M[:, 0] = e
    QM, _ = qr(M, mode="full")        
    if QM[:, 0] @ e < 0:
        QM[:, 0] *= -1
    return y_star, R, QM              
