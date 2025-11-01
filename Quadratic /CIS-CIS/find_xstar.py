import numpy as np 
from scipy.linalg import qr
from get_distance_quadratic import solve_distance

def find_x_star(Q,r,s,loss_threshold):
 
    d = Q.shape[0]

    x_star,R = solve_distance(np.array([0]),Q,r,s,loss_threshold,return_x_star=True)
    # build basis so e=x*/R is first column
    e = x_star / R
    M = np.eye(d); M[:,0] = e
    QM, _ = qr(M, mode='full')
    if QM[:,0].dot(e) < 0:
        QM[:,0] *= -1
    return x_star,R, QM
