import numpy as np 
import quadprog
from scipy.linalg import qr

def find_x_star(A,b):
 
    d = A.shape[1]
    b_q = np.atleast_1d(b).astype(float)

    # 1) solve min½‖x‖²  s.t.  A x ≥ b
    G, a, C = np.eye(d), np.zeros(d), A.T
    x_star = quadprog.solve_qp(G, a, C, b_q)[0]
    R = np.linalg.norm(x_star)
    if R == 0:
        QM = np.eye(d)
        return x_star, R, QM
    # 2) build basis so e=x*/R is first column
    e = x_star / R
    M = np.eye(d); M[:,0] = e
    QM, _ = qr(M, mode='full')
    if QM[:,0].dot(e) < 0:
        QM[:,0] *= -1
    return x_star,R, QM