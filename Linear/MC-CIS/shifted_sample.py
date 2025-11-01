import numpy as np 
from get_distance_linear import get_distance_linear

def shift_samples(A,b,R,QM,u0,N,rng):
    d = A.shape[1]
    
    if R == 0:
        # plain MC: X~N(0,I), L=1, distances via get_distance_linear
        X = rng.standard_normal((d, N))
        L = np.ones(N)
        # region A x≥b  ⇒  -wᵀx≥v with w=-A.flatten(), v=b_q[0]
        w = -A.flatten()
        dist = get_distance_linear(X, w, b)
        return X, L, dist**2

 
    if not (0 <= u0 < R**2):
        raise ValueError(f"Need 0 ≤ u0 < R²; got u0={u0}, R²={R**2}")
    s0 = np.sqrt(u0)
    shift = R - s0
    Z = rng.exponential(scale=1/shift, size=N)
    y1 = shift + Z

    # likelihood ratio
    L1 = np.exp(-0.5*y1**2 + shift*(y1 - shift)) / (shift * np.sqrt(2*np.pi))

    #  remaining Gaussians
    Y = np.vstack([y1, rng.standard_normal((d-1, N))])

    # rotate back
    X = QM @ Y
    
    # distances
    w = -A.flatten()
    dist = get_distance_linear(X, w, b)
    d2 = dist**2

    return X,L1,d2