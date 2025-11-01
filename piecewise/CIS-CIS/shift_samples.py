import numpy as np 
from get_distance_piecewise import compute_distances_parallel

def shift_samples(A,B,a,b,loss_threshold,w,R,QM,u0,N,rng):
    d = A.shape[1]

    if R == 0:
        # plain MC: X~N(0,I), L=1 
        X = rng.standard_normal((d, N))
        L = np.ones(N)
       
        d2 = compute_distances_parallel(X, A, B, a, b,w,loss_threshold)**2
        
        return X, L, d2

    if not (0 <= u0 < R**2):
        raise ValueError(f"Need 0 ≤ u0 < R²; got u0={u0}, R²={R**2}")
    s0 = np.sqrt(u0)
    shift = R - s0
    Z = rng.exponential(scale=1/shift, size=N)
    y1 = shift + Z

    # likelihood ratio
    L1 = np.exp(-0.5*y1**2 + shift*(y1 - shift)) / (shift * np.sqrt(2*np.pi))

    # remaining Gaussians
    Y = np.vstack([y1, rng.standard_normal((d-1, N))])

    # rotate back
    X = QM @ Y
    
    #  distances
    d2 = compute_distances_parallel(X, A, B, a, b,w,loss_threshold)**2
   
    return X,L1,d2