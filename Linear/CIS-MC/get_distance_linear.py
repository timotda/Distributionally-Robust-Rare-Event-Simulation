import numpy as np

def get_distance_linear(X, w, loss_threshold):
    proj = w.T @ X                               # shape (n,)
    raw  = proj + loss_threshold                 # how far “outside” the half-space
    d    = np.maximum(raw, 0) / np.linalg.norm(w)
    return d




