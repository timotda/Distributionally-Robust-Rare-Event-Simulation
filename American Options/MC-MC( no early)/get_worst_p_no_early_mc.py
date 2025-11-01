
import numpy as np
from scipy.linalg import cholesky
from get_radius_mc_AO import get_radius_mc_AO
from robust_mc_AO import robust_mc_AO


def _build_H_sigma_grid(sigma: float, T: float, K_grid: int):
    dt   = T / K_grid
    step = sigma * np.sqrt(dt)
    H = np.tril(np.ones((K_grid, K_grid))) * step   # Y = c + H z
    return H, dt

def build_constraints_no_exercise_at_all_put(
    S0: float, K_strike: float, r: float, q: float, sigma: float, T: float,
    b_grid: np.ndarray  # length K_grid: b_k = log s*(T - t_k)
):
    K_grid = len(b_grid)
    H, dt = _build_H_sigma_grid(sigma, T, K_grid)
    k = np.arange(1, K_grid + 1)
    c = np.log(S0) + (r - q - 0.5 * sigma * sigma) * k * dt  # drift part

    # Half-spaces: (H z)_k >= b_k - c_k  for all k
    G = H.copy()
    h = b_grid - c

    # --- NEW: maturity OTM for the put ---
    # Need Y_K >= log K  as well as Y_K >= b_K  -> combine as max(...)
    h_T = np.log(K_strike) - c[-1]
    h[-1] = max(h[-1], h_T)   # one constraint at k=K is enough

    return G, h

def build_constraints_no_early_exercise_put(S0, K_strike, r, q, sigma, T, b_grid):
    K_grid = len(b_grid)
    # Y = c + H z, with lower-triangular H
    dt   = T / K_grid
    step = sigma * np.sqrt(dt)
    H    = np.tril(np.ones((K_grid, K_grid))) * step
    k    = np.arange(1, K_grid + 1)
    c    = np.log(S0) + (r - q - 0.5 * sigma * sigma) * k * dt

    # Use only k = 1..K-1
    G = H[:-1, :]
    h = b_grid[:-1] - c[:-1]

    # sanity: event must not include the origin
    if np.max(h) <= 0:
        raise ValueError("Degenerate event: all pre-maturity constraints satisfied at the origin (R=0). "
                         "Raise the near-maturity boundary or adjust (S0, T, r, q, σ).")
    return G, h

def build_constraints_no_early_exercise_call(
    S0: float, K_strike: float, r: float, q: float, sigma: float, T: float,
    b_grid: np.ndarray  # b_k = log s*(T - t_k), length K_grid
):
    K_grid = len(b_grid)
    dt   = T / K_grid
    step = sigma * np.sqrt(dt)
    H    = np.tril(np.ones((K_grid, K_grid))) * step
    k    = np.arange(1, K_grid + 1)
    mu   = (r - q - 0.5 * sigma * sigma)
    c    = np.log(S0) + mu * k * dt

    # Use only 1..K-1 (exclude maturity)
    G = -H[:-1, :]
    h = c[:-1] - b_grid[:-1]

    # (Optional sanity) if all rows are ≤0 you’re likely non-rare:
    # if np.max(h) <= 0: print("Warning: likely non-rare event for call.")

    return G, h


def get_worst_p_no_early(
    delta,              # Wasserstein radius (not squared)         # mean/cov of the Gaussian driver X (d=K)
    S0, K_strike, r, q, sigma, T, # GBM params to form constraints
    b_grid,             # array of size K: b_k = log s*(T - t_k)
    alpha, N, rng
):

    # Build constraints in Z-coordinates (standard normal)
    G, h = build_constraints_no_early_exercise_put(S0, K_strike, r, q, sigma, T, b_grid)

    N1 = int(alpha * N)
    N2 = int(N - N1)

    u_bar = get_radius_mc_AO(delta**2, N1, G, h, rng)
    p, err, var_p, var_H = robust_mc_AO(u_bar, N2, G, h, rng)
    return u_bar, p, err, var_H, var_p
