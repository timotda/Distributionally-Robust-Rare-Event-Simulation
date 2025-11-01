# --- inside get_worst_p_no_early.py ---

import numpy as np
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

def get_worst_p_no_exercise_at_all_put(
    delta: float,            # Wasserstein radius
    N_total: int, alpha: float,
    S0: float, K_strike: float, r: float, q: float, sigma: float, T: float,
    b_grid: np.ndarray,
    rng
):
    # Build polyhedron under Z ~ N(0, I)
    G, h = build_constraints_no_exercise_at_all_put(S0, K_strike, r, q, sigma, T, b_grid)

    N1 = int(alpha * N_total)
    N2 = N_total - N1

    u_bar = get_radius_mc_AO(delta**2, N1, G, h, rng)
    p, err, var_p, var_H = robust_mc_AO(u_bar, N2, G, h, rng)
    return u_bar, p, err, var_H, var_p
