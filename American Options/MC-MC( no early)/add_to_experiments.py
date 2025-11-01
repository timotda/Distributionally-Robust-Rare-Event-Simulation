# --- inside add_to_experiments.py ---
import math
from get_worst_p_no_early_mc import get_worst_p_no_early
import numpy as np

def lsre_from_plug_in(p_hat, u_hat, var_H, var_P, alpha):
    num = (var_H / (alpha * (u_hat**2))) + (var_P / (1.0 - alpha))
    return np.sqrt(num) / p_hat

def make_put_boundary_grid(K, s_T, T, K_grid, gamma=0.6, eps=1e-6):
    """
    s_T = s*(T): immediate-exercise threshold at t=0 (e.g., ~79.5 for your params).
    Returns b_grid with b_k = log s*(T - t_k) for k=1..K_grid (in log space).
    gamma controls how fast the boundary rises toward K as tau ↓ 0.
    """
    dt   = T / K_grid
    taus = T - np.arange(1, K_grid + 1) * dt           # tau_k = T - t_k
    s    = K - (K - s_T) * (taus / T)**gamma           # ↑ toward K as tau→0
    s    = np.minimum(s, K - eps)                      # avoid exactly K at maturity
    return np.log(s)
def make_put_boundary_grid(K, s_T, T, K_grid, gamma=0.6, eps=1e-6):
    dt   = T / K_grid
    taus = T - np.arange(1, K_grid + 1) * dt
    s    = K - (K - s_T) * (taus / T)**gamma
    s[-1] = min(s[-1], K - eps)  # cap just the last point
    return np.log(s)


def run_no_exercise_at_all_block():
    # Match your §5.2 style parameters
    S0, K_strike = 90.0, 100.0
    r, q, sigma, T = 0.04, 0.0, 0.5, 1.0
    K_grid = 52    # use 252 for daily
    delta  = 0.02
    alpha  = 0.5
    N_tot  = 1e5
    N_alpha = 10000
    rng    = np.random.default_rng(12345)

    # Boundary b_k = log s*(T - t_k). For a quick start use a constant proxy:
    s_star = 89
    b_grid = make_put_boundary_grid(K_strike, s_star, T, K_grid, gamma=1.6)
    u_bar, p, err, var_H, var_p = get_worst_p_no_early(
        delta, S0, K_strike, r, q, sigma, T, b_grid, alpha, N_tot, rng
    )
    lsre = lsre_from_plug_in(p, u_bar, var_H, var_p, alpha)
    print(f'alpha = {np.sqrt(var_H)/(np.sqrt(var_H)+ u_bar*np.sqrt(var_p))}')
    
    print(f"[NO-EXERCISE-AT-ALL put] u_bar={u_bar:.4g}, p={p:.3e} ± {err:.1e}, lsre={lsre:.1e}")

run_no_exercise_at_all_block()