# --- inside add_to_experiments.py ---

from get_worst_p_no_exercise_mc import get_worst_p_no_exercise_at_all_put
import numpy as np

def lsre_from_plug_in(p_hat, u_hat, var_H, var_P, alpha):
    num = (var_H / (alpha * (u_hat**2))) + (var_P / (1.0 - alpha))
    return np.sqrt(num) / p_hat

def constant_boundary_log_sstar(K_grid: int, log_s_star: float):
    return np.full(K_grid, float(log_s_star)) 

def run_no_exercise_at_all_block():
    # Match your §5.2 style parameters
    S0, K_strike = 90.0, 100.0
    r, q, sigma, T = 0.04, 0.00, 0.20, 1.0
    K_grid = 12    # use 252 for daily
    delta  = 0.02
    alpha  = 0.5
    N_tot  = 1e5
    rng    = np.random.default_rng(12345)

    # Boundary b_k = log s*(T - t_k). For a quick start use a constant proxy:
    s_star =  140
    b_grid = constant_boundary_log_sstar(K_grid, np.log(s_star))

    u_bar, p, err, var_H, var_p = get_worst_p_no_exercise_at_all_put(
        delta, N_tot, alpha, S0, K_strike, r, q, sigma, T, b_grid, rng
    )
    lsre = lsre_from_plug_in(p, u_bar, var_H, var_p, alpha)
    print(f"[NO-EXERCISE-AT-ALL put] u_bar={u_bar:.4g}, p={p:.3e} ± {err:.1e}, lsre={lsre:.1e}")
    
run_no_exercise_at_all_block()