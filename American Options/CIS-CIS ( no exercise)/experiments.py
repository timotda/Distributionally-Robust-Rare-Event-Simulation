# --- inside add_to_experiments.py ---
import math
from get_worst_p_no_exercise_cis import get_worst_p_no_exercise_at_all_put
from get_worst_p_no_robust import get_worst_p_no_robust
import numpy as np

def lsre_from_plug_in(p_hat, u_hat, var_H, var_P, alpha):
    num = (var_H / (alpha * (u_hat**2))) + (var_P / (1.0 - alpha))
    return np.sqrt(num) / p_hat

def make_put_boundary_grid(K, T, K_grid,sigma):

    dt   = T / K_grid
    taus = T - np.arange(1, K_grid + 1) * dt
    s    = K - sigma*np.sqrt(taus)               # there should be a beta before the sigma but I keep it simple
    return np.log(s)



def run_no_exercise_at_all_block():
    S0, K_strike = 154.51, 165.0
    r, q, sigma, T = 0.03, 0.0, 1, 1.0
    K_grid = 252    # use 252 for daily
    delta  = 0.02
    alpha  = 0.5
    N_tot  = int(1e3)
    rng    = np.random.default_rng(12345)

    b_grid = make_put_boundary_grid(K_strike, T, K_grid, sigma)

    u_bar, p, err, var_H, var_p = get_worst_p_no_exercise_at_all_put(
        delta, N_tot, alpha, S0, K_strike, r, q, sigma, T, b_grid, rng
    )
    lsre = lsre_from_plug_in(p, u_bar, var_H, var_p, alpha)
    print(f'alpha = {np.sqrt(var_H)/(np.sqrt(var_H)+ u_bar*np.sqrt(var_p))}')


   

    '''
    runs = 1
    u_bar, _, _, var_H_, var_p_ = get_worst_p_no_exercise_at_all_put(
        delta, N_alpha, alpha, S0, K_strike, r, q, sigma, T, b_grid, rng
    )
    alpha = np.sqrt(var_H_)/(np.sqrt(var_H_)+ u_bar*np.sqrt(var_p_))
    breakpoint()
    print('alpha = ', alpha)
    ### estimate p ###

    p_list = []
    t_list = []
    lsre_list = []
    for i in range(runs):
        start = time.perf_counter()
        u_bar, p, err, var_H, var_p = get_worst_p_no_exercise_at_all_put(
        delta, N_tot, alpha, S0, K_strike, r, q, sigma, T, b_grid, rng, u_opt=1.1*u_bar
    )
        elapsed = time.perf_counter() - start
        p_list.append(p)
        t_list.append(elapsed)
        lsre = lsre_from_plug_in(p, u_bar, var_H, var_p, alpha)
        lsre_list.append(lsre)  

    p_arr = np.array(p_list)
    mean_p = p_arr.mean()
    # standard error of the mean
    se = p_arr.std(ddof=1) / np.sqrt(runs)
    # 95% CI half‐width
    ci_half = 1.96 * se

    t_arr = np.array(t_list)
    mean_time = t_arr.mean()

    lsre_arr = np.array(lsre_list)                 ### NEW
    lsre_mean = lsre_arr.mean()                    ### NEW
    lsre_median = np.median(lsre_arr) 
    '''

    print(f"[NO-EXERCISE-AT-ALL put] u_bar={u_bar:.4g}, p={p:.3e} ± {err:.1e}, lsre={lsre:.1e}")

run_no_exercise_at_all_block()