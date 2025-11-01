import time
import numpy as np
import pandas as pd
from get_worst_p_quadratic_cis_cis import get_worst_p_quadratic_cis_cis

def lsre_from_plug_in(p_hat, u_hat, var_H, var_P, alpha):
    num = (var_H / (alpha * (u_hat**2))) + (var_P / (1.0 - alpha))
    return np.sqrt(num) / p_hat

def compute_average_worst_prob(delta, loss_threshold, mu, Sigma, N_alpha,N_tot, runs, rng,Q,r,s):
    """
    Repeatedly computes the worst-case probability with a two-stage MC-MC estimator,
    then returns the mean over `runs` and the 95% CI half-width on that mean.
    """
    p_list = []
    t_list = []
    lsre_list = [] 
    ### find optimal allocation ### 
    u_bar,_,_,var_H_,var_p_,_= get_worst_p_quadratic_cis_cis(delta,loss_threshold,mu,Sigma,1/2,N_alpha,rng,Q,r,s)
    alpha = np.sqrt(var_H_)/(np.sqrt(var_H_)+ u_bar*np.sqrt(var_p_))
    ###------------------------###
    print('alpha = ', alpha)
    if alpha == 1.0:
        alpha = 1/2
    if np.isnan(alpha):
        alpha = 1/2
    ### estimate p ###
    for i in range(runs):
        start = time.perf_counter()
        u_bar_,p_hat, _, var_H,var_p, R= get_worst_p_quadratic_cis_cis(delta,
                                            loss_threshold,
                                            mu,
                                            Sigma,
                                            alpha, ### change sample allocation
                                            N_tot-N_alpha,
                                            rng,
                                            Q,r,s, 
                                            u_opt=u_bar) #### change u_opt if can't find different fmin and fmax
        elapsed = time.perf_counter() - start
        p_list.append(p_hat)
        t_list.append(elapsed)
        lsre = lsre_from_plug_in(p_hat, u_bar_, var_H, var_p, alpha)   ### NEW
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

    return mean_p, ci_half, mean_time,var_H,var_p,u_bar_,lsre_mean, lsre_median, lsre_arr,R

def run_threshold_sweep(
    thresholds, delta, mu, Sigma, N_alpha, N_tot, runs, rng,Q,r,s, out_csv="results.csv"
):
    results = []

    for loss_threshold in thresholds:
        print(f"\n=== loss_threshold = {loss_threshold} ===")
        # run your existing experiment
        mean_p, ci_half, mean_time, var_H, var_p, u_bar, lsre_mean, lsre_median, lsre_arr,R = \
            compute_average_worst_prob(
                delta,
                loss_threshold,
                mu,
                Sigma,
                N_alpha,
                N_tot,
                runs,
                rng,
                Q,
                r,
                s
            )

        # relative error
        rel_error = ci_half / mean_p


        results.append({
            "loss_threshold":    loss_threshold,
            "mean_p":            mean_p,
            "rel_error":         rel_error,
            "lsre_mean":         lsre_mean,
            "lsre_median":       lsre_median,
            "mean_time_sec":     mean_time,
            "Distance": R**2
        })

    # build a DataFrame and write to CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote sweep results to {out_csv}")

if __name__ == "__main__":
    # === Data provided ===
    rng = np.random.default_rng(42)

    # mean returns
    mu = np.array([
        0.227468298, 0.96887651,  0.200208297, 0.06242237,  0.154425293,
        0.156871276, 0.120151978, -0.017124922, 0.10038554,  0.09309727
    ])

    # return correlation matrix
    Rho = np.array([
        [ 1.         ,  0.725136788,  0.68100574 ,  0.63393554 ,  0.539078858,
          0.690132247,  0.512103785,  0.437472519,  0.325859205,  0.408232001],
        [ 0.725136788,  1.         ,  0.630841989,  0.733369794,  0.547701009,
          0.660094118,  0.478342761,  0.539899973,  0.429105754, -0.003619947],
        [ 0.68100574 ,  0.630841989,  1.         ,  0.66411717 ,  0.306714616,
          0.585097461,  0.476639998,  0.596988328,  0.311675141,  0.307342874],
        [ 0.63393554 ,  0.733369794,  0.66411717 ,  1.         ,  0.439927261,
          0.671072765,  0.41412774 ,  0.64553384 ,  0.243637825,  0.023564451],
        [ 0.539078858,  0.547701009,  0.306714616,  0.439927261,  1.         ,
          0.385859118,  0.236928706,  0.256335071,  0.209418264, -0.184499541],
        [ 0.690132247,  0.660094118,  0.585097461,  0.671072765,  0.385859118,
          1.         ,  0.393339284,  0.466339975,  0.319079797,  0.25705194 ],
        [ 0.512103785,  0.478342761,  0.476639998,  0.41412774 ,  0.236928706,
          0.393339284,  1.         ,  0.193225855,  0.602548381,  0.368659465],
        [ 0.437472519,  0.539899973,  0.596988328,  0.64553384 ,  0.256335071,
          0.466339975,  0.193225855,  1.         ,  0.137074353,  0.1556824  ],
        [ 0.325859205,  0.429105754,  0.311675141,  0.243637825,  0.209418264,
          0.319079797,  0.602548381,  0.137074353,  1.         ,  0.193546677],
        [ 0.408232001, -0.003619947,  0.307342874,  0.023564451, -0.184499541,
          0.25705194 ,  0.368659465,  0.1556824  ,  0.193546677,  1.        ]
    ])

    # volatilities
    Std = np.array([
        0.843835167, 2.037446523, 0.947836462, 1.188395074, 1.612105777,
        0.938271533, 0.672178053, 2.156438679, 0.944739056, 0.675976581
    ])

    # reduce to first n assets
    n = 5
    mu    = mu[:n]
    Rho   = Rho[:n, :n]
    Std   = Std[:n]

    # covariance matrix Σ = diag(Std) @ Rho @ diag(Std)
    Sigma = np.diag(Std) @ Rho @ np.diag(Std)

    
    
    # === parameters ===
    delta = 0.01              # ambiguity‐set radius
    loss_threshold = 2   #  loss‐threshold
    N_tot = int(1e5)          # number of MC samples per run
    N_alpha = int(10000)      # number of samples used to find the optimal allocation
    runs = 4             # number of independent repetitions
    # ==============================
    
    stock_price = np.array([437.11, 117.93, 224.31, 183.13, 476.79, 177.66, 434.47, 239.20, 209.78, 565.33])
    strike_price = np.array([435, 117, 220, 180, 480, 175, 440, 235, 210, 560])
    option_price = np.array([50.90, 19.71, 28.37, 28.35, 61.81, 24.85, 22.55, 57.10, 14.78, 64.45])
    is_call = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool)

    r = np.zeros(5, dtype=float)
    # Constant (theta over one trading day)
    s = -2.1200075039407476   # = Theta_total * (1/252), Theta_total ≈ -534.24189099 per year 

    '''
    Stock	Gamma (
    Gamma)
    Stock 1	0.000456
    Stock 2	0.001648
    Stock 3	0.001968
    Stock 4	0.001923
    Stock 5	0.000494

    '''
    Q = np.array([
        [87.23, 0., 0., 0., 0.],
        [0., 22.92, 0., 0., 0.],
        [0., 0., 98.90, 0., 0.],
        [0., 0., 0., 64.49, 0.],
        [0., 0., 0., 0., 112.02]
    ])

   
    loss_thresholds = [0.5,1,1.5,2] 

    run_threshold_sweep(
        thresholds=loss_thresholds,
        delta=delta,
        mu=mu,
        Sigma=Sigma,
        N_alpha=N_alpha,
        N_tot=N_tot,
        runs=runs,
        rng=rng,Q=Q,r=r,s=s,
        out_csv="CIS_quadratic_results.csv"
    )
    '''
    mean_p, ci_half, mean_time ,var_H,var_p, u_bar,lsre_mean, lsre_median, lsre_arr = compute_average_worst_prob(
    delta,
    loss_threshold,
    mu,
    Sigma,
    N_alpha,
    N_tot,
    runs,
    rng,
    Q,r,s
)

    rel_error = ci_half / mean_p

    print(f"Over {runs} runs:")
    print(f"Average worst-case probability = {mean_p:.6e}")
    print(f"Relative 95% CI half-width   = {rel_error:.3%}" )
    print(f"Mean time per run             = {mean_time:.3f} seconds")
    print(f"u_bar = {u_bar}")
    print(f"variance of H(u_bar) = {var_H}")
    print(f"variance of p(u_bar) = {var_p}")
    print(f"optimal sample allocation = {np.sqrt(var_H)/(np.sqrt(var_H)+ u_bar*np.sqrt(var_p))}")

    print(f"lsre_mean = {lsre_mean}")
    print(f"lsre_median = {lsre_median}")

    rel_single, rel_mean = rel_hw_from_lsre(lsre_arr, N_tot - N_alpha, runs)
    print(f"95% relative half-width (single run) — mean over runs: {rel_single.mean()*100:.4f}%")
    print(f"95% relative half-width (mean of {runs} runs): {rel_mean.mean()*100:.4f}%")
 '''