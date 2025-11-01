import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from get_worst_p_delta_cis_cis import get_worst_p_delta_cis_cis

def lsre_from_plug_in(p_hat, u_hat, var_H, var_P, alpha):
    num = (var_H / (alpha * (u_hat**2))) + (var_P / (1.0 - alpha))
    return np.sqrt(num) / p_hat


def calculate_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma



def compute_average_worst_prob(delta, loss_threshold, mu, Sigma, N_alpha,N_tot, runs, rng,S0,gamma,linear_costs,quad_costs,fixed_costs):
    """
    Repeatedly computes the worst-case probability with a two-stage MC-MC estimator,
    then returns the mean over `runs` and the 95% CI half-width on that mean.
    """
    p_list = []
    t_list = []
    lsre_list = [] 
    ### find optimal allocation ### 
    u_bar,_,_,var_H_,var_p_= get_worst_p_delta_cis_cis(delta,loss_threshold,mu,Sigma,1/2,N_alpha,rng,S0,gamma,linear_costs,quad_costs,fixed_costs)
    alpha = np.sqrt(var_H_)/(np.sqrt(var_H_)+ u_bar*np.sqrt(var_p_))
    ###------------------------###
    print('alpha = ', alpha)
    ### estimate p ###
    for i in range(runs):
        start = time.perf_counter()
        u_bar_,p_hat, _, var_H,var_p= get_worst_p_delta_cis_cis(delta,
                                            loss_threshold,
                                            mu,
                                            Sigma,
                                            alpha, ### change sample allocation
                                            N_tot-N_alpha,
                                            rng,
                                            S0,
                                            gamma,
                                            linear_costs,
                                            quad_costs,
                                            fixed_costs,
                                            u_opt=1.1*u_bar) #### change u_opt if can't find different fmin and fmax
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

    return mean_p, ci_half, mean_time,var_H,var_p,u_bar_,lsre_mean, lsre_median, lsre_arr


def run_threshold_sweep(
    thresholds, delta, mu, Sigma, w, N_alpha, N_tot, runs, rng,A,B,a,b, out_csv="results.csv"
):
    results = []

    for loss_threshold in thresholds:
        print(f"\n=== loss_threshold = {loss_threshold} ===")
        # run your existing experiment
        mean_p, ci_half, mean_time, var_H, var_p, u_bar, lsre_mean, lsre_median, lsre_arr = \
            compute_average_worst_prob(
                delta,
                loss_threshold,
                mu,
                Sigma,
                w,
                N_alpha,
                N_tot,
                runs,
                rng, A, B,a,b
            )

        # relative error
        rel_error = ci_half / mean_p

        results.append({
            "loss_threshold":    loss_threshold,
            "mean_p":            mean_p,
            "rel_error":         rel_error,
            "lsre_mean":         lsre_mean,
            "lsre_median":       lsre_median,
            "mean_time_sec":     mean_time
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

    stock_price = np.array([437.11, 117.93, 224.31, 183.13, 476.79, 177.66, 434.47, 239.20, 209.78, 565.33])
    strike_price = np.array([435, 117, 220, 180, 480, 175, 440, 235, 210, 560])
    option_price = np.array([50.90, 19.71, 28.37, 28.35, 61.81, 24.85, 22.55, 57.10, 14.78, 64.45])
    is_call = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    # Order of stocks: MSFT, NVDA, AAPL, AMZN, META, GOOGL, BRK.B, TSLA, JPM, UNH
    stock_tickers = ['MSFT', 'NVDA', 'AAPL', 'AMZN', 'META', 'GOOGL', 'BRK.B', 'TSLA', 'JPM', 'UNH']


    fixed_costs = 1


    linear_costs = np.array([
        0.015,  # MSFT
        0.050,  # NVDA
        0.015,  # AAPL
        0.015,  # AMZN
        0.065,  # META
        0.015,  # GOOGL
        0.070,  # BRK.B
        0.050,  # TSLA
        0.065,  # JPM
        0.065   # UNH
    ])


    quadratic_costs = np.array([
        2.25e-06,  # MSFT
        3.75e-07,  # NVDA
        5.00e-07,  # AAPL
        5.50e-07,  # AMZN
        3.33e-06,  # META
        8.00e-07,  # GOOGL
        1.12e-05,  # BRK.B
        3.33e-07,  # TSLA
        2.20e-06,  # JPM
        2.00e-05   # UNH
    ])

    # reduce to first n assets
    n = 5
    mu    = mu[:n]
    Rho   = Rho[:n, :n]
    Std   = Std[:n]
    stock_price  = stock_price[:n]
    strike_price = strike_price[:n]
    option_price = option_price[:n]
    is_call      = is_call[:n]
    quad_costs = quadratic_costs[:n]
    linear_costs = linear_costs[:n]
    gamma = calculate_gamma(stock_price, strike_price, 0.5, 0.05, Std)

    Sigma = np.diag(Std) @ Rho @ np.diag(Std)

    # === parameters ===
    delta = 0.02              # ambiguity-set radius
    loss_threshold = 0     #  loss-threshold
    N_tot = int(1e5)          # number of MC samples per run
    N_alpha = int(10000)      # number of samples used to find the optimal allocation
    runs = 1             # number of independent repetitions
    # ==============================
 

    mean_p, ci_half, mean_time ,var_H,var_p, u_bar,lsre_mean, lsre_median, lsre_arr = compute_average_worst_prob(
        delta,
        loss_threshold,
        mu,
        Sigma,
        N_alpha,
        N_tot,
        runs,
        rng,
        stock_price,
        gamma,
        linear_costs,
        quad_costs,
        fixed_costs
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

    
    '''
    loss_thresholds = [1, 2] #, 4, 6, 10]

    run_threshold_sweep(
        thresholds=loss_thresholds,
        delta=delta,
        mu=mu,
        Sigma=Sigma,
        w=w,
        N_alpha=N_alpha,
        N_tot=N_tot,
        runs=runs,
        rng=rng,A=A,B=B,a=a,b=b,
        out_csv="CIS_piecewise_results.csv"
    )
    '''

