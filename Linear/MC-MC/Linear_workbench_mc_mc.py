import numpy as np
from VaR_mc_mc import get_VaR_linear_mc_mc, bootstrap_VaR_linear_mc_mc
from get_worst_p_linear_mc_mc import get_worst_p_linear_mc_mc

# S&P 500 stocks (in this order): MSFT, NVDA, AAPL, AMZN, META, GOOGL, BRK.B, TSLA, JPM, UNH
# Data: returns from June 2021 to June 2024
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

# reduce dimension to first n assets
n = 5
mu    = mu[:n]
Rho   = Rho[:n, :n]
Std   = Std[:n]

# covariance matrix Σ = diag(Std) @ Rho @ diag(Std)
Sigma = np.diag(Std) @ Rho @ np.diag(Std)

# (optional) equal‐weight portfolio on these n assets
w = np.ones(n) / n
N = 1e7  # number of samples for Monte Carlo simulation
delta = 0.02
betas = [0.05, 0.01, 0.001, 0.0001]  # VaR at confidence levels: 95%, 99%, 99.9%
# quick sanity check
print("mu:", mu)
print("Sigma:\n", Sigma)
print("weights:", w)



for beta in betas:
    
    var_est = get_VaR_linear_mc_mc(
        beta=beta,
        delta=delta,  # non-zero ambiguity
        mu=mu,
        Sigma=Sigma,
        w=w,
        N=int(N),
        rng=rng
    )
    print(f"VaR at beta={beta}: {var_est:.4f}")

    # bootstrap the VaR estimate
    var_est, err = bootstrap_VaR_linear_mc_mc(
        beta=beta,
        delta=delta,  # non-zero ambiguity
        mu=mu,
        Sigma=Sigma,
        w=w,
        N=int(N),
        s=2,
        rng=rng
    )
    print(f"Bootstrap VaR: {var_est:.4f} ± {err:.4f}")
