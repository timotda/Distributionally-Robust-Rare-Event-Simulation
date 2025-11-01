import numpy as np
from scipy.optimize import root_scalar
from get_worst_p_linear_mc_mc import get_worst_p_linear_mc_mc
from tqdm import tqdm

def get_VaR_linear_mc_mc(beta, delta, mu, Sigma, w, N,rng):
    """
    Solve for VaR u such that
      sup_{||ν−μ||≤δ} P_ν(-wᵀX ≥ u) = β,
    using a dynamic bracket for root finding.
    """

    # Define f(u) = p̂(u) - β
    def f(u):
        _, p_hat, _ ,_,_ = get_worst_p_linear_mc_mc(delta, u, mu, Sigma, w,1/2, N,rng)
        return p_hat - beta

    # 1) f(0) should be ≈ 1 - β > 0
    u_low = 0.0
    f_low = f(u_low)
    if f_low < 0:
        # If even at u=0 we are below β, there's no VaR solution
        raise ValueError(f"f(0)={f_low:.6f} < 0: β too large")

    # 2) Find u_high until f(u_high) < 0
    u_high = 1.0
    f_high = f(u_high)
    BIG    = 1e8
    while f_high > 0:
        u_high *= 2
        f_high = f(u_high)
        if u_high > BIG:
            raise ValueError(
                f"Unable to bracket root for VaR: f({u_high})={f_high:.6f}"
            )
    # 3) Now f(u_low)>0>f(u_high). Bisect to find the root.
    sol = root_scalar(f, bracket=[u_low, u_high], method='bisect')
    return sol.root

def bootstrap_VaR_linear_mc_mc(beta, delta, mu, Sigma, w, N, rng,s=30):
    """Bootstrap the above estimator to get a 95%-CI half-width."""
    var_estimates = [
        get_VaR_linear_mc_mc(beta, delta, mu, Sigma, w, N,rng)
        for _ in tqdm(range(s), desc="Bootstrapping VaR")
    ]
    err = 1.96 * np.std(var_estimates, ddof=1)
    return np.mean(var_estimates), err
