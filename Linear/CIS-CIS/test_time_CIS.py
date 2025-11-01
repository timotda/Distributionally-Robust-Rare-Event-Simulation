#!/usr/bin/env python3
"""
run_cis_timing.py
Time each block of the CIS algorithm and print a breakdown.
"""

import time
from collections import defaultdict
import numpy as np
from scipy.linalg import cholesky

# --- your project imports ---
from get_radius_cis_linear import get_radius_cis
from robust_cis_linear import robust_cis_linear
from find_xstar import find_x_star
from shifted_sample import shift_samples



# ---------- Timed wrapper ----------
def timed_get_worst_p_linear_cis_cis(delta, loss_threshold, mu, Sigma, w, alpha, N, rng):
    """Run CIS and record wall-clock time of each logical block."""
    t = defaultdict(float)
    t0 = time.perf_counter()

    N1 = int(alpha * N)
    N2 = int((1 - alpha) * N)

    # --- Precompute pieces ---
    tic = time.perf_counter()
    Lc = cholesky(Sigma, lower=True)
    A = (-w @ Lc).reshape(1, -1)
    b0 = float(loss_threshold + w.dot(mu))
    u0 = 0.5 * (b0 / np.linalg.norm(A)) ** 2
    t["precompute"] += time.perf_counter() - tic

    # --- Stage 1 ---
    tic = time.perf_counter()
    x_star, R, QM = find_x_star(A, b0)
    t["find_x_star"] += time.perf_counter() - tic

    tic = time.perf_counter()
    X, L1, d2 = shift_samples(A, b0, R, QM, u0, N1, rng)
    t["shift_samples_stage1"] += time.perf_counter() - tic

    tic = time.perf_counter()
    u_bar, var_H = get_radius_cis(d2, L1, delta ** 2)
    t["get_radius_cis"] += time.perf_counter() - tic

    # --- Stage 2 ---
    tic = time.perf_counter()
    X, L1, d2 = shift_samples(A, b0, R, QM, u0, N2, rng)
    t["shift_samples_stage2"] += time.perf_counter() - tic

    tic = time.perf_counter()
    p_hat, err, var_p = robust_cis_linear(u_bar, d2, L1)
    t["robust_cis_linear"] += time.perf_counter() - tic

    t["total"] = time.perf_counter() - t0
    return u_bar, p_hat, err, var_H, var_p, dict(t)


def bench(runs, **kwargs):
    """Run multiple times and return averages (mean, std)."""
    times = defaultdict(list)
    last_outputs = None
    for _ in range(runs):
        outputs = timed_get_worst_p_linear_cis_cis(**kwargs)
        *vals, tdict = outputs
        last_outputs = vals
        for k, v in tdict.items():
            times[k].append(v)

    stats = {k: (float(np.mean(v)), float(np.std(v))) for k, v in times.items()}
    return last_outputs, stats


def pretty_print_stats(stats):
    keys_sorted = sorted(stats.items(), key=lambda kv: kv[1][0], reverse=True)
    print("\nTiming breakdown (mean ± std) over runs:\n")
    for k, (m, s) in keys_sorted:
        print(f"{k:22s} {m*1e3:10.2f} ± {s*1e3:7.2f} ms")


def main():
    # === Seed / RNG ===
    rng = np.random.default_rng(42)

    # === Input data ===
    mu = np.array([
        0.227468298, 0.96887651,  0.200208297, 0.06242237,  0.154425293,
        0.156871276, 0.120151978, -0.017124922, 0.10038554,  0.09309727
    ])

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

    Std = np.array([
        0.843835167, 2.037446523, 0.947836462, 1.188395074, 1.612105777,
        0.938271533, 0.672178053, 2.156438679, 0.944739056, 0.675976581
    ])

    # reduce to first n assets
    n = 5
    mu  = mu[:n]
    Rho = Rho[:n, :n]
    Std = Std[:n]

    # covariance matrix Σ = diag(Std) @ Rho @ diag(Std)
    Sigma = np.diag(Std) @ Rho @ np.diag(Std)

    # equal-weight portfolio on these n assets
    w = np.ones(n) / n

    # === parameters ===
    delta = 0.02          # ambiguity-set radius
    loss_threshold = 2    # loss-threshold
    N_tot = int(1e7)      # number of MC samples per run
    alpha = 0.5           # split between stages
    runs = 3              # repetitions

    print("Running CIS timing with:")
    print(f"  N_tot = {N_tot:,}")
    print(f"  runs  = {runs}")

    (u_bar, p_hat, err, var_H, var_p), stats = bench(
        runs=runs,
        delta=delta,
        loss_threshold=loss_threshold,
        mu=mu,
        Sigma=Sigma,
        w=w,
        alpha=alpha,
        N=N_tot,
        rng=rng
    )

    pretty_print_stats(stats)

    print("\nOutputs (last run):")
    print(f"u_bar   = {u_bar}")
    print(f"p_hat   = {p_hat}")
    print(f"err     = {err}")
    print(f"var_H   = {var_H}")
    print(f"var_p   = {var_p}")


if __name__ == "__main__":
    main()
