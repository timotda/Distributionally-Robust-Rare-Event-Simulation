#!/usr/bin/env python3
import time
import numpy as np

from get_distance_linear import get_distance_linear

# ---- original function, timed version ----
def shift_samples_timed(A, b, R, QM, u0, N, rng):
    """
    Same outputs as shift_samples plus a dict with timing (in seconds)
    for each numbered block in the original code.
    """
    times = {}
    tic = time.perf_counter

    t0 = tic()
    d = A.shape[1]
    b_q = np.atleast_1d(b).astype(float)
    times['prep'] = tic() - t0

    if R == 0:
        # 1) plain MC branch
        t1 = tic()
        X = rng.standard_normal((d, N))
        L = np.ones(N)
        times['rand_plain'] = tic() - t1

        t2 = tic()
        w = -A.flatten()
        dist = get_distance_linear(X, w, b_q[0])
        d2 = dist**2
        times['distance'] = tic() - t2

        return X, L, d2, times

    # 2) check u0 < R²
    t2 = tic()
    if not (0 <= u0 < R**2):
        raise ValueError(f"Need 0 ≤ u0 < R²; got u0={u0}, R²={R**2}")
    s0 = np.sqrt(u0)
    shift = R - s0
    times['check_params'] = tic() - t2

    # 3–4) sample Z and y1
    t3 = tic()
    Z = rng.exponential(scale=1/shift, size=N)
    y1 = shift + Z
    times['sample_y1'] = tic() - t3

    # 5) likelihood ratio
    t4 = tic()
    L1 = np.exp(-0.5*y1**2 + shift*(y1 - shift)) / (shift * np.sqrt(2*np.pi))
    times['likelihood'] = tic() - t4

    # 6) remaining Gaussians / build Y
    t5 = tic()
    Y = np.vstack([y1, rng.standard_normal((d-1, N))])
    times['build_Y'] = tic() - t5

    # 7) rotate back
    t6 = tic()
    X = QM @ Y
    times['rotate'] = tic() - t6

    # 8) distances
    t7 = tic()
    w = -A.flatten()
    dist = get_distance_linear(X, w, b_q[0])
    d2 = dist**2
    times['distance'] = tic() - t7

    return X, L1, d2, times


# ---- helper to run several times and print ----
def run_benchmark(runs, **kwargs):
    block_names = None
    acc = {}
    for i in range(runs):
        X, L, d2, tdict = shift_samples_timed(**kwargs)
        if block_names is None:
            block_names = list(tdict.keys())
            acc = {k: 0.0 for k in block_names}
        for k in block_names:
            acc[k] += tdict[k]

    print(f"\nTimings over {runs} runs (seconds):")
    for k in block_names:
        print(f"  {k:>14s}: {acc[k] / runs:.6f}")


# ---- build inputs from your data and call ----
if __name__ == "__main__":
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
    w_port = np.ones(n) / n

    # === parameters ===
    delta = 0.02          # ambiguity-set radius (not directly used here)
    loss_threshold = 5    # used as 'b' below (example)
    N_tot = int(1e7)      # number of MC samples per run
    alpha = 0.5           # split between stages (use for u0 fraction)
    runs = 3              # repetitions

    # --- Map to shift_samples inputs (adjust if your theory differs) ---
    d = n
    A = w_port.reshape(1, -1)        # simple half-space normal
    b = loss_threshold               # scalar threshold
    R = 1.0                          # pick an R > 0 to trigger IS branch
    u0 = alpha * R**2                # just an example consistent with 0<=u0<R^2

    # QM: any orthonormal matrix. For realism, use eigenvectors of Sigma.
    # QR of a random Gaussian is fine too.
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    QM = Q

    kwargs = dict(A=A, b=b, R=R, QM=QM, u0=u0, N=N_tot, rng=rng)

    run_benchmark(runs, **kwargs)
