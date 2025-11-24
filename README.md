# Distributionally Robust Rare-Event Simulation in Finance

This repository contains the code from my summer research project on **distributionally robust rare-event simulation** for financial risk.

The goal is to estimate **tail losses** and **Value-at-Risk (VaR)** under **model uncertainty** for:

- Linear and nonlinear portfolios  
- Portfolios of European options (piecewise-linear payoffs)  
- Quadratic portfolios  
- American options  
- Delta-hedging strategies with transaction costs  

using **Wasserstein ball** ambiguity sets and a **two-stage conditional importance sampling (CIS)** algorithm.

---

## TL;DR – What this repo does

- Implements a **distributionally robust tail probability / VaR estimator** based on Wasserstein balls around a nominal Gaussian model.
- Uses a **rare-event geometry + distance formulation** to turn a worst-case probability problem into a probability of a “robustified” event.
- Uses **CIS variance reduction** in a **two-stage scheme**:
  1. Estimate a dual radius parameter $\( \bar{u} \)$ from data.
  2. Plug $\(\bar{u}\)$ into a CIS estimator of the worst-case tail probability.
- Applies this to:
  - Linear portfolios of assets,
  - Portfolios with piecewise-linear option payoffs,
  - Quadratic loss functions,
  - American options (exercise region as a polyhedron),
  - Delta-hedging strategies with quadratic + linear transaction costs.

---

## 1. Background (short)

In risk management we care about events like

> “Portfolio loss exceeds a large threshold.”

These events are **rare**, so:

- Naive Monte Carlo needs **huge sample sizes** to see enough losses.
- The **model is uncertain**: the “true” distribution may differ from the estimated one.

This project models uncertainty by a **Wasserstein ball** around a nominal distribution:

- The set of plausible models is  
  \[
  \mathcal{P} = $\{ \mathbb{Q} : W_2(\mathbb{Q}, \mathbb{P}_0) \le \delta \}$
  \]
  where $\(W_2\)$ is the 2-Wasserstein distance and $\(\mathbb{P}_0\)$ is a Gaussian reference model.
- We are interested in the **worst-case tail probability** over that ball.

Using duality, the worst-case tail probability can be rewritten as an event of the form $ \{d(X, E)^2 \le \bar{u}\} $

under the nominal model, where $\(E\)$ is a rare-event set (loss ≥ threshold) and $\(d(\cdot, E)\)$ is the distance to $\(E\)$.  
So everything boils down to:

1. **Distance computations** for the chosen financial model.
2. **Efficient estimation** of  
   $\[
   \mathbb{P}_0\big(d(X, E)^2 \le \bar{u}\big)
   \]$
   using CIS.

---

## 2. Repository structure

### 2.1 Core CIS / robust estimation code

These modules implement the generic CIS-based robust rare-event estimator, independent of the specific financial payoff.

- **`find_xstar.py`**  
  Solves the quadratic program
  $\[
  \min_x \frac12\|x\|^2 \quad \text{s.t. } A x \ge b
  \]$
  to find:
  - `x_star`: closest point in the rare-event set to the origin,
  - `R`: its norm (distance),
  - `QM`: an orthogonal matrix whose first column is aligned with `x_star / R`.  
  This “most likely failure direction” is used to orient the CIS sampler.

- **`get_distance_linear.py`**  
  Distance to a **linear loss event**:
  - Inputs: samples `X` (shape `(d, n)`), weights `w`, loss threshold.
  - Computes the distance from each point to the half-space defined by the linear loss constraint.

- **`shifted_sample.py`**  
  Implements **Conditional Importance Sampling** for Gaussian inputs:
  - If `R == 0`, falls back to standard Monte Carlo `X ~ N(0, I)`.
  - Otherwise:
    - Samples along the critical direction using a shifted distribution controlled by `u0`,
    - Samples remaining coordinates as standard normals,
    - Rotates back via `QM`,
    - Computes distances and likelihood ratios.
  - Returns: shifted samples `X`, likelihood ratios `L`, and squared distances `d2`.

- **`get_radius_cis_linear.py`**  
  Given squared distances `d2` and likelihood ratios `L`, solves for  
 $ \[
  u: \quad \hat{h}(u) = \frac{1}{N}\sum d_i^2 \mathbf{1}\{d_i^2 \le u\} L_i = \delta^2.
  \]$
  Uses Brent’s method to find the root, giving a simulation-based estimate of $\( \bar{u} \)$.

- **`robust_cis_linear.py`**  
  Core robust estimator once \( \bar{u} \) is known:
  - Computes  
    $\[
    \hat{p} = \mathbb{E}\big[ \mathbf{1}\{d^2 \le \bar{u}\} L \big]
    \]$
    as a Monte Carlo average.
  - Returns the estimate `p`, a 95% CI half-width `err`, and sample variances `var_p`, `var_H`.

- **`get_worst_p_linear_cis_cis.py`**  
  Driver for the **linear Gaussian portfolio** case with CIS in both stages:
  - Builds the covariance factor `Lc` from `Sigma`.
  - Constructs `(A, b)` for the linear loss event.
  - Calls `find_xstar` to get `x_star`, `R`, `QM`.
  - **Stage 1**: CIS sampling + `get_radius_cis` to estimate $\( \bar{u} \)$.
  - **Stage 2**: CIS sampling at $\(u = \bar{u}\)$ + `robust_cis_linear` to estimate worst-case tail probability and associated variances.

There are analogous functions (not shown in this snippet) for other models:

- `get_worst_p_piecewise_cis_cis.py` – piecewise-linear portfolio  
- `get_worst_p_quadratic_cis_cis.py` – quadratic loss  
- `get_worst_p_delta_cis_cis.py` – delta-hedging cost  
- `get_worst_p_no_exercise_cis.py` – special case for American put “no-exercise-at-all” scenario  

They all follow the same **two-stage CIS pattern** but use different distance functions.

---

### 2.2 Distance modules for each model

These are the parts that change most between experiments: they define the rare-event set \(E\) via a distance function.

#### Linear portfolio

- **`get_distance_linear.py`**  
  Already described above; it’s a simple closed-form distance to a half-space.

#### Piecewise-linear portfolio (options)

- **`get_distance_piecewise.py`**  
  Distance to a **piecewise-affine** loss region of the form
  $\[
  \{x : w^\top \max(Ax + a,\ Bx + b) \le -v\},
  \]$
  which corresponds to a portfolio built from linear pieces (e.g. longs/shorts in calls and puts).

  - `solve_distance(...)` solves
   $$ \[
    \min_x \tfrac12 \|x - x_0\|^2
    \quad \text{s.t. } y \ge A x + a,\ y \ge B x + b,\ w^\top y \le -v
    \]$$
    using CVXPY, and returns the Euclidean distance.
  - `compute_distances_parallel(...)` applies that solver to many points in parallel using `joblib`.
  - Additional helper `piecewise_active_report(...)` (in `experiments_cis_piecewise.py`) inspects which affine branch is active at the closest point, for geometric diagnostics.

#### Quadratic portfolio

- **`get_distance_quadratic.py`**  
  Distance to a **quadratic loss region** of the form
  $$\[
  \{x : x^\top Q x + r^\top x + s \le -v\}.
  \]$$
  - Uses CVXPY to minimize \( \frac12 \|x - x_0\|^2 \) subject to the quadratic inequality.
  - `compute_distances_parallel(...)` runs this over columns of a sample matrix in parallel.

#### American options

- **`get_distance_AO.py`**  
  Distance to a **polyhedral exercise region** \(E = \{y : G y \ge h\}\), which arises when discretizing the early-exercise boundary of an American option on a grid.
  - `solve_distance_AO(...)` solves
    $$\[
    \min_y \tfrac12 \|y - x_0\|^2 \quad \text{s.t. } G y \ge h
    \]$$
    with CVXPY (using OSQP).
  - `compute_distances_parallel(...)` applies this to many samples.

#### Delta-hedging with transaction costs

- **`get_distance_delta_hedge.py`**  
  Distance in **factor space** $\(y\)$ to a set of states where a delta-hedging strategy’s total cost is below a loss threshold:
  - Model:$ \(x = \mu + L_{\text{chol}} y\) $is the underlying state (e.g. returns).
  - Cost is modeled as a combination of:
    - A **quadratic term** (matrix `Q_costs`),
    - An **L1 term** (`L_costs`), capturing proportional transaction costs,
    - A constant term `F_costs`.
  - `solve_distance(...)` solves
    $$\[
    \min_y \tfrac12\|y - y_0\|^2
    \quad \text{s.t. } 
    (x - x_\text{charm})^\top Q_\text{costs} (x - x_\text{charm})
      + \|L_\text{costs}^\top (x - x_\text{charm})\|_1
      + F_\text{costs} \le v,
    \]$$
    with $\(x = \mu + L_{\text{chol}} y\)$.
  - Again, `compute_distances_parallel(...)` vectorizes this over many samples.

---

### 2.3 Experiment scripts

These are the entry points you actually run. They all follow the same pattern:

1. Set up model parameters (`mu`, `Sigma`, payoff parameters).
2. Choose Wasserstein radius `delta`, loss thresholds, and sample sizes.
3. Call a `get_worst_p_*_cis_cis` driver repeatedly.
4. Aggregate results across runs and report mean worst-case probability, CI, and large-sample relative error (LSRE).

#### Linear portfolio – CIS vs Monte Carlo

- **`experiments_cis_linear.py`**  
  - Uses real-looking data for `mu`, correlation matrix `Rho`, and volatilities `Std` for 10 assets (optionally truncated to fewer assets).
  - Constructs covariance `Sigma` and portfolio weights `w`.
  - Runs the **CIS–CIS** two-stage method via `get_worst_p_linear_cis_cis`.
  - Reports:
    - Mean worst-case tail probability,
    - 95% CI half-width and relative error,
    - Average runtime per run,
    - Estimated $\( \bar{u} \)$, variance terms `var_H`, `var_p`,
    - Plug-in **large-sample relative error** via `lsre_from_plug_in`.

- **`experiments_mc_linear.py`**  
  - Same setup as `experiments_cis_linear.py` but calls `get_worst_p_linear_mc_mc` (a two-stage estimator using plain Monte Carlo in both stages).
  - Used as a **baseline** to compare CIS vs MC in terms of variance reduction and runtime.

#### Piecewise-linear portfolio (options)

- **`experiments_cis_piecewise.py`**  
  - Builds a portfolio of **European options** from arrays of:
    - `stock_price`,
    - `strike_price`,
    - `option_price`,
    - `is_call` (call/put flag).
  - Constructs matrices `A`, `B`, vectors `a`, `b`, and weights `w` so that
    $$\[
    w^\top \max(Ax + a,\ Bx + b)
    \]$$
    reproduces the portfolio payoff.
  - Defines a loss threshold and Wasserstein radius, then calls
    `get_worst_p_piecewise_cis_cis` inside `compute_average_worst_prob`.
  - Includes a `run_threshold_sweep(...)` helper to sweep over multiple loss thresholds and write a CSV (e.g. `CIS_piecewise_results.csv`) with:
    - Threshold,
    - Mean worst-case probability,
    - Relative CI width,
    - LSRE metrics and runtime.

This experiment captures **nonlinear, piecewise-linear payoffs** like portfolios of calls and puts marked to market.

#### Quadratic portfolio

- **`experiments_cis_quadratic.py`**  
  - Uses the same `mu`, `Sigma` data and then sets up a **quadratic loss function** defined by matrices `(Q, r, s)`.
  - Calls `get_worst_p_quadratic_cis_cis` to run the two-stage CIS estimator with `get_distance_quadratic`.
  - Reports the same statistics as the piecewise experiment (worst-case probability, CI, LSRE, runtime).

This models a portfolio whose loss is quadratic in the underlying factors (e.g. portfolios with convexity or variance-type payoffs).

#### Delta-hedging with transaction costs

- **`experiments_cis_delta.py`**  
  - Uses the same asset data plus:
    - Option prices and strikes,
    - Transaction cost coefficients (`quadratic_costs`, `linear_costs`, `fixed_costs`),
    - Black–Scholes **gamma** via `calculate_gamma`.
  - Models the **P&L of a delta-hedging strategy** for an option with:
    - Discrete rebalancing,
    - Quadratic + linear transaction costs.
  - Constructs a cost function in terms of the state vector and uses `get_worst_p_delta_cis_cis` (with `get_distance_delta_hedge`) to estimate the **worst-case probability** that hedging losses exceed a threshold under model misspecification.
  - As before, there is a `run_threshold_sweep(...)` helper for sweeping thresholds and saving CSVs.

#### American option experiments

- **`experiments.py`**  
  - Focuses on an **American put** in a simplified “no exercise at all” benchmark:
    - `make_put_boundary_grid(...)` builds an approximate early-exercise boundary on a time grid.
    - `get_worst_p_no_exercise_at_all_put(...)` evaluates the probability of **never exercising** vs the optimal boundary, including robust and non-robust variants (`get_worst_p_no_robust`).
  - Uses the same LSRE machinery to measure estimator efficiency.

This experiment shows how the same robust rare-event framework can be applied to **path-dependent / early-exercise** products.

---

## 3. Installation & dependencies

This is research code written in Python.

### 3.1 Python version

- Recommended: **Python 3.9+**

### 3.2 Python packages

You’ll need:

- `numpy`
- `scipy`
- `pandas`
- `quadprog` (for `find_xstar.py`)
- `cvxpy`
- A QP/SOCP solver supported by CVXPY (e.g. `OSQP`, `SCS`, `ECOS`)
- `joblib` (for parallel distance computations)
- `scipy.stats` (for Black–Scholes gamma in delta-hedging experiments)

You can install the typical stack with:

```bash
pip install numpy scipy pandas quadprog cvxpy joblib
