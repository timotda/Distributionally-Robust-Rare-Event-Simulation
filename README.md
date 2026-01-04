# Distributionally Robust Rare-Event Simulation in Finance

This repository contains the code from my summer research project on **distributionally robust rare-event simulation** for financial risk.

## Project Overview

The research addresses robust risk management problems of the form:
```
sup_{ν: d(ν,μ) ≤ δ} P_ν(Loss ≥ threshold)
```
where we seek the worst-case probability over all distributions ν within distance δ of a nominal distribution with mean μ.

## Repository Structure

### Core Problem Types

- **`Linear/`** - Linear constraint problems (portfolio loss functions)
- **`Quadratic/`** - Quadratic constraint problems 
- **`piecewise/`** - Piecewise linear constraints (options portfolios)
- **`Rebalance Delta Hedging/`** - Dynamic hedging with rebalancing costs
- **`American Options/`** - American option boundary analysis
- **`American option boundary/`** - Additional American option studies

### Estimation Methods

Each problem type contains subdirectories for different estimation approaches:

- **`CIS-CIS/`** - Two-stage Conditional Importance Sampling
- **`MC-MC/`** - Two-stage Monte Carlo 
- **`CIS-MC/`** - Hybrid: CIS for radius estimation, MC for probability
- **`MC-CIS/`** - Hybrid: MC for radius estimation, CIS for probability

## Key Algorithms

### Two-Stage Estimation Framework

1. **Stage 1**: Estimate the optimal ambiguity radius `u*` that satisfies the constraint
2. **Stage 2**: Estimate the worst-case probability at radius `u*`

### Method Implementations

- **Monte Carlo (MC)**: Standard sampling-based estimation
- **Conditional Importance Sampling (CIS)**: Variance reduction through optimal importance sampling
- **Hybrid Methods**: Combine MC and CIS for different stages

## Main Files

### Core Functions (per problem type)
- `get_worst_p_[type]_[method1]_[method2].py` - Main estimation functions
- `experiments_[method]_[type].py` - Experimental frameworks and parameter sweeps
- `get_radius_[method]_[type].py` - Stage 1 radius estimation
- `robust_[method]_[type].py` - Stage 2 probability estimation

### Utilities
- [`find_xstar.py`](Linear/CIS-CIS/find_xstar.py) - Optimization problem solver
- [`shift_samples.py`](Linear/CIS-CIS/shift_samples.py) - Importance sampling transformations
- Various helper functions for each constraint type

## Data and Results

### Input Data
The experiments use S&P 500 stock data (MSFT, NVDA, AAPL, AMZN, META, GOOGL, BRK.B, TSLA, JPM, UNH):
- Returns from June 2021 to June 2024
- Mean returns μ and correlation matrix Rho defined in experiment files

### Output Files
- `*_results.csv` - Experimental results with performance metrics
- [`ratios_vs_CIS_linear_results_with_MC.csv`](Linear/ratios_vs_CIS_linear_results_with_MC.csv) - Method comparisons
- [`plot.ipynb`](Linear/plot.ipynb) - Visualization and analysis

## Running Experiments

### Basic Usage

```python
# Example: Linear CIS-CIS experiment
from Linear.CIS_CIS.experiments_cis_linear import run_threshold_sweep

# Set parameters
thresholds = [1, 2, 4, 6, 10]  # Loss thresholds to test
delta = 0.02                    # Ambiguity radius
N_tot = int(1e5)               # Total samples
runs = 5                       # Independent runs

# Run experiment
run_threshold_sweep(thresholds, delta, mu, Sigma, w, N_alpha, N_tot, runs, rng)
```

### Key Parameters

- `delta`: Ambiguity set radius (Wasserstein distance)
- `loss_threshold`: Loss level for probability estimation
- `mu`, `Sigma`: Mean and covariance of nominal distribution
- `w`: Portfolio weights (linear case) or constraint matrices (other cases)
- `N_alpha`, `N_tot`: Sample allocation between stages
- `runs`: Number of independent replications

## Performance Metrics

The experiments track several key metrics:

- **`mean_p`**: Average worst-case probability estimate
- **`rel_error`**: Relative error (confidence interval / estimate)
- **`lsre_mean`**: Log-scale relative efficiency
- **`mean_time_sec`**: Computational time
- **`var_H`**, **`var_p`**: Variance estimates for each stage

## Method Comparison

The [`ratios_vs_CIS_linear_results_with_MC.csv`](Linear/ratios_vs_CIS_linear_results_with_MC.csv) file contains comparative analysis showing:
- Efficiency ratios between methods (CIS, MC, hybrid approaches)
- Time complexity comparisons
- Accuracy trade-offs

## Specialized Applications

### American Options
- [`american_put_boundary.py`](American option boundary/CIS-CIS ( no exercise) /american_put_boundary.py) - Boundary computation
- Analysis of early exercise decisions under ambiguity

### VaR Estimation  
- [`VaR_cis_cis.py`](Linear/CIS-CIS/VaR_cis_cis.py) - Value-at-Risk under distributional ambiguity
- Bootstrap confidence intervals for risk measures

### Delta Hedging
- Dynamic rebalancing with transaction costs
- Robust hedging under parameter uncertainty

## Dependencies

```python
numpy
scipy
pandas
matplotlib
numba  # For performance optimization
```

## Research Context

This work addresses fundamental challenges in financial risk management:

1. **Model Uncertainty**: Traditional risk models assume known distributions
2. **Computational Efficiency**: Robust optimization problems are computationally intensive  
3. **Practical Implementation**: Methods must scale to realistic portfolio sizes

The hybrid CIS-MC approaches often provide the best balance of accuracy and computational efficiency, as demonstrated in the comparative results.
