# config.py

"""
This file centralizes all experimental configurations for the simulation study.
By modifying the parameters in this file, the entire behavior of the
simulation and analysis can be controlled without altering the core logic
of the other scripts.
"""

import numpy as np
from scipy.stats import gamma
from scipy.special import logit

# ==============================================================================
# --- Simulation Parameters ---
# ==============================================================================
# Replications for the main study, run after sensitivity analysis
N_REPLICATIONS = 10
# Replications for the sensitivity analysis (fewer runs for speed)
SENSITIVITY_REPLICATIONS = 5
# Length of the time series for each simulated dataset
T = 200
# Global random seed for reproducibility
SEED = 2024

# ==============================================================================
# --- Data Generation Parameters ---
# ==============================================================================
# Defines a symmetric case wave function for generating daily case counts
CASE_WAVE_FN = lambda t: 3000 - 5 * abs(100 - t)
# Defines the delay distribution from case confirmation to death using a Gamma distribution
# Parameters are based on the manuscript's specifications.
DELAY_DIST = gamma(a=2.03, scale=15.43 / 2.03)

# ==============================================================================
# --- RJMCMC Sampler Configuration ---
# ==============================================================================
MCMC_ITER = 20000
MCMC_BURN_IN = 5000
K_MAX = 10  # Maximum number of change points allowed in the model

# ==============================================================================
# --- Priors for RJMCMC (as per the manuscript) ---
# ==============================================================================
# These are the default values. The main simulation will use the optimal
# values determined by the sensitivity analysis.
PRIOR_THETA_MU = -3.5

# Prior for k: Geometric distribution p(k) = (1-p)^k * p.
# A higher p encourages sparsity (fewer changepoints).
PRIOR_K_GEOMETRIC_P = 0.9

# Prior for latent parameters: theta_s ~ Normal(MU, SIGMA^2)
PRIOR_THETA_SIGMA = 0.5

# ==============================================================================
# --- RJMCMC Proposal Distributions ---
# ==============================================================================
# Standard deviation for the auxiliary variable u in birth/death moves
PROPOSAL_U_SIGMA = 0.5
# Standard deviation for the random walk proposal when updating theta values
PROPOSAL_THETA_SIGMA = 0.2
# Symmetric window size (M) for the "move" move proposal
PROPOSAL_MOVE_WINDOW = 5

# ==============================================================================
# --- Simulation Scenarios (Piecewise-Constant) ---
# ==============================================================================
# Scenarios are defined in terms of the latent, unconstrained parameter theta.
SCENARIOS = {
    "Constant": {"true_cps": [], "true_theta_values": [logit(0.034)]},
    "Stepwise Increasing": {"true_cps": [80, 140], "true_theta_values": [logit(0.02), logit(0.04), logit(0.06)]},
    "Single Abrupt Increase": {"true_cps": [100], "true_theta_values": [logit(0.02), logit(0.05)]},
    "Single Abrupt Decrease": {"true_cps": [80], "true_theta_values": [logit(0.05), logit(0.02)]},
    "Increase then Decrease": {"true_cps": [80, 120], "true_theta_values": [logit(0.02), logit(0.06), logit(0.04)]},
    "Decrease then Increase": {"true_cps": [80, 120], "true_theta_values": [logit(0.06), logit(0.02), logit(0.04)]}
}

# ==============================================================================
# --- Sensitivity Analysis Configuration ---
# ==============================================================================
# Finer grid of hyperparameter values to test in the sensitivity analysis
SENSITIVITY_GRID = {
    'PRIOR_K_GEOMETRIC_P': [0.1, 0.3, 0.5, 0.7, 0.9],
    'PRIOR_THETA_SIGMA': [0.5, 1.0, 1.5, 2.0, 2.5]
}
# A representative scenario to use for detailed sensitivity plots
SENSITIVITY_SCENARIO_FOR_PLOT = "Increase then Decrease"

# ==============================================================================
# --- Output Directories ---
# ==============================================================================
DATA_DIR = "data"
SENSITIVITY_RESULTS_DIR = "results_sensitivity"
MAIN_RESULTS_DIR = "results_main"
SIGNAL_CACHE_DIR = "results_cache"
PLOTS_DIR = "plots"
