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
SEED = 2025

# ==============================================================================
# --- Data Generation Parameters ---
# ==============================================================================
# Defines a symmetric case wave function for generating daily case counts
CASE_WAVE_FN = lambda t: 3000 - 5 * abs(100 - t)
# Defines the default delay distribution from case confirmation to death
# This is also the "perfectly matched" scenario.
DELAY_DIST = gamma(a=2.03, scale=15.43 / 2.03)

# ==============================================================================
# --- RJMCMC Sampler Configuration ---
# ==============================================================================
# MCMC iterations and burn-in period
MCMC_ITER = 20000
MCMC_BURN_IN = 5000
# Maximum number of changepoints allowed in the model
K_MAX = 10

# Default priors - will be overridden by sensitivity analysis or optimal values
PRIOR_K_GEOMETRIC_P = 0.9
PRIOR_THETA_MU = -3.5
PRIOR_THETA_SIGMA = 1.0

REAL_DATA_PRIOR_K_GEOMETRIC_P = 0.95
REAL_DATA_PRIOR_THETA_SIGMA = 1.0

# Proposal distribution variances for the MCMC updates
PROPOSAL_U_SIGMA = 0.5
PROPOSAL_THETA_SIGMA = 0.5
PROPOSAL_MOVE_WINDOW = 5

# ==============================================================================
# --- Benchmark Method Configuration ---
# ==============================================================================
# Cache directory for the computationally expensive rtaCFR signal estimation
SIGNAL_CACHE_DIR = "rtacfr_cache"

# ==============================================================================
# --- Directory Configuration ---
# ==============================================================================
# Directories for storing simulation results and output plots
MAIN_RESULTS_DIR = "results/main"
SENSITIVITY_RESULTS_DIR = "results/sensitivity"
PLOTS_DIR = "plots"

# ==============================================================================
# --- Simulation Scenarios ---
# ==============================================================================
# Defines the different ground truth scenarios for the simulation study
SCENARIOS = {
    "No Change": {"true_cps": [], "true_theta_values": [logit(0.04)]},
    "Single Abrupt Increase": {"true_cps": [100], "true_theta_values": [logit(0.02), logit(0.05)]},
    "Single Abrupt Decrease": {"true_cps": [80], "true_theta_values": [logit(0.05), logit(0.02)]},
    "Increase then Decrease": {"true_cps": [80, 120], "true_theta_values": [logit(0.02), logit(0.06), logit(0.04)]},
    "Decrease then Increase": {"true_cps": [80, 120], "true_theta_values": [logit(0.06), logit(0.02), logit(0.04)]}
}

# ==============================================================================
# --- Sensitivity Analysis Configuration ---
# ==============================================================================
# Grid for prior hyperparameters
SENSITIVITY_GRID_PRIORS = {
    'PRIOR_K_GEOMETRIC_P': [0.1, 0.3, 0.5, 0.7, 0.9],
    'PRIOR_THETA_SIGMA': [0.5, 1.0, 1.5, 2.0, 2.5]
}

# Grid for delay distribution misspecification
# The mean delay is shifted by +/- 3 days for under/overestimation
DELAY_DIST_SENSITIVITY = {
    "Underestimated Delay (Mean=12.43)": gamma(a=2.03, scale=12.43 / 2.03),
    "Perfectly Matched Delay (Mean=15.43)": gamma(a=2.03, scale=15.43 / 2.03),
    "Overestimated Delay (Mean=18.43)": gamma(a=2.03, scale=18.43 / 2.03)
}

# To be filled by the analysis script after sensitivity run
OPTIMAL_PARAMS = {
    'PRIOR_K_GEOMETRIC_P': None,
    'PRIOR_THETA_SIGMA': None
}