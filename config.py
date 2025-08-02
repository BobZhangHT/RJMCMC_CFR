# config.py

import numpy as np
from scipy.stats import gamma

# --- Simulation Parameters ---
N_REPLICATIONS = 100  # Number of replicate datasets for each scenario
T = 200               # Length of the time series
SEED = 2025           # For reproducibility

# --- Data Generation Parameters ---
# Defines a symmetric case wave: c(t) = 3000 - 5 * |100 - t|
CASE_WAVE_FN = lambda t: 3000 - 5 * abs(100 - t)
# Delay from case confirmation to death (Gamma distribution)
# Matches mean=15.43, shape=2.03 from the report
DELAY_DIST = gamma(a=2.03, scale=15.43 / 2.03)

# --- RJMCMC Sampler Configuration ---
MCMC_ITER = 20000
MCMC_BURN_IN = 5000
K_MAX = 10  # Maximum number of change points allowed

# --- Priors for RJMCMC ---
# p(k) ~ Poisson(LAMBDA)
PRIOR_K_LAMBDA = 1.0
# p_j ~ Beta(alpha, beta)
PRIOR_P_ALPHA = 1.0
PRIOR_P_BETA = 1.0

# --- RJMCMC Proposal Distributions ---
# u ~ Beta(2, 2) for birth/death moves
PROPOSAL_U_DIST = "beta" # or "normal"
# For logit-space proposals on p_j
PROPOSAL_P_LOGIT_STD = 0.2

# --- Simulation Scenarios ---
# Each scenario is a dictionary defining the true fatality rate p(t)
# Scenarios are adapted from the report to be explicitly piecewise-constant
SCENARIOS = {
    # "Constant": {
    #     "true_cps": [],
    #     "true_p_values": [0.034]
    # },
    # "Step-wise Increasing": {
    #     "true_cps": [80, 140],
    #     "true_p_values": [0.02, 0.04, 0.06]
    # },
    # "Single Abrupt Increase": {
    #     "true_cps": [100],
    #     "true_p_values": [0.02, 0.05]
    # },
    # "Single Abrupt Decrease": {
    #     "true_cps": [80],
    #     "true_p_values": [0.05, 0.02]
    # },
    # "Increase-then-Decrease": {
    #     "true_cps": [80, 120],
    #     "true_p_values": [0.02, 0.06, 0.02]
    # },
    "Decrease-then-Increase": {
        "true_cps": [80, 120],
        "true_p_values": [0.06, 0.02, 0.06]
    }
}

# --- Output Directories ---
DATA_DIR = "data"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
