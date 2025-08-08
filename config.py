# config.py

import numpy as np
from scipy.stats import gamma
from scipy.special import logit

# --- Simulation Parameters ---
# Replications for the main study, run after sensitivity analysis
N_REPLICATIONS = 10
# Replications for the sensitivity analysis (fewer runs for speed)
SENSITIVITY_REPLICATIONS = 5
T = 200
SEED = 2024

# --- Data Generation Parameters ---
CASE_WAVE_FN = lambda t: 3000 - 5 * abs(100 - t)
DELAY_DIST = gamma(a=2.03, scale=15.43 / 2.03)

# --- RJMCMC Sampler Configuration ---
MCMC_ITER = 20000
MCMC_BURN_IN = 5000
K_MAX = 10

# --- Priors for RJMCMC ---
# These are the default values. The main simulation will use the optimal
# values determined by the sensitivity analysis.
PRIOR_THETA_MU = -3.5

# --- RJMCMC Proposal Distributions ---
PROPOSAL_U_SIGMA = 0.5
PROPOSAL_THETA_SIGMA = 0.2
PROPOSAL_MOVE_WINDOW = 5

# --- Simulation Scenarios (Piecewise-Constant) ---
SCENARIOS = {
    "Constant": {"true_cps": [], "true_theta_values": [logit(0.034)]},
    "Stepwise Increasing": {"true_cps": [80, 140], "true_theta_values": [logit(0.02), logit(0.04), logit(0.06)]},
    "Single Abrupt Increase": {"true_cps": [100], "true_theta_values": [logit(0.02), logit(0.05)]},
    "Single Abrupt Decrease": {"true_cps": [80], "true_theta_values": [logit(0.05), logit(0.02)]},
    "Increase then Decrease": {"true_cps": [80, 120], "true_theta_values": [logit(0.02), logit(0.06), logit(0.04)]},
    "Decrease then Increase": {"true_cps": [80, 120], "true_theta_values": [logit(0.06), logit(0.02), logit(0.04)]}
}

# --- Sensitivity Analysis Configuration ---
# Finer grid for the sensitivity analysis
SENSITIVITY_GRID = {
    'PRIOR_K_GEOMETRIC_P': [0.1, 0.3, 0.5, 0.7, 0.9],
    'PRIOR_THETA_SIGMA': [0.5, 1.0, 1.5, 2.0, 2.5]
}
# Scenario to use for the detailed sensitivity analysis figure
SENSITIVITY_SCENARIO_FOR_PLOT = "Increase then Decrease"

# --- Output Directories ---
DATA_DIR = "data"
SENSITIVITY_RESULTS_DIR = "results_sensitivity"
MAIN_RESULTS_DIR = "results_main"
PLOTS_DIR = "plots"
