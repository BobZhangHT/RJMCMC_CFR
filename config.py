# config.py

import numpy as np
from scipy.stats import gamma, norm
from scipy.special import logit

# --- Simulation Parameters ---
N_REPLICATIONS = 100  # Number of replicate datasets for each scenario
T = 200               # Length of the time series
SEED = 2024           # For reproducibility

# --- Data Generation Parameters ---
CASE_WAVE_FN = lambda t: 3000 - 5 * abs(100 - t)
DELAY_DIST = gamma(a=2.03, scale=15.43 / 2.03)

# --- RJMCMC Sampler Configuration ---
MCMC_ITER = 20000
MCMC_BURN_IN = 5000
K_MAX = 10  # Maximum number of change points allowed

# --- Priors for RJMCMC ---
# # p(k) ~ Poisson(LAMBDA)
PRIOR_K_LAMBDA = 0.001 #1.0

# NEW GEOMETRIC PRIOR for k: p(k) = (1-p)^k * p
# A higher p means a stronger penalty against complexity. p=0.5 is a good default.
PRIOR_K_GEOMETRIC_P = 0.95

# Prior for latent parameters: theta_s ~ Normal(MU, SIGMA^2)
PRIOR_THETA_MU = -3.5  # A reasonable center for logit(p), e.g., logit(0.03) ~ -3.5
PRIOR_THETA_SIGMA = 1.5

# --- RJMCMC Proposal Distributions ---
# u ~ Normal(0, SIGMA_U^2) for birth/death moves
PROPOSAL_U_SIGMA = 0.5
# Proposal for updating theta: theta' ~ Normal(theta, SIGMA_THETA^2)
PROPOSAL_THETA_SIGMA = 0.2

# --- Simulation Scenarios ---
# Defined in terms of the latent, unconstrained parameter theta.
# p_values are shown in comments for interpretability.
SCENARIOS = {
    "Constant": {
        "true_cps": [],
        "true_theta_values": [logit(0.034)] # p ~ 0.034
    },
    "Step-wise Increasing": {
        "true_cps": [80, 140],
        "true_theta_values": [logit(0.02), logit(0.04), logit(0.06)]
    },
    "Single Abrupt Increase": {
        "true_cps": [100],
        "true_theta_values": [logit(0.02), logit(0.05)]
    },
    "Single Abrupt Decrease": {
        "true_cps": [80],
        "true_theta_values": [logit(0.05), logit(0.02)]
    },
    "Increase-then-Decrease": {
        "true_cps": [80, 120],
        "true_theta_values": [logit(0.02), logit(0.06), logit(0.02)]
    },
    "Decrease-then-Increase": {
        "true_cps": [80, 120],
        "true_theta_values": [logit(0.06), logit(0.02), logit(0.06)]
    }
}

# --- Output Directories ---
DATA_DIR = "data"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
