# data_generation.py

"""
This script contains the function for generating synthetic datasets based on the
configurations specified in config.py. It simulates daily case and death counts
for various changepoint scenarios.
"""

import numpy as np
from scipy.stats import poisson
from scipy.special import expit

from config import T, CASE_WAVE_FN, DELAY_DIST, SEED


def generate_dataset(scenario_config, rep_idx):
    """
    Generates a single dataset (cases and deaths) for a given scenario.

    This function operates based on the latent process model where the case
    fatality rate p_t is a sigmoid transformation of a piecewise-constant
    latent process, theta_t.

    Args:
        scenario_config (dict): A dictionary from the SCENARIOS constant in
                                config.py, defining the true changepoints and
                                latent parameter values.
        rep_idx (int): The replication index for the simulation run, used to
                       ensure a unique and reproducible random seed.

    Returns:
        dict: A dictionary containing the simulated time series for 'cases',
              'deaths', 'true_p_t', 'true_theta_t', and 'true_cps'.
    """
    # Set a reproducible random seed for each unique simulation run
    np.random.seed(SEED + rep_idx)

    # 1. Generate daily case counts from the defined wave function
    time_points = np.arange(1, T + 1)
    cases = np.array([CASE_WAVE_FN(t) for t in time_points])

    # 2. Generate the true piecewise-constant LATENT parameter series, theta(t)
    true_cps = scenario_config["true_cps"]
    true_theta_values = scenario_config["true_theta_values"]

    theta_t = np.zeros(T)
    boundaries = [0] + true_cps + [T]
    for i, theta_val in enumerate(true_theta_values):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        theta_t[start_idx:end_idx] = theta_val

    # 3. Transform the latent process to the true case fatality rate p(t)
    p_t = expit(theta_t)  # p_t = sigmoid(theta_t)

    # 4. Pre-calculate the delay PMF using the difference of the CDF
    delay_pmf = np.diff(DELAY_DIST.cdf(np.arange(T + 1)))

    # 5. Calculate expected deaths via convolution
    signal = cases * p_t
    expected_deaths = np.convolve(signal, delay_pmf)[:T]

    # 6. Generate observed deaths from a Poisson distribution
    expected_deaths[expected_deaths < 0] = 0
    deaths = poisson.rvs(expected_deaths)

    return {
        "cases": cases,
        "deaths": deaths,
        "true_p_t": p_t,
        "true_theta_t": theta_t,
        "true_cps": true_cps
    }
