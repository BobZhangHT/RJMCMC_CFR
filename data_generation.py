# data_generation.py

import numpy as np
from scipy.stats import poisson

from config import T, CASE_WAVE_FN, DELAY_DIST, SEED

def generate_dataset(scenario_config, rep_idx):
    """
    Generates a single dataset (cases and deaths) for a given scenario.
    """
    np.random.seed(SEED + rep_idx)

    # 1. Generate daily case counts from the defined wave function
    time_points = np.arange(1, T + 1)
    cases = np.array([CASE_WAVE_FN(t) for t in time_points])

    # 2. Generate the true piecewise-constant fatality rate p(t)
    true_cps = scenario_config["true_cps"]
    true_p_values = scenario_config["true_p_values"]
    
    p_t = np.zeros(T)
    boundaries = [0] + true_cps + [T]
    for i, p_val in enumerate(true_p_values):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        p_t[start_idx:end_idx] = p_val
    
    # 3. Pre-calculate the delay PMF
    s = np.arange(0, T)
    delay_pmf = DELAY_DIST.pdf(s)
    delay_pmf /= np.sum(delay_pmf) # Normalize to sum to 1

    # 4. Calculate expected deaths via convolution
    # The term p(t-s)c(t-s)f(s+1) in the report is a convolution.
    # We convolve the product of cases and p_t with the delay distribution.
    signal = cases * p_t
    # We need to be careful with indices. f(s+1) means we use delay_pmf[s]
    expected_deaths = np.convolve(signal, delay_pmf)[:T]

    # 5. Generate observed deaths from a Poisson distribution
    # Ensure the mean of the Poisson is non-negative
    expected_deaths[expected_deaths < 0] = 0
    deaths = poisson.rvs(expected_deaths)

    return {
        "cases": cases,
        "deaths": deaths,
        "true_p_t": p_t,
        "true_cps": true_cps
    }
