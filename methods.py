# methods.py

import numpy as np
from scipy.stats import poisson, beta, norm
from scipy.special import logit, expit
import ruptures as rpt
from numba import njit

from config import (MCMC_ITER, MCMC_BURN_IN, K_MAX, DELAY_DIST, T,
                    PRIOR_K_LAMBDA, PRIOR_P_ALPHA, PRIOR_P_BETA,
                    PROPOSAL_U_DIST, PROPOSAL_P_LOGIT_STD)

# --- Numba JIT Accelerated Log-Likelihood ---
@njit
def calculate_log_likelihood(deaths, cases, p_t, delay_pmf):
    """
    Calculates the log-likelihood of the death series given the parameters.
    Accelerated with Numba's Just-In-Time (JIT) compiler.
    """
    signal = cases * p_t
    # Use np.convolve equivalent for Numba
    expected_deaths = np.zeros(T)
    for i in range(T):
        for j in range(i + 1):
            if j < len(delay_pmf):
                expected_deaths[i] += signal[i-j] * delay_pmf[j]

    log_lik = 0.0
    for t in range(T):
        mu = max(1e-9, expected_deaths[t]) # Avoid log(0)
        # log(mu^d * e^-mu / d!) = d*log(mu) - mu - log(d!)
        # We can ignore log(d!) as it's a constant for observed data
        log_lik += deaths[t] * np.log(mu) - mu
    
    return log_lik

# --- RJMCMC Sampler ---
def run_rjmcmc(data):
    """
    The main RJMCMC sampler for the CFR model.
    """
    cases, deaths = data["cases"], data["deaths"]
    
    # Pre-calculate delay PMF
    s = np.arange(0, T)
    delay_pmf = DELAY_DIST.pdf(s)
    delay_pmf /= np.sum(delay_pmf)

    # --- MCMC State Initialization ---
    k = 0
    taus = []
    p_values = [np.random.beta(PRIOR_P_ALPHA, PRIOR_P_BETA)]
    
    # Storage for posterior samples
    k_samples = []
    taus_samples = []
    p_samples = []

    # --- MCMC Loop ---
    for iter_idx in range(MCMC_ITER + MCMC_BURN_IN):
        # Determine move type
        u_move = np.random.rand()
        if k == 0:
            move_type = "birth"
        elif k == K_MAX:
            move_type = "death"
        else:
            if u_move < 0.25: move_type = "birth"
            elif u_move < 0.50: move_type = "death"
            elif u_move < 0.75: move_type = "move"
            else: move_type = "update"

        # --- Calculate current state likelihood and prior ---
        current_boundaries = [0] + taus + [T]
        p_t_current = np.zeros(T)
        for i, p_val in enumerate(p_values):
            p_t_current[current_boundaries[i]:current_boundaries[i+1]] = p_val
        
        log_lik_current = calculate_log_likelihood(deaths, cases, p_t_current, delay_pmf)
        log_prior_current = poisson.logpmf(k, PRIOR_K_LAMBDA) + \
                            np.sum(beta.logpdf(p_values, PRIOR_P_ALPHA, PRIOR_P_BETA))

        if move_type == "birth":
            # Propose new changepoint
            possible_cps = set(range(1, T)) - set(taus)
            tau_star = np.random.choice(list(possible_cps))
            
            # Find which segment it splits
            seg_idx = np.searchsorted(taus, tau_star)
            p_j = p_values[seg_idx]
            
            # Propose new p_values using logit-space split
            u_aux = np.random.beta(2, 2) if PROPOSAL_U_DIST == "beta" else np.random.randn()
            logit_p_j = logit(p_j)
            
            # This split is from Richardson & Green (1997)
            logit_p1 = logit_p_j - u_aux
            logit_p2 = logit_p_j + u_aux
            p1_star, p2_star = expit(logit_p1), expit(logit_p2)

            if not (0 < p1_star < 1 and 0 < p2_star < 1): continue

            # New state
            k_new = k + 1
            taus_new = sorted(taus + [tau_star])
            p_values_new = p_values[:seg_idx] + [p1_star, p2_star] + p_values[seg_idx+1:]
            
            # Acceptance probability
            p_t_new = np.zeros(T)
            new_boundaries = [0] + taus_new + [T]
            for i, p_val in enumerate(p_values_new):
                p_t_new[new_boundaries[i]:new_boundaries[i+1]] = p_val

            log_lik_new = calculate_log_likelihood(deaths, cases, p_t_new, delay_pmf)
            log_prior_new = poisson.logpmf(k_new, PRIOR_K_LAMBDA) + \
                            np.sum(beta.logpdf(p_values_new, PRIOR_P_ALPHA, PRIOR_P_BETA))
            
            # Jacobian for logit-space transform
            log_jacobian = np.log(p1_star * (1-p1_star) * p2_star * (1-p2_star) / (p_j * (1-p_j)))
            
            # Proposal ratio
            log_proposal_ratio = np.log(0.25 / 0.25) + np.log(len(possible_cps)) - np.log(beta.pdf(u_aux, 2, 2))
            
            log_alpha = (log_lik_new - log_lik_current) + \
                        (log_prior_new - log_prior_current) - \
                        log_proposal_ratio + log_jacobian

            if np.log(np.random.rand()) < log_alpha:
                k, taus, p_values = k_new, taus_new, p_values_new

        # ... (Implement death, move, update moves similarly) ...

        # --- Store Sample ---
        if iter_idx >= MCMC_BURN_IN:
            k_samples.append(k)
            taus_samples.append(taus)
            p_samples.append(p_values)

    # Post-process to get posterior estimates
    est_k = int(np.median(k_samples))
    
    # Find the most common set of changepoints for the modal k
    frequent_taus = [tuple(t) for t in taus_samples if len(t) == est_k]
    est_taus = []
    if frequent_taus:
        est_taus = list(max(set(frequent_taus), key=frequent_taus.count))

    return {"k": est_k, "taus": est_taus}


# --- Benchmark Methods ---
def run_pelt(data):
    """Wrapper for PELT method from the 'ruptures' library."""
    # Use a naive death-to-case ratio as the signal for benchmark methods
    signal = data["deaths"] / (data["cases"] + 1e-6)
    algo = rpt.Pelt(model="rbf").fit(signal)
    # Use a penalty to select the number of changepoints
    result = algo.predict(pen=np.log(T) * 2) # MBIC-like penalty
    # The last element is T, so we exclude it
    return {"k": len(result)-1, "taus": result[:-1]}

def run_binseg(data):
    """Wrapper for Binary Segmentation method."""
    signal = data["deaths"] / (data["cases"] + 1e-6)
    algo = rpt.Binseg(model="rbf").fit(signal)
    
    # The penalty for model selection, a common choice is log(T)
    penalty = np.log(T)
    
    # Find the optimal number of changepoints by minimizing a penalized cost
    best_k = 0
    # Initialize min_cost with the cost of a model with 0 changepoints.
    # The segmentation for 0 changepoints is just the end of the signal.
    min_cost = algo.cost.sum_of_costs(bkps=[T]) 

    for k_candidate in range(1, K_MAX + 1):
        # Get the locations for this number of changepoints
        bkps_for_k = algo.predict(n_bkps=k_candidate)
        # Calculate the cost for this specific segmentation
        cost = algo.cost.sum_of_costs(bkps=bkps_for_k) + penalty * k_candidate
        
        if cost < min_cost:
            min_cost = cost
            best_k = k_candidate
            
    # Once the best k is found, predict the final locations
    result = algo.predict(n_bkps=best_k)
    
    # The last element of the result is T (the length of the signal), so we exclude it
    est_taus = result[:-1]

    return {"k": best_k, "taus": est_taus}
