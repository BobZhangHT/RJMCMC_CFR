# methods.py

import numpy as np
from scipy.stats import poisson, norm
from scipy.special import expit
import ruptures as rpt
from numba import njit
from collections import Counter

from config import (MCMC_ITER, MCMC_BURN_IN, K_MAX, DELAY_DIST, T,
                    PRIOR_K_LAMBDA,  
                    PRIOR_K_GEOMETRIC_P,
                    PRIOR_THETA_MU, PRIOR_THETA_SIGMA,
                    PROPOSAL_U_SIGMA, PROPOSAL_THETA_SIGMA)

# ==============================================================================
# ACCELERATED HELPER FUNCTIONS (NUMBA JIT)
# These functions are compiled to machine code by Numba for maximum speed.
# ==============================================================================

@njit
def _sigmoid(x):
    """Numba-compatible sigmoid function to map latent parameters to probabilities."""
    return 1.0 / (1.0 + np.exp(-x))

@njit
def _calculate_log_likelihood(deaths, cases, theta_t, delay_pmf):
    """
    Calculates the log-likelihood of the death series given the latent parameters.
    This corresponds to Equation (3) in the manuscript.
    """
    # 1. Transform latent theta_t to case fatality rate p_t
    p_t = _sigmoid(theta_t)
    
    # 2. Calculate the expected number of deaths (mu_t) via convolution
    # This corresponds to Equation (2) in the manuscript.
    signal = cases * p_t
    expected_deaths = np.zeros(T)
    for i in range(T):
        for j in range(i + 1):
            if j < len(delay_pmf):
                expected_deaths[i] += signal[i - j] * delay_pmf[j]

    # 3. Calculate the Poisson log-likelihood
    log_lik = 0.0
    for t in range(T):
        mu = max(1e-9, expected_deaths[t]) # Add epsilon to avoid log(0)
        # log(Poisson(d | mu)) = d*log(mu) - mu - log(d!)
        # The log(d!) term is a constant for the observed data, so it can be ignored
        # as it cancels out in all acceptance ratio calculations.
        log_lik += deaths[t] * np.log(mu) - mu
    
    return log_lik

@njit
def _get_theta_t_from_state(k, taus, theta_values):
    """Constructs the full theta(t) time series from a given state (k, taus, theta_values)."""
    theta_t = np.zeros(T)
    # Define segment boundaries: [0, tau_1, tau_2, ..., T]
    boundaries = np.array([0] + list(taus) + [T])
    for i in range(k + 1):
        # Assign the constant theta value to each segment
        theta_t[boundaries[i]:boundaries[i+1]] = theta_values[i]
    return theta_t

@njit
def _log_pdf_normal(x, mu, sigma):
    """Numba-compatible log PDF for a Normal distribution for prior calculations."""
    return -0.5 * np.log(2 * np.pi * sigma**2) - ((x - mu)**2) / (2 * sigma**2)

@njit
def _log_pmf_poisson(k, lam):
    """Numba-compatible log PMF for a Poisson distribution for prior calculations."""
    # Using log-gamma for factorial term for numerical stability
    log_factorial_k = 0.0
    for i in range(1, int(k) + 1):
        log_factorial_k += np.log(i)
    return k * np.log(lam) - lam - log_factorial_k

@njit
def _log_pmf_geometric(k, p):
    """Numba-compatible log PMF for a Geometric distribution: p(k) = (1-p)^k * p."""
    if k < 0 or p <= 0 or p > 1:
        return -np.inf
    return k * np.log(1.0 - p) + np.log(p)

# ==============================================================================
# RJMCMC SAMPLER (NUMBA JIT ACCELERATED)
# ==============================================================================

# @njit
def _rjmcmc_sampler_numba(deaths, cases, delay_pmf):
    """
    The core RJMCMC sampler loop for the latent parameter model, optimized with Numba.
    This function implements the algorithm described in Section 3.2 of the manuscript.
    """
    # --- MCMC State Initialization ---
    k = 0
    taus = np.array([], dtype=np.int64)
    theta_values = np.array([np.random.normal(PRIOR_THETA_MU, PRIOR_THETA_SIGMA)])
    
    # Storage for posterior samples (pre-allocate for speed)
    k_samples = np.zeros(MCMC_ITER, dtype=np.int64)
    taus_samples = np.full((MCMC_ITER, K_MAX), -1, dtype=np.int64)
    theta_samples = np.full((MCMC_ITER, K_MAX + 1), np.nan, dtype=np.float64)

    # --- MCMC Loop ---
    for iter_idx in range(MCMC_ITER + MCMC_BURN_IN):
        # --- Calculate current state likelihood and prior ---
        theta_t_current = _get_theta_t_from_state(k, taus, theta_values)
        log_lik_current = _calculate_log_likelihood(deaths, cases, theta_t_current, delay_pmf)
        
        log_prior_current = _log_pmf_poisson(k, PRIOR_K_LAMBDA) #_log_pmf_geometric(k, PRIOR_K_GEOMETRIC_P) 
        for val in theta_values:
            log_prior_current += _log_pdf_normal(val, PRIOR_THETA_MU, PRIOR_THETA_SIGMA)

        # --- Select and Perform Move ---
        u_move = np.random.rand()
        if k == 0: move_type = "birth"
        elif k == K_MAX: move_type = "death"
        else:
            if u_move < 0.25: move_type = "birth"
            elif u_move < 0.50: move_type = "death"
            elif u_move < 0.75: move_type = "move"
            else: move_type = "update"

        # --- 1. BIRTH MOVE (Section 3.2.1) ---
        if move_type == "birth":
            possible_cps = np.array([i for i in range(1, T) if i not in taus])
            tau_star = np.random.choice(possible_cps)
            seg_idx = np.searchsorted(taus, tau_star)
            
            theta_j = theta_values[seg_idx]
            u_aux = np.random.normal(0, PROPOSAL_U_SIGMA)
            
            theta1_star = theta_j - u_aux
            theta2_star = theta_j + u_aux

            k_new, taus_new = k + 1, np.sort(np.append(taus, tau_star))
            theta_values_new = np.concatenate((theta_values[:seg_idx], np.array([theta1_star, theta2_star]), theta_values[seg_idx+1:]))
            
            theta_t_new = _get_theta_t_from_state(k_new, taus_new, theta_values_new)
            log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf)

            # print(k, PRIOR_K_LAMBDA, _log_pmf_poisson(k, PRIOR_K_LAMBDA), PRIOR_K_GEOMETRIC_P, _log_pmf_geometric(k, PRIOR_K_GEOMETRIC_P) )
            log_prior_new = _log_pmf_poisson(k, PRIOR_K_LAMBDA) #_log_pmf_geometric(k, PRIOR_K_GEOMETRIC_P) 
            for val in theta_values_new:
                log_prior_new += _log_pdf_normal(val, PRIOR_THETA_MU, PRIOR_THETA_SIGMA)
            
            # Acceptance probability (Equation 4)
            # log(q(reverse)/q(forward)) = log(p_death/p_birth) + log(T-1-k) - log(k+1) - log(g(u))
            # Assuming p_death=p_birth, this simplifies.
            log_proposal_ratio = np.log(T - 1 - k) - np.log(k + 1) - _log_pdf_normal(u_aux, 0, PROPOSAL_U_SIGMA)

            # FIX: Break calculation into multiple lines to help Numba's type inference
            log_lik_term = log_lik_new - log_lik_current
            log_prior_term = log_prior_new - log_prior_current
            log_alpha = log_lik_term + log_prior_term + log_proposal_ratio

            if np.log(np.random.rand()) < log_alpha:
                k, taus, theta_values = k_new, taus_new, theta_values_new

        # --- 2. DEATH MOVE (Section 3.2.2) ---
        elif move_type == "death":
            idx_to_remove = np.random.randint(0, k)
            theta1, theta2 = theta_values[idx_to_remove], theta_values[idx_to_remove+1]
            
            theta_j_star = (theta1 + theta2) / 2.0
            u_aux = (theta2 - theta1) / 2.0

            k_new, taus_new = k - 1, np.delete(taus, idx_to_remove)
            theta_values_new = np.concatenate((theta_values[:idx_to_remove], np.array([theta_j_star]), theta_values[idx_to_remove+2:]))

            theta_t_new = _get_theta_t_from_state(k_new, taus_new, theta_values_new)
            log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf)
            
            log_prior_new = _log_pmf_poisson(k, PRIOR_K_LAMBDA) #_log_pmf_geometric(k, PRIOR_K_GEOMETRIC_P) 
            for val in theta_values_new:
                log_prior_new += _log_pdf_normal(val, PRIOR_THETA_MU, PRIOR_THETA_SIGMA)
            
            # This is the reverse of the birth move's proposal ratio
            # log(q(reverse)/q(forward)) = log(p_birth/p_death) + log(k) - log(T-k) + log(g(u))
            log_proposal_ratio = np.log(k) - np.log(T - 1 - k_new) + _log_pdf_normal(u_aux, 0, PROPOSAL_U_SIGMA)
            
            log_lik_term = log_lik_new - log_lik_current
            log_prior_term = log_prior_new - log_prior_current
            log_alpha = log_lik_term + log_prior_term + log_proposal_ratio

            if np.log(np.random.rand()) < log_alpha:
                k, taus, theta_values = k_new, taus_new, theta_values_new

        # --- 3. MOVE MOVE (Section 3.2.3) ---
        elif move_type == "move":
            idx_to_move = np.random.randint(0, k)
            lower_bound = taus[idx_to_move-1] + 1 if idx_to_move > 0 else 1
            upper_bound = taus[idx_to_move+1] - 1 if idx_to_move < k - 1 else T - 1
            
            if lower_bound < upper_bound:
                tau_new = np.random.randint(lower_bound, upper_bound + 1)
                taus_new = np.copy(taus)
                taus_new[idx_to_move] = tau_new
                
                theta_t_new = _get_theta_t_from_state(k, np.sort(taus_new), theta_values)
                log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf)
                
                # Acceptance probability (Equation 5)
                log_alpha = log_lik_new - log_lik_current
                if np.log(np.random.rand()) < log_alpha:
                    taus = np.sort(taus_new)

        # --- 4. UPDATE MOVE (Section 3.2.4) ---
        elif move_type == "update":
            for j in range(k + 1):
                theta_current = theta_values[j]
                theta_proposal = np.random.normal(theta_current, PROPOSAL_THETA_SIGMA)
                
                theta_values_new = np.copy(theta_values)
                theta_values_new[j] = theta_proposal
                
                theta_t_new = _get_theta_t_from_state(k, taus, theta_values_new)
                log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf)
                
                log_prior_ratio = _log_pdf_normal(theta_proposal, PRIOR_THETA_MU, PRIOR_THETA_SIGMA) - \
                                  _log_pdf_normal(theta_current, PRIOR_THETA_MU, PRIOR_THETA_SIGMA)
                
                # Acceptance probability (Equation 6)
                log_alpha = (log_lik_new - log_lik_current) + log_prior_ratio
                
                if np.log(np.random.rand()) < log_alpha:
                    theta_values[j] = theta_proposal
                    log_lik_current = log_lik_new

        # --- Store Sample ---
        if iter_idx >= MCMC_BURN_IN:
            sample_idx = iter_idx - MCMC_BURN_IN
            k_samples[sample_idx] = k
            if k > 0:
                taus_samples[sample_idx, :k] = taus
            theta_samples[sample_idx, :(k + 1)] = theta_values

    return k_samples, taus_samples, theta_samples

# ==============================================================================
# MAIN WRAPPER AND BENCHMARK FUNCTIONS
# ==============================================================================

def run_rjmcmc(data):
    """
    Main Python wrapper for the Numba-accelerated RJMCMC sampler.
    This function handles data preparation and post-processing of the results.
    """
    # Pre-calculate the delay PMF from the configured distribution
    s = np.arange(0, T)
    delay_pmf = DELAY_DIST.pdf(s)
    delay_pmf /= np.sum(delay_pmf) # Normalize to ensure it's a valid PMF
    
    # Run the accelerated Numba sampler
    k_samples, taus_samples, _ = _rjmcmc_sampler_numba(
        data["deaths"], data["cases"], delay_pmf
    )
    
    # --- Post-process samples to find the posterior mode for k and taus ---
    # Estimate k as the mode (most frequent value) of the posterior samples
    if len(k_samples) == 0: return {"k": 0, "taus": []}
    est_k = int(Counter(k_samples).most_common(1)[0][0])
    
    # For the estimated k, find the most frequent set of changepoints
    est_taus = []
    if est_k > 0:
        # Filter samples where k equals the estimated k
        relevant_taus = taus_samples[k_samples == est_k, :est_k]
        # Convert rows to tuples to make them hashable for counting
        tau_tuples = [tuple(row) for row in relevant_taus]
        if tau_tuples:
            # Find the most common tuple of changepoint locations
            est_taus = sorted(list(Counter(tau_tuples).most_common(1)[0][0]))

    return {"k": est_k, "taus": est_taus}

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
    min_cost = algo.cost.sum_of_costs(algo.predict(n_bkps=0)) 

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
