# methods.py

"""
This script contains the core estimation algorithms used in the simulation study.
It includes:
- The proposed RJMCMC sampler, accelerated with Numba.
- The rtaCFR fused lasso signal estimator, implemented with cvxpy.
- Wrappers for the benchmark methods (PELT and Binary Segmentation).
"""

import os
import numpy as np
from scipy.stats import norm
from scipy.special import expit
import ruptures as rpt
from numba import njit
from collections import Counter
import cvxpy as cp

from config import (MCMC_ITER, MCMC_BURN_IN, K_MAX, DELAY_DIST,
                    PRIOR_K_GEOMETRIC_P, PRIOR_THETA_MU, PRIOR_THETA_SIGMA,
                    PROPOSAL_U_SIGMA, PROPOSAL_THETA_SIGMA, PROPOSAL_MOVE_WINDOW,
                    SIGNAL_CACHE_DIR)

# ==============================================================================
# ACCELERATED HELPER FUNCTIONS (NUMBA JIT)
# These functions are compiled to machine code by Numba for maximum speed.
# ==============================================================================

@njit
def _sigmoid(x):
    """Numba-compatible sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


@njit
def _calculate_log_likelihood(deaths, cases, theta_t, delay_pmf, T):
    """Calculates the log-likelihood of the death series."""
    p_t = _sigmoid(theta_t)
    signal = cases * p_t
    expected_deaths = np.zeros(T)
    for i in range(T):
        for j in range(i + 1):
            if j < len(delay_pmf):
                expected_deaths[i] += signal[i - j] * delay_pmf[j]
    log_lik = 0.0
    for t in range(T):
        mu = max(1e-9, expected_deaths[t])
        log_lik += deaths[t] * np.log(mu) - mu
    return log_lik


@njit
def _get_theta_t_from_state(k, taus, theta_values, T):
    """Constructs the full theta(t) time series from a given state."""
    theta_t = np.zeros(T)
    boundaries = np.array([0] + list(taus) + [T])
    for i in range(k + 1):
        theta_t[boundaries[i]:boundaries[i+1]] = theta_values[i]
    return theta_t


@njit
def _log_pdf_normal(x, mu, sigma):
    """Numba-compatible log PDF for a Normal distribution."""
    return -0.5 * np.log(2 * np.pi * sigma**2) - ((x - mu)**2) / (2 * sigma**2)


@njit
def _log_pmf_geometric(k, p):
    """Numba-compatible log PMF for a Geometric distribution."""
    if k < 0 or p <= 0 or p > 1:
        return -np.inf
    return k * np.log(1.0 - p) + np.log(p)


# ==============================================================================
# RJMCMC SAMPLER (NUMBA JIT ACCELERATED)
# ==============================================================================

# Note: The @njit decorator is commented out to allow the function to be
# pickled by the multiprocessing library. If running serially, uncommenting
# this line will provide a significant speedup.
# @njit
def _rjmcmc_sampler_numba(deaths, cases, delay_pmf, T, p_geom, theta_mu, theta_sigma, u_sigma, theta_prop_sigma, move_window):
    """The core RJMCMC sampler loop, parameterized for sensitivity analysis."""
    k = 0
    taus = np.array([], dtype=np.int64)
    theta_values = np.array([np.random.normal(theta_mu, theta_sigma)])
    k_samples = np.zeros(MCMC_ITER, dtype=np.int64)
    taus_samples = np.full((MCMC_ITER, K_MAX), -1, dtype=np.int64)
    theta_samples = np.full((MCMC_ITER, K_MAX + 1), np.nan, dtype=np.float64)

    for iter_idx in range(MCMC_ITER + MCMC_BURN_IN):
        theta_t_current = _get_theta_t_from_state(k, taus, theta_values, T)
        log_lik_current = _calculate_log_likelihood(deaths, cases, theta_t_current, delay_pmf, T)
        
        log_prior_k_current = _log_pmf_geometric(k, p_geom)
        log_prior_theta_current = 0.0
        for val in theta_values:
            log_prior_theta_current += _log_pdf_normal(val, theta_mu, theta_sigma)

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

        if move_type == "birth":
            possible_cps = np.array([i for i in range(1, T) if i not in taus])
            if len(possible_cps) > 0:
                tau_star = np.random.choice(possible_cps)
                seg_idx = np.searchsorted(taus, tau_star)
                theta_j = theta_values[seg_idx]
                u_aux = np.random.normal(0, u_sigma)
                theta1_star, theta2_star = theta_j - u_aux, theta_j + u_aux
                k_new, taus_new = k + 1, np.sort(np.append(taus, tau_star))
                theta_values_new = np.concatenate((theta_values[:seg_idx], np.array([theta1_star, theta2_star]), theta_values[seg_idx+1:]))
                theta_t_new = _get_theta_t_from_state(k_new, taus_new, theta_values_new, T)
                log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf, T)
                log_prior_k_new = _log_pmf_geometric(k_new, p_geom)
                log_prior_theta_new = 0.0
                for val in theta_values_new:
                    log_prior_theta_new += _log_pdf_normal(val, theta_mu, theta_sigma)
                log_lik_ratio = log_lik_new - log_lik_current
                log_prior_ratio = (log_prior_k_new - log_prior_k_current) + (log_prior_theta_new - log_prior_theta_current)
                log_proposal_ratio = -_log_pdf_normal(u_aux, 0, u_sigma)
                log_alpha = log_lik_ratio + log_prior_ratio + log_proposal_ratio
                if np.log(np.random.rand()) < log_alpha:
                    k, taus, theta_values = k_new, taus_new, theta_values_new

        elif move_type == "death":
            idx_to_remove = np.random.randint(0, k)
            theta1, theta2 = theta_values[idx_to_remove], theta_values[idx_to_remove+1]
            theta_j_star = (theta1 + theta2) / 2.0
            u_aux = (theta2 - theta1) / 2.0
            k_new, taus_new = k - 1, np.delete(taus, idx_to_remove)
            theta_values_new = np.concatenate((theta_values[:idx_to_remove], np.array([theta_j_star]), theta_values[idx_to_remove+2:]))
            theta_t_new = _get_theta_t_from_state(k_new, taus_new, theta_values_new, T)
            log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf, T)
            log_prior_k_new = _log_pmf_geometric(k_new, p_geom)
            log_prior_theta_new = 0.0
            for val in theta_values_new:
                log_prior_theta_new += _log_pdf_normal(val, theta_mu, theta_sigma)
            log_lik_ratio = log_lik_new - log_lik_current
            log_prior_ratio = (log_prior_k_new - log_prior_k_current) + (log_prior_theta_new - log_prior_theta_current)
            log_proposal_ratio = _log_pdf_normal(u_aux, 0, u_sigma)
            log_alpha = log_lik_ratio + log_prior_ratio + log_proposal_ratio
            if np.log(np.random.rand()) < log_alpha:
                k, taus, theta_values = k_new, taus_new, theta_values_new

        elif move_type == "move" and k > 0:
            idx_to_move = np.random.randint(0, k)
            tau_current = taus[idx_to_move]
            lower_prop = max(1, tau_current - move_window)
            upper_prop = min(T - 1, tau_current + move_window)
            if lower_prop < upper_prop:
                tau_new = np.random.randint(lower_prop, upper_prop + 1)
                lower_bound = taus[idx_to_move-1] + 1 if idx_to_move > 0 else 1
                upper_bound = taus[idx_to_move+1] - 1 if idx_to_move < k - 1 else T - 1
                if lower_bound <= tau_new <= upper_bound and tau_new != tau_current:
                    taus_new = np.copy(taus)
                    taus_new[idx_to_move] = tau_new
                    theta_t_new = _get_theta_t_from_state(k, np.sort(taus_new), theta_values, T)
                    log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf, T)
                    log_alpha = log_lik_new - log_lik_current
                    if np.log(np.random.rand()) < log_alpha:
                        taus = np.sort(taus_new)

        elif move_type == "update":
            for j in range(k + 1):
                theta_current = theta_values[j]
                theta_proposal = np.random.normal(theta_current, theta_prop_sigma)
                theta_values_new = np.copy(theta_values)
                theta_values_new[j] = theta_proposal
                theta_t_new = _get_theta_t_from_state(k, taus, theta_values_new, T)
                log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf, T)
                log_prior_ratio = _log_pdf_normal(theta_proposal, theta_mu, theta_sigma) - _log_pdf_normal(theta_current, theta_mu, theta_sigma)
                log_alpha = (log_lik_new - log_lik_current) + log_prior_ratio
                if np.log(np.random.rand()) < log_alpha:
                    theta_values[j] = theta_proposal
                    log_lik_current = log_lik_new

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

def run_rjmcmc(data, p_geom=PRIOR_K_GEOMETRIC_P, theta_sigma=PRIOR_THETA_SIGMA):
    """
    Main wrapper for the RJMCMC sampler. This version calculates summary
    statistics and discards the raw samples to save memory.
    """
    T_data = data["cases"].shape[0]
    delay_pmf = np.diff(DELAY_DIST.cdf(np.arange(T_data + 1)))
    
    k_samples, taus_samples, theta_samples = _rjmcmc_sampler_numba(
        data["deaths"], data["cases"], delay_pmf, T_data,
        p_geom, PRIOR_THETA_MU, theta_sigma,
        PROPOSAL_U_SIGMA, PROPOSAL_THETA_SIGMA, PROPOSAL_MOVE_WINDOW
    )
    
    # --- Post-processing ---
    # Calculate p(t) for each sample
    p_t_samples = np.zeros((MCMC_ITER, T_data))
    for i in range(MCMC_ITER):
        k, taus, thetas = k_samples[i], taus_samples[i, :k_samples[i]], theta_samples[i, :(k_samples[i] + 1)]
        theta_t_sample = _get_theta_t_from_state(k, taus, thetas, T_data)
        p_t_samples[i, :] = _sigmoid(theta_t_sample)
    
    # Calculate summary statistics
    p_t_mean = np.mean(p_t_samples, axis=0)
    p_t_lower_ci = np.percentile(p_t_samples, 2.5, axis=0)
    p_t_upper_ci = np.percentile(p_t_samples, 97.5, axis=0)
    
    est_k = int(Counter(k_samples).most_common(1)[0][0])
    est_taus = []
    if est_k > 0:
        relevant_taus = taus_samples[k_samples == est_k, :est_k]
        tau_tuples = [tuple(row) for row in relevant_taus]
        if tau_tuples:
            est_taus = sorted(list(Counter(tau_tuples).most_common(1)[0][0]))
            
    # Return ONLY the summarized results
    return {
        "k_est": est_k, 
        "taus_est": est_taus, 
        "p_t_hat": p_t_mean, # Use p_t_hat for consistency with benchmarks
        "p_t_lower_ci": p_t_lower_ci,
        "p_t_upper_ci": p_t_upper_ci
    }



def _run_rtacfr_fusedlasso_internal(data):
    """Internal function to get the fused lasso signal estimate using cvxpy."""
    ct, dt = data['cases'], data['deaths']
    N = len(ct)
    delay_pmf = np.diff(DELAY_DIST.cdf(np.arange(N + 1)))
    fmat = np.zeros((N, N))
    for i in range(N):
        fmat[i, :(i + 1)] = delay_pmf[:(i + 1)][::-1]
    Xmat = fmat @ np.diag(ct)
    p = cp.Variable(N)
    lambda_param = cp.Parameter(nonneg=True)
    loss = cp.sum_squares(dt - Xmat @ p)
    penalty = cp.norm1(p[1:] - p[:-1])
    objective = cp.Minimize(loss + lambda_param * penalty)
    constraints = [p >= 0, p <= 1]
    problem = cp.Problem(objective, constraints)
    lambda_grid = np.logspace(-2, 4, 30)
    min_bic, best_p_hat = np.inf, None
    for lambda_val in lambda_grid:
        lambda_param.value = lambda_val
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            if p.value is not None:
                p_hat = p.value
                rss = np.sum((dt - Xmat @ p_hat)**2)
                if rss < 1e-9:
                    continue
                num_params = len(np.unique(np.round(p_hat, 4)))
                bic = N * np.log(rss / N) + num_params * np.log(N)
                if bic < min_bic:
                    min_bic, best_p_hat = bic, p_hat
        except cp.error.SolverError:
            continue
    if best_p_hat is None:
        return np.zeros(N)
    return best_p_hat

def get_rtacfr_signal(data, scenario_name, rep_idx):
    """Calculates the fused lasso signal, using a cache to avoid re-computation."""
    cache_file = os.path.join(SIGNAL_CACHE_DIR, f"signal_scen={scenario_name.replace(' ', '-')}_rep={rep_idx}.npz")
    if os.path.exists(cache_file):
        return np.load(cache_file)['signal']
    p_hat = _run_rtacfr_fusedlasso_internal(data)
    np.savez(cache_file, signal=p_hat)
    return p_hat

def run_rtacfr(data, scenario_name, rep_idx):
    """
    Wrapper for the pure rtaCFR-fusedlasso method as a benchmark.
    This provides the raw smoothed signal for plotting.
    """
    signal = get_rtacfr_signal(data, scenario_name, rep_idx)
    diffs = np.diff(signal)
    taus_est = np.where(np.abs(diffs) > 1e-6)[0] + 1
    k_est = len(taus_est)
    return {"k_est": k_est, "taus_est": list(taus_est), "p_t_hat": signal}

def run_pelt(data, scenario_name, rep_idx):
    """Wrapper for PELT method applied to the cached rtaCFR signal."""
    T = data["cases"].shape[0]
    signal = get_rtacfr_signal(data, scenario_name, rep_idx)
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=np.log(T) * 2)
    p_t_hat = np.zeros(T)
    boundaries = [0] + result
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if end > start:
            p_t_hat[start:end] = np.mean(signal[start:end])
    return {"k_est": len(result)-1, "taus_est": result[:-1], "p_t_hat": p_t_hat}


def run_binseg(data, scenario_name, rep_idx):
    """Wrapper for Binary Segmentation method applied to the cached rtaCFR signal."""
    T = data["cases"].shape[0]
    signal = get_rtacfr_signal(data, scenario_name, rep_idx)
    algo = rpt.Binseg(model="rbf").fit(signal)
    penalty = np.log(T)
    best_k = 0
    min_cost = algo.cost.sum_of_costs(algo.predict(n_bkps=0))
    for k_candidate in range(1, K_MAX + 1):
        bkps_for_k = algo.predict(n_bkps=k_candidate)
        cost = algo.cost.sum_of_costs(bkps_for_k) + penalty * k_candidate
        if cost < min_cost:
            min_cost, best_k = cost, k_candidate
    result = algo.predict(n_bkps=best_k)
    p_t_hat = np.zeros(T)
    boundaries = [0] + result
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if end > start:
            p_t_hat[start:end] = np.mean(signal[start:end])
    return {"k_est": best_k, "taus_est": result[:-1], "p_t_hat": p_t_hat}
