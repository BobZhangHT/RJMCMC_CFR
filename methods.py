# methods.py

"""
This script contains the core estimation algorithms used in the simulation study.
It includes:
- The proposed RJMCMC sampler, accelerated with Numba.
- Three different methods for summarizing the posterior distribution of changepoints.
- The rtaCFR fused-lasso signal estimator with an in-process ADMM backend.
- Wrappers for the benchmark methods (PELT and Binary Segmentation).
"""

import hashlib
import os
import re
import numpy as np
from scipy import linalg
from scipy.stats import norm
from scipy.special import expit
from scipy.signal import find_peaks
from numba import njit
from collections import Counter

from config import (MCMC_ITER, MCMC_BURN_IN, K_MAX, DELAY_DIST,
                    PRIOR_K_GEOMETRIC_P, PRIOR_THETA_MU, PRIOR_THETA_SIGMA,
                    PROPOSAL_U_SIGMA, PROPOSAL_THETA_SIGMA, PROPOSAL_MOVE_WINDOW,
                    SIGNAL_CACHE_DIR)

try:
    from _rjmcmc_c import sample as _rjmcmc_sample_c
    _C_BACKEND_IMPORT_ERROR = None
except ImportError as exc:
    _rjmcmc_sample_c = None
    _C_BACKEND_IMPORT_ERROR = exc

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
    """
    Calculates the log-likelihood of the death series given the latent CFR process.
    This is the core of the model's objective function.
    """
    p_t = _sigmoid(theta_t)
    signal = cases * p_t
    expected_deaths = np.zeros(T)
    # Convolve the signal of new fatal cases with the delay distribution
    for i in range(T):
        for j in range(i + 1):
            if j < len(delay_pmf):
                expected_deaths[i] += signal[i - j] * delay_pmf[j]
    
    # Calculate Poisson log-likelihood
    log_lik = 0.0
    for t in range(T):
        mu = max(1e-9, expected_deaths[t]) # Ensure mu is positive
        # Standard Poisson log-likelihood: deaths[t] * log(mu) - mu - log(deaths[t]!)
        # We can ignore the factorial term as it's constant across models for BIC comparison
        log_lik += deaths[t] * np.log(mu) - mu
    return log_lik


@njit
def _get_theta_t_from_state(k, taus, theta_values, T):
    """Constructs the full theta(t) time series from a given state (k, taus, theta_values)."""
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
def _move_probabilities(k, max_k):
    """Return (birth, death, move, update) probabilities at a state boundary."""
    if max_k == 0:
        return 0.0, 0.0, 0.0, 1.0
    if k == 0:
        return 0.25, 0.0, 0.0, 0.75
    if k == max_k:
        return 0.0, 0.25, 0.25, 0.50
    return 0.25, 0.25, 0.25, 0.25


def _relocation_support(taus, idx, T, move_window):
    """Return the valid relocation interval and its non-self support size."""
    tau_current = int(taus[idx])
    lower = max(2, tau_current - move_window)
    upper = min(T - 1, tau_current + move_window)
    if idx > 0:
        lower = max(lower, int(taus[idx - 1]) + 1)
    if idx < len(taus) - 1:
        upper = min(upper, int(taus[idx + 1]) - 1)
    return lower, upper, max(0, upper - lower)


def _relocation_mixture_probability(taus, idx, T, move_window, candidate,
                                    global_move_prob):
    """Probability of one relocation candidate under a local/global mixture."""
    local_lower, local_upper, local_size = _relocation_support(
        taus, idx, T, move_window
    )
    global_lower, global_upper, global_size = _relocation_support(
        taus, idx, T, T
    )
    probability = 0.0
    if global_size > 0 and global_lower <= candidate <= global_upper:
        probability += global_move_prob / global_size
    if local_size > 0 and local_lower <= candidate <= local_upper:
        probability += (1.0 - global_move_prob) / local_size
    return probability


def _rjmcmc_sampler_numba(deaths, cases, delay_pmf, T, p_geom, theta_mu, theta_sigma, u_sigma, theta_prop_sigma, move_window, global_move_prob,
                          iterations=None, burn_in=None, max_k=None,
                          initial_state=None, return_diagnostics=False):
    """The core RJMCMC sampler loop, parameterized for sensitivity analysis."""
    mcmc_iter = MCMC_ITER if iterations is None else int(iterations)
    mcmc_burn_in = MCMC_BURN_IN if burn_in is None else int(burn_in)
    max_k = K_MAX if max_k is None else int(max_k)

    if mcmc_iter < 0 or mcmc_burn_in < 0 or max_k < 0 or max_k >= T:
        raise ValueError("invalid RJMCMC iteration or state-space limits")

    # Initialize state variables
    if initial_state is None:
        k = 0
        taus = np.array([], dtype=np.int64)
        theta_values = np.array([np.random.normal(theta_mu, theta_sigma)])
    else:
        k = int(initial_state[0])
        taus = np.asarray(initial_state[1], dtype=np.int64).copy()
        theta_values = np.asarray(initial_state[2], dtype=np.float64).copy()
        if k < 0 or k > max_k or len(taus) != k or len(theta_values) != k + 1:
            raise ValueError("initial_state must contain k taus and k+1 theta values")
        if k > 1 and np.any(np.diff(taus) <= 0):
            raise ValueError("initial_state changepoints must be strictly increasing")
        if k > 0 and (np.min(taus) < 2 or np.max(taus) >= T):
            raise ValueError("initial_state changepoints must lie in [2, T-1]")
    
    # Pre-allocate arrays to store posterior samples
    k_samples = np.zeros(mcmc_iter, dtype=np.int64)
    taus_samples = np.full((mcmc_iter, max_k), -1, dtype=np.int64)
    theta_samples = np.full((mcmc_iter, max_k + 1), np.nan, dtype=np.float64)
    proposed = np.zeros(4, dtype=np.int64)
    accepted = np.zeros(4, dtype=np.int64)

    for iter_idx in range(mcmc_iter + mcmc_burn_in):
        # Calculate current log-likelihood and priors
        theta_t_current = _get_theta_t_from_state(k, taus, theta_values, T)
        log_lik_current = _calculate_log_likelihood(deaths, cases, theta_t_current, delay_pmf, T)
        log_prior_k_current = _log_pmf_geometric(k, p_geom)
        log_prior_theta_current = np.sum(_log_pdf_normal(theta_values, theta_mu, theta_sigma))

        # Randomly choose a move type
        u_move = np.random.rand()
        p_birth, p_death, p_move, p_update = _move_probabilities(k, max_k)
        if u_move < p_birth:
            move_type, move_idx = "birth", 0
        elif u_move < p_birth + p_death:
            move_type, move_idx = "death", 1
        elif u_move < p_birth + p_death + p_move:
            move_type, move_idx = "move", 2
        else:
            move_type, move_idx = "update", 3
        proposed[move_idx] += (k + 1) if move_type == "update" else 1

        # --- RJMCMC Move Types ---
        if move_type == "birth":
            # Propose adding a new changepoint
            possible_cps = np.array([i for i in range(2, T) if i not in taus])
            if len(possible_cps) > 0:
                tau_star = np.random.choice(possible_cps)
                seg_idx = np.searchsorted(taus, tau_star)
                theta_j = theta_values[seg_idx]
                u_aux = np.random.normal(0, u_sigma)
                theta1_star, theta2_star = theta_j - u_aux, theta_j + u_aux
                
                # New state
                k_new, taus_new = k + 1, np.sort(np.append(taus, tau_star))
                theta_values_new = np.concatenate((theta_values[:seg_idx], np.array([theta1_star, theta2_star]), theta_values[seg_idx+1:]))
                
                # Metropolis-Hastings-Green acceptance probability
                theta_t_new = _get_theta_t_from_state(k_new, taus_new, theta_values_new, T)
                log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf, T)
                log_prior_k_new = _log_pmf_geometric(k_new, p_geom)
                log_prior_theta_new = np.sum(_log_pdf_normal(theta_values_new, theta_mu, theta_sigma))
                
                log_lik_ratio = log_lik_new - log_lik_current
                log_prior_ratio = (log_prior_k_new - log_prior_k_current) + (log_prior_theta_new - log_prior_theta_current)
                p_birth_new = _move_probabilities(k_new, max_k)[0]
                p_death_new = _move_probabilities(k_new, max_k)[1]
                # Uniform location-prior and location-selection counts cancel.
                log_proposal_ratio = (
                    np.log(p_death_new / p_birth) -
                    _log_pdf_normal(u_aux, 0, u_sigma) + np.log(2)
                )
                
                log_alpha = log_lik_ratio + log_prior_ratio + log_proposal_ratio
                if np.log(np.random.rand()) < log_alpha:
                    k, taus, theta_values = k_new, taus_new, theta_values_new
                    accepted[move_idx] += 1

        elif move_type == "death":
            # Propose removing an existing changepoint
            idx_to_remove = np.random.randint(0, k)
            theta1, theta2 = theta_values[idx_to_remove], theta_values[idx_to_remove+1]
            theta_j_star = (theta1 + theta2) / 2.0
            u_aux = (theta2 - theta1) / 2.0
            
            # New state
            k_new, taus_new = k - 1, np.delete(taus, idx_to_remove)
            theta_values_new = np.concatenate((theta_values[:idx_to_remove], np.array([theta_j_star]), theta_values[idx_to_remove+2:]))
            
            # Acceptance probability
            theta_t_new = _get_theta_t_from_state(k_new, taus_new, theta_values_new, T)
            log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf, T)
            log_prior_k_new = _log_pmf_geometric(k_new, p_geom)
            log_prior_theta_new = np.sum(_log_pdf_normal(theta_values_new, theta_mu, theta_sigma))
            
            log_lik_ratio = log_lik_new - log_lik_current
            log_prior_ratio = (log_prior_k_new - log_prior_k_current) + (log_prior_theta_new - log_prior_theta_current)
            p_birth_new = _move_probabilities(k_new, max_k)[0]
            p_death_new = _move_probabilities(k_new, max_k)[1]
            log_proposal_ratio = (
                np.log(p_birth_new / p_death) +
                _log_pdf_normal(u_aux, 0, u_sigma) - np.log(2)
            )
            
            log_alpha = log_lik_ratio + log_prior_ratio + log_proposal_ratio
            if np.log(np.random.rand()) < log_alpha:
                k, taus, theta_values = k_new, taus_new, theta_values_new
                accepted[move_idx] += 1

        elif move_type == "move" and k > 0:
            # Mix local relocation with occasional global jumps between neighbors.
            idx_to_move = np.random.randint(0, k)
            tau_current = taus[idx_to_move]
            use_global = np.random.rand() < global_move_prob
            active_window = T if use_global else move_window
            lower_prop, upper_prop, support_size = _relocation_support(
                taus, idx_to_move, T, active_window
            )
            if support_size > 0:
                proposal_idx = np.random.randint(0, support_size)
                tau_new = lower_prop + proposal_idx
                if tau_new >= tau_current:
                    tau_new += 1
                if lower_prop <= tau_new <= upper_prop and tau_new != tau_current:
                    taus_new = np.copy(taus)
                    taus_new[idx_to_move] = tau_new
                    theta_t_new = _get_theta_t_from_state(
                        k, np.sort(taus_new), theta_values, T
                    )
                    log_lik_new = _calculate_log_likelihood(deaths, cases, theta_t_new, delay_pmf, T)
                    q_forward = _relocation_mixture_probability(
                        taus, idx_to_move, T, move_window, tau_new,
                        global_move_prob
                    )
                    q_reverse = _relocation_mixture_probability(
                        np.sort(taus_new), idx_to_move, T, move_window,
                        tau_current, global_move_prob
                    )
                    log_alpha = (
                        log_lik_new - log_lik_current +
                        np.log(q_reverse / q_forward)
                    )
                    if np.log(np.random.rand()) < log_alpha:
                        taus = np.sort(taus_new)
                        accepted[move_idx] += 1

        elif move_type == "update":
            # Update the value of a theta parameter
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
                    accepted[move_idx] += 1

        # Store the sample after the burn-in period
        if iter_idx >= mcmc_burn_in:
            sample_idx = iter_idx - mcmc_burn_in
            k_samples[sample_idx] = k
            if k > 0:
                taus_samples[sample_idx, :k] = taus
            theta_samples[sample_idx, :(k + 1)] = theta_values
            
    if return_diagnostics:
        return k_samples, taus_samples, theta_samples, {
            "proposed": proposed,
            "accepted": accepted,
            "move_names": ("birth", "death", "move", "update"),
        }
    return k_samples, taus_samples, theta_samples


def c_backend_available():
    """Return whether the optional compiled C sampler is importable."""
    return _rjmcmc_sample_c is not None


def available_backends():
    """Return the sampler backends available in the current environment."""
    return ("numba", "c") if c_backend_available() else ("numba",)


def _rjmcmc_sampler_c(deaths, cases, delay_pmf, T, p_geom, theta_mu,
                      theta_sigma, u_sigma, theta_prop_sigma, move_window,
                      global_move_prob,
                      seed=None, iterations=None, burn_in=None, max_k=None,
                      initial_state=None, return_diagnostics=False):
    """Call the optional C kernel using the same sampler contract as Numba."""
    if _rjmcmc_sample_c is None:
        detail = " Build it with `python setup.py build_ext --inplace`."
        raise RuntimeError(
            "The C RJMCMC backend is not available." + detail
        ) from _C_BACKEND_IMPORT_ERROR

    if seed is None:
        seed = int(np.random.randint(1, np.iinfo(np.int64).max))
    mcmc_iter = MCMC_ITER if iterations is None else int(iterations)
    mcmc_burn_in = MCMC_BURN_IN if burn_in is None else int(burn_in)
    state_k = -1
    state_taus = np.empty(0, dtype=np.int64)
    state_theta = np.empty(0, dtype=np.float64)
    if initial_state is not None:
        state_k, state_taus, state_theta = initial_state
    result = _rjmcmc_sample_c(
        np.asarray(deaths, dtype=np.float64),
        np.asarray(cases, dtype=np.float64),
        np.asarray(delay_pmf, dtype=np.float64),
        float(p_geom), float(theta_mu), float(theta_sigma),
        float(u_sigma), float(theta_prop_sigma), int(move_window),
        float(global_move_prob),
        int(mcmc_iter), int(mcmc_burn_in),
        int(K_MAX if max_k is None else max_k), int(seed), int(state_k),
        np.asarray(state_taus, dtype=np.int64),
        np.asarray(state_theta, dtype=np.float64), int(return_diagnostics)
    )
    return result


def _coerce_initial_state(initial_state, T, max_k):
    """Normalize the public initial-state form for both sampler kernels."""
    if initial_state is None:
        return None
    if isinstance(initial_state, dict):
        k = initial_state.get("k")
        taus = initial_state.get("taus")
        theta = initial_state.get("theta_values", initial_state.get("theta"))
    else:
        try:
            k, taus, theta = initial_state
        except (TypeError, ValueError):
            raise ValueError("initial_state must be (k, taus, theta_values) or a mapping")
    if k is None or taus is None or theta is None:
        raise ValueError("initial_state must contain k, taus, and theta_values")
    k = int(k)
    taus = np.asarray(taus, dtype=np.int64)
    theta = np.asarray(theta, dtype=np.float64)
    if k < 0 or k > max_k or taus.ndim != 1 or theta.ndim != 1:
        raise ValueError("initial_state has invalid dimensions")
    if len(taus) != k or len(theta) != k + 1:
        raise ValueError("initial_state must contain k taus and k+1 theta values")
    if k and (np.any(taus < 2) or np.any(taus >= T) or np.any(np.diff(taus) <= 0)):
        raise ValueError("initial_state changepoints must be strictly increasing in [2, T-1]")
    return k, taus.copy(), theta.copy()


# ==============================================================================
# MAIN WRAPPER AND BENCHMARK FUNCTIONS
# ==============================================================================

def posterior_inclusion_probabilities(k_samples, taus_samples, T):
    """
    Computes the Posterior Inclusion Probability (PIP) for each time point.
    PIP at time t is the posterior probability that a changepoint occurs at t.
    """
    hits = np.zeros(T, dtype=np.int64)
    S = len(k_samples)
    for s in range(S):
        k = int(k_samples[s])
        if k <= 0: continue
        taus = taus_samples[s, :k]
        for tau in taus:
            if 0 <= tau < T:
                hits[int(tau)] += 1
    return hits / float(S)

def pick_cps_from_pip(pip, w=7, min_height=None, max_k=None, min_prominence=0.0):
    """
    Identifies changepoints by finding peaks in the PIP curve.
    """
    T = len(pip)
    height = (0.5 * pip.max()) if min_height is None else min_height
    peaks, _ = find_peaks(pip, distance=w, height=height, prominence=min_prominence)
    if max_k is not None and len(peaks) > max_k:
        order = np.argsort(pip[peaks])[::-1][:max_k]
        peaks = np.sort(peaks[order])
    return peaks.astype(int)

def rank_conditioned_cp_summary(k_samples, taus_samples):
    """
    Summarizes changepoints by conditioning on the posterior mode of K.
    It then finds the median location for each ranked changepoint.
    """
    K_star = Counter(k_samples).most_common(1)[0][0]
    subset = np.where(k_samples == K_star)[0]
    if K_star == 0 or len(subset) == 0:
        return []
    taus_ranked = np.sort(taus_samples[subset, :K_star], axis=1)
    cps = np.median(taus_ranked, axis=0).astype(int).tolist()
    return cps

def run_rjmcmc(data, p_geom=PRIOR_K_GEOMETRIC_P, theta_sigma=PRIOR_THETA_SIGMA,
               summary_method='mode', backend='numba', seed=None,
               iterations=None, burn_in=None, initial_state=None,
               diagnostics=False, return_samples=False, delay_dist=None,
               move_window=None, theta_prop_sigma=None,
               global_move_prob=0.0,
               max_k=None, u_prop_sigma=None):
    """
    Main wrapper for the RJMCMC sampler. Returns summarized statistics.

    ``backend`` may be ``"numba"`` (the historical default), ``"c"`` for
    the optional compiled kernel, or ``"auto"`` to prefer C and fall back to
    Numba when the extension has not been built. ``seed`` is optional and is
    useful for reproducible backend comparisons; the two backends do not
    promise identical random-number streams. ``iterations``, ``burn_in``,
    ``initial_state``, and ``diagnostics`` are optional controls intended for
    reproducible checks and sampler investigations.
    """
    if backend not in {"numba", "c", "auto"}:
        raise ValueError("backend must be one of: 'numba', 'c', or 'auto'")
    selected_backend = backend
    if backend == "auto":
        selected_backend = "c" if c_backend_available() else "numba"

    T_data = data["cases"].shape[0]
    mcmc_iter = MCMC_ITER if iterations is None else int(iterations)
    mcmc_burn_in = MCMC_BURN_IN if burn_in is None else int(burn_in)
    max_k = K_MAX if max_k is None else int(max_k)
    if max_k < 0 or max_k >= T_data:
        raise ValueError("max_k must be between 0 and T - 1")
    active_move_window = PROPOSAL_MOVE_WINDOW if move_window is None else int(move_window)
    active_theta_prop_sigma = (
        PROPOSAL_THETA_SIGMA if theta_prop_sigma is None else float(theta_prop_sigma)
    )
    active_u_prop_sigma = (
        PROPOSAL_U_SIGMA if u_prop_sigma is None else float(u_prop_sigma)
    )
    if active_move_window < 1:
        raise ValueError("move_window must be at least 1")
    if active_theta_prop_sigma <= 0:
        raise ValueError("theta_prop_sigma must be positive")
    if active_u_prop_sigma <= 0:
        raise ValueError("u_prop_sigma must be positive")
    active_global_move_prob = float(global_move_prob)
    if active_global_move_prob < 0.0 or active_global_move_prob > 1.0:
        raise ValueError("global_move_prob must be between 0 and 1")
    normalized_initial_state = _coerce_initial_state(initial_state, T_data, max_k)
    active_delay_dist = DELAY_DIST if delay_dist is None else delay_dist
    delay_pmf = np.diff(active_delay_dist.cdf(np.arange(T_data + 1)))

    if selected_backend == "c":
        sampler_result = _rjmcmc_sampler_c(
            data["deaths"], data["cases"], delay_pmf, T_data,
            p_geom, PRIOR_THETA_MU, theta_sigma,
            active_u_prop_sigma, active_theta_prop_sigma, active_move_window,
            active_global_move_prob,
            seed=seed, iterations=mcmc_iter, burn_in=mcmc_burn_in,
            max_k=max_k, initial_state=normalized_initial_state,
            return_diagnostics=diagnostics
        )
    else:
        if seed is not None:
            np.random.seed(seed)
        sampler_result = _rjmcmc_sampler_numba(
            data["deaths"], data["cases"], delay_pmf, T_data,
            p_geom, PRIOR_THETA_MU, theta_sigma,
            active_u_prop_sigma, active_theta_prop_sigma, active_move_window,
            active_global_move_prob,
            iterations=mcmc_iter, burn_in=mcmc_burn_in, max_k=max_k,
            initial_state=normalized_initial_state,
            return_diagnostics=diagnostics
        )
    if diagnostics:
        k_samples, taus_samples, theta_samples, sampler_diagnostics = sampler_result
    else:
        k_samples, taus_samples, theta_samples = sampler_result
    
    # --- Post-processing ---
    p_t_samples = np.zeros((mcmc_iter, T_data))
    for i in range(mcmc_iter):
        k, taus, thetas = k_samples[i], taus_samples[i, :k_samples[i]], theta_samples[i, :(k_samples[i] + 1)]
        theta_t_sample = _get_theta_t_from_state(k, taus, thetas, T_data)
        p_t_samples[i, :] = _sigmoid(theta_t_sample)
    
    p_t_mean = np.mean(p_t_samples, axis=0)
    p_t_lower_ci = np.percentile(p_t_samples, 2.5, axis=0)
    p_t_upper_ci = np.percentile(p_t_samples, 97.5, axis=0)
    
    # --- Calculate all three changepoint summaries ---
    # 1. Posterior Mode of K (original method)
    k_est_mode = int(Counter(k_samples).most_common(1)[0][0])
    taus_est_mode = []
    if k_est_mode > 0:
        relevant_taus = taus_samples[k_samples == k_est_mode, :k_est_mode]
        tau_tuples = [tuple(row) for row in relevant_taus]
        if tau_tuples:
            taus_est_mode = sorted(list(Counter(tau_tuples).most_common(1)[0][0]))

    # 2. Posterior Inclusion Probability (PIP)
    pip = posterior_inclusion_probabilities(k_samples, taus_samples, T_data)
    taus_est_pip = pick_cps_from_pip(pip)

    # 3. Rank-Conditioned Summary
    taus_est_cond = rank_conditioned_cp_summary(k_samples, taus_samples)

    all_estimates = {
        'pip': taus_est_pip,
        'mode': taus_est_mode,
        'cond': taus_est_cond
    }
    
    # Set the primary output based on the chosen method (default is mode)
    final_taus = all_estimates.get(summary_method, taus_est_mode)
    final_k = len(final_taus)
            
    result = {
        "k_est": final_k, 
        "taus_est": final_taus, 
        "p_t_hat": p_t_mean,
        "p_t_lower_ci": p_t_lower_ci,
        "p_t_upper_ci": p_t_upper_ci,
        "taus_est_pip": taus_est_pip,
        "taus_est_mode": taus_est_mode,
        "taus_est_cond": taus_est_cond,
        "pip_array": pip,
        "backend": selected_backend,
    }
    if diagnostics:
        result["sampler_diagnostics"] = sampler_diagnostics
    if return_samples:
        result["samples"] = {
            "k": k_samples,
            "taus": taus_samples,
            "theta": theta_samples,
            "p_t": p_t_samples,
        }
    return result

def _rtacfr_design_matrix(data, delay_dist=None):
    """Build the rtaCFR convolution design matrix for an assumed delay."""
    try:
        ct = np.asarray(data["cases"], dtype=np.float64)
        dt = np.asarray(data["deaths"], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        raise ValueError("data must contain one-dimensional 'cases' and 'deaths'") from exc
    if ct.ndim != 1 or dt.ndim != 1 or len(ct) != len(dt) or len(ct) == 0:
        raise ValueError("cases and deaths must be non-empty one-dimensional arrays of equal length")
    if not np.isfinite(ct).all() or not np.isfinite(dt).all():
        raise ValueError("cases and deaths must contain only finite values")

    active_delay_dist = DELAY_DIST if delay_dist is None else delay_dist
    delay_cdf = np.asarray(
        active_delay_dist.cdf(np.arange(len(ct) + 1)), dtype=np.float64
    )
    delay_pmf = np.diff(delay_cdf)
    if delay_pmf.shape != (len(ct),) or not np.isfinite(delay_pmf).all():
        raise ValueError("delay_dist.cdf must produce finite values on the requested grid")
    if np.any(delay_pmf < -1e-12):
        raise ValueError("delay_dist.cdf must be nondecreasing")
    delay_pmf = np.maximum(delay_pmf, 0.0)

    n_obs = len(ct)
    fmat = np.zeros((n_obs, n_obs), dtype=np.float64)
    for i in range(n_obs):
        fmat[i, :i + 1] = delay_pmf[:i + 1][::-1]
    # Multiplication by diag(ct) without materializing a second dense matrix.
    xmat = fmat * ct[np.newaxis, :]
    return ct, dt, delay_pmf, xmat


def _rtacfr_difference_matrix(n_obs):
    """Return the first-difference matrix used by the fused-lasso penalty."""
    difference = np.zeros((max(0, n_obs - 1), n_obs), dtype=np.float64)
    if n_obs > 1:
        rows = np.arange(n_obs - 1)
        difference[rows, rows] = -1.0
        difference[rows, rows + 1] = 1.0
    return difference


def _rtacfr_difference_apply(values):
    """Apply first differences without materializing a dense matrix product."""
    values = np.asarray(values)
    return values[1:] - values[:-1]


def _rtacfr_difference_transpose(values, n_obs):
    """Apply the transpose of the first-difference matrix in linear time."""
    values = np.asarray(values)
    result = np.zeros(n_obs, dtype=np.float64)
    if n_obs > 1:
        result[0] = -values[0]
        result[1:-1] = values[:-1] - values[1:]
        result[-1] = values[-1]
    return result


def _rtacfr_soft_threshold(values, threshold):
    """Apply the elementwise proximal map of the l1 norm."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _rtacfr_admm_solve(
    xmat,
    dt,
    difference,
    factor,
    rho,
    lambda_val,
    state=None,
    rhs_data=None,
    max_iter=5000,
    abs_tol=1e-5,
    rel_tol=1e-4,
):
    """Solve one boxed fused-lasso problem with scaled ADMM.

    The split variables are ``z_d = D p`` and ``z_p = p``.  The former has
    the soft-thresholding update and the latter has the box projection.  The
    factorization of the p-update is supplied by the caller so it can be
    reused for every lambda value.
    """
    n_obs = len(dt)
    n_diff = difference.shape[0]
    if rhs_data is None:
        rhs_data = 2.0 * (xmat.T @ dt)
    else:
        rhs_data = np.asarray(rhs_data, dtype=np.float64)
        if rhs_data.shape != (n_obs,):
            raise ValueError("rhs_data must have one entry per signal value")
    if state is None:
        p = linalg.cho_solve(factor, rhs_data, check_finite=False)
        p = np.clip(p, 0.0, 1.0)
        z_d = difference @ p
        z_p = p.copy()
        u_d = np.zeros(n_diff, dtype=np.float64)
        u_p = np.zeros(n_obs, dtype=np.float64)
    else:
        p = np.asarray(state["p"], dtype=np.float64).copy()
        z_d = np.asarray(state["z_d"], dtype=np.float64).copy()
        z_p = np.asarray(state["z_p"], dtype=np.float64).copy()
        u_d = np.asarray(state["u_d"], dtype=np.float64).copy()
        u_p = np.asarray(state["u_p"], dtype=np.float64).copy()

    if (
        p.shape != (n_obs,) or z_d.shape != (n_diff,) or z_p.shape != (n_obs,)
        or u_d.shape != (n_diff,) or u_p.shape != (n_obs,)
    ):
        raise ValueError("invalid ADMM warm-start state")

    objective_history = []
    converged = False
    primal_residual = np.inf
    dual_residual = np.inf
    for iteration in range(1, int(max_iter) + 1):
        rhs = rhs_data + rho * (
            _rtacfr_difference_transpose(z_d - u_d, n_obs) + z_p - u_p
        )
        p = linalg.cho_solve(factor, rhs, check_finite=False)

        z_d_old = z_d.copy()
        z_p_old = z_p.copy()
        difference_p = _rtacfr_difference_apply(p)
        z_d = _rtacfr_soft_threshold(difference_p + u_d, lambda_val / rho)
        z_p = np.clip(p + u_p, 0.0, 1.0)

        residual_d = difference_p - z_d
        residual_p = p - z_p
        u_d += residual_d
        u_p += residual_p

        primal_vector = np.concatenate((residual_d, residual_p))
        primal_residual = linalg.norm(primal_vector)
        dual_vector = (
            _rtacfr_difference_transpose(z_d - z_d_old, n_obs)
            + z_p - z_p_old
        )
        dual_residual = rho * linalg.norm(dual_vector)
        ap_norm = np.sqrt(np.sum(difference_p ** 2) + np.sum(p ** 2))
        z_norm = np.sqrt(np.sum(z_d ** 2) + np.sum(z_p ** 2))
        eps_primal = np.sqrt(n_diff + n_obs) * abs_tol + rel_tol * max(
            ap_norm, z_norm
        )
        eps_dual = np.sqrt(n_obs) * abs_tol + rel_tol * rho * linalg.norm(
            _rtacfr_difference_transpose(u_d, n_obs) + u_p
        )
        residual = dt - xmat @ p
        objective_history.append(
            float(np.sum(residual ** 2) + lambda_val * np.sum(np.abs(difference_p)))
        )
        if not (
            np.isfinite(p).all()
            and np.isfinite(z_d).all()
            and np.isfinite(z_p).all()
            and np.isfinite(u_d).all()
            and np.isfinite(u_p).all()
            and np.isfinite(primal_residual)
            and np.isfinite(dual_residual)
        ):
            break
        if primal_residual <= eps_primal and dual_residual <= eps_dual:
            converged = True
            break

    state = {"p": p, "z_d": z_d, "z_p": z_p, "u_d": u_d, "u_p": u_p}
    diagnostics = {
        "converged": converged,
        "iterations": iteration,
        "primal_residual": float(primal_residual),
        "dual_residual": float(dual_residual),
        "objective": float(objective_history[-1]) if objective_history else np.inf,
        "objective_history": np.asarray(objective_history, dtype=np.float64),
    }
    return p, state, diagnostics


def _rtacfr_admm_path(xmat, dt, lambda_grid):
    """Solve the full lambda path using one factorization and warm starts."""
    n_obs = len(dt)
    difference = _rtacfr_difference_matrix(n_obs)
    x_scale = np.sum(xmat ** 2) / max(1, n_obs)
    # A moderate fixed rho balances the least-squares curvature and the split
    # constraints while keeping one reusable factorization for the full path.
    rho = max(1.0, 0.05 * float(x_scale))
    normal_matrix = 2.0 * (xmat.T @ xmat)
    normal_matrix += rho * (difference.T @ difference + np.eye(n_obs))
    factor = linalg.cho_factor(normal_matrix, lower=True, check_finite=False)
    rhs_data = 2.0 * (xmat.T @ dt)

    state = None
    path = []
    for lambda_val in lambda_grid:
        p_hat, state, diagnostics = _rtacfr_admm_solve(
            xmat,
            dt,
            difference,
            factor,
            rho,
            float(lambda_val),
            state=state,
            rhs_data=rhs_data,
        )
        path.append((np.clip(p_hat, 0.0, 1.0), diagnostics))
    return path


def _rtacfr_cvxpy_path(xmat, dt, lambda_grid):
    """Solve the optional historical cvxpy backend without importing it by default."""
    import cvxpy as cp

    n_obs = len(dt)
    p = cp.Variable(n_obs)
    lambda_param = cp.Parameter(nonneg=True)
    loss = cp.sum_squares(dt - xmat @ p)
    penalty = cp.norm1(p[1:] - p[:-1])
    objective = cp.Minimize(loss + lambda_param * penalty)
    constraints = [p >= 0, p <= 1]
    problem = cp.Problem(objective, constraints)
    path = []
    for lambda_val in lambda_grid:
        lambda_param.value = lambda_val
        try:
            problem.solve(solver=cp.ECOS, verbose=False, warm_start=True)
            if p.value is not None:
                path.append((np.asarray(p.value, dtype=np.float64).copy(), None))
            else:
                path.append((None, None))
        except cp.error.SolverError:
            path.append((None, None))
    return path


def _run_rtacfr_fusedlasso_internal(data, delay_dist=None, solver="admm"):
    """Estimate the boxed fused-lasso rtaCFR signal.

    ``solver="admm"`` is the deterministic default and imports only NumPy and
    SciPy.  ``solver="cvxpy"`` retains the historical implementation as an
    explicit opt-in backend.
    """
    if solver not in {"admm", "cvxpy"}:
        raise ValueError("solver must be 'admm' or 'cvxpy'")
    _, dt, _, xmat = _rtacfr_design_matrix(data, delay_dist=delay_dist)
    n_obs = len(dt)
    lambda_grid = np.logspace(-2, 4, 30)
    path = (
        _rtacfr_admm_path(xmat, dt, lambda_grid)
        if solver == "admm"
        else _rtacfr_cvxpy_path(xmat, dt, lambda_grid)
    )

    min_bic = np.inf
    best_p_hat = None
    fallback_p_hat = None
    fallback_rss = np.inf
    for p_hat, _ in path:
        if p_hat is None or not np.isfinite(p_hat).all():
            continue
        p_hat = np.clip(np.asarray(p_hat, dtype=np.float64), 0.0, 1.0)
        rss = np.sum((dt - xmat @ p_hat) ** 2)
        if not np.isfinite(rss):
            continue
        if rss < fallback_rss:
            fallback_rss = rss
            fallback_p_hat = p_hat.copy()
        if rss < 1e-9:
            continue
        num_params = len(np.unique(np.round(p_hat, 4)))
        bic = n_obs * np.log(rss / n_obs) + num_params * np.log(n_obs)
        if bic < min_bic:
            min_bic = bic
            best_p_hat = p_hat.copy()
    if best_p_hat is None:
        best_p_hat = fallback_p_hat
    if best_p_hat is None:
        return np.zeros(n_obs, dtype=np.float64)
    return best_p_hat


def _rtacfr_safe_cache_component(value):
    """Convert a user-facing cache tag into a stable filename component."""
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return component.strip("-._") or "default"


def _rtacfr_cache_tag(delay_dist, delay_pmf, cache_tag=None, delay_name=None):
    """Create a readable and delay-specific cache tag."""
    labels = []
    if cache_tag is not None:
        labels.append(str(cache_tag))
    if delay_name is not None:
        labels.append(str(delay_name))
    if labels:
        label = "-".join(labels)
    elif delay_dist is None:
        label = "default"
    else:
        distribution = getattr(delay_dist, "dist", None)
        label = getattr(distribution, "name", type(delay_dist).__name__)
    digest = hashlib.sha1(
        np.ascontiguousarray(delay_pmf, dtype=np.float64).tobytes()
    ).hexdigest()[:10]
    return f"{_rtacfr_safe_cache_component(label)}-{digest}"


def get_rtacfr_signal(
    data,
    scenario_name,
    rep_idx,
    delay_dist=None,
    cache_tag=None,
    delay_name=None,
):
    """Calculate and cache the fused-lasso signal for an assumed delay."""
    n_obs = len(data["cases"])
    active_delay_dist = DELAY_DIST if delay_dist is None else delay_dist
    delay_pmf = np.diff(
        np.asarray(active_delay_dist.cdf(np.arange(n_obs + 1)), dtype=np.float64)
    )
    delay_tag = _rtacfr_cache_tag(
        delay_dist, delay_pmf, cache_tag=cache_tag, delay_name=delay_name
    )
    scenario_tag = _rtacfr_safe_cache_component(scenario_name)
    cache_file = os.path.join(
        SIGNAL_CACHE_DIR,
        f"signal_scen={scenario_tag}_rep={rep_idx}_delay={delay_tag}.npz",
    )
    os.makedirs(SIGNAL_CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_file):
        return np.load(cache_file)["signal"]

    # Preserve reads from caches generated by the pre-delay-tag API.  New
    # writes always use the delay-specific filename above.
    if delay_dist is None and cache_tag is None and delay_name is None:
        legacy_file = os.path.join(
            SIGNAL_CACHE_DIR, f"signal_scen={scenario_tag}_rep={rep_idx}.npz"
        )
        if os.path.exists(legacy_file):
            return np.load(legacy_file)["signal"]

    p_hat = _run_rtacfr_fusedlasso_internal(
        data, delay_dist=delay_dist, solver="admm"
    )
    np.savez(cache_file, signal=p_hat)
    return p_hat


def run_rtacfr(
    data,
    scenario_name,
    rep_idx,
    delay_dist=None,
    cache_tag=None,
    delay_name=None,
):
    """
    Wrapper for the pure rtaCFR-fusedlasso method as a benchmark.
    This provides the raw smoothed signal for plotting.
    """
    signal = get_rtacfr_signal(
        data, scenario_name, rep_idx, delay_dist=delay_dist,
        cache_tag=cache_tag, delay_name=delay_name,
    )
    diffs = np.diff(signal)
    taus_est = np.where(np.abs(diffs) > 1e-6)[0] + 1
    k_est = len(taus_est)
    return {"k_est": k_est, "taus_est": list(taus_est), "p_t_hat": signal}

def run_pelt(
    data,
    scenario_name,
    rep_idx,
    delay_dist=None,
    cache_tag=None,
    delay_name=None,
):
    """Wrapper for PELT method applied to the cached rtaCFR signal."""
    T = data["cases"].shape[0]
    import ruptures as rpt

    signal = get_rtacfr_signal(
        data, scenario_name, rep_idx, delay_dist=delay_dist,
        cache_tag=cache_tag, delay_name=delay_name,
    )
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=np.log(T) * 2)
    p_t_hat = np.zeros(T)
    boundaries = [0] + result
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if end > start:
            p_t_hat[start:end] = np.mean(signal[start:end])
    return {"k_est": len(result)-1, "taus_est": result[:-1], "p_t_hat": p_t_hat}


def run_binseg(
    data,
    scenario_name,
    rep_idx,
    delay_dist=None,
    cache_tag=None,
    delay_name=None,
):
    """Wrapper for Binary Segmentation method applied to the cached rtaCFR signal."""
    T = data["cases"].shape[0]
    import ruptures as rpt

    signal = get_rtacfr_signal(
        data, scenario_name, rep_idx, delay_dist=delay_dist,
        cache_tag=cache_tag, delay_name=delay_name,
    )
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
