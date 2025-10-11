# evaluation_realdata.py

"""
This script provides functions for the quantitative evaluation of changepoint
detection methods on real-world time series data.
"""

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import gamma
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff


from methods import _calculate_log_likelihood, _get_theta_t_from_state, _sigmoid
from config import DELAY_DIST, REAL_DATA_PRIOR_K_GEOMETRIC_P, REAL_DATA_PRIOR_THETA_SIGMA

# ==============================================================================
# 1. Parsimony Evaluation
# ==============================================================================

def calculate_bic(deaths, cases, p_t_hat, k_est, delay_dist):
    """
    Computes the Bayesian Information Criterion (BIC) for a given CFR estimate.
    Lower BIC indicates a better balance of model fit and parsimony.
    Uses the complete Poisson log-likelihood including the factorial term.
    """
    from scipy.special import gammaln
    
    T = len(deaths)
    delay_pmf = np.diff(delay_dist.cdf(np.arange(T + 1)))
    
    # Calculate expected deaths
    signal = cases * p_t_hat
    expected_deaths = np.zeros(T)
    for i in range(T):
        for j in range(i + 1):
            if j < len(delay_pmf):
                expected_deaths[i] += signal[i - j] * delay_pmf[j]
    
    # Calculate complete Poisson log-likelihood including factorial term
    log_likelihood = 0.0
    for t in range(T):
        mu = max(1e-9, expected_deaths[t])
        d_t = deaths[t]
        
        # Standard Poisson log-likelihood: d_t * log(mu) - mu - log(d_t!)
        if d_t == 0:
            log_factorial = 0.0  # log(0!) = log(1) = 0
        else:
            log_factorial = gammaln(d_t + 1)  # log(d_t!)
        
        log_likelihood += d_t * np.log(mu) - mu - log_factorial
    
    # Number of parameters = k changepoints + (k+1) segment levels
    num_params = k_est + (k_est + 1)
    
    bic = -2 * log_likelihood + num_params * np.log(T)
    return bic

# ==============================================================================
# 2. Prediction Error
# ==============================================================================

def calculate_out_of_sample_rmse(data, method_func, delay_dist, n_splits=5):
    """
    Computes the out-of-sample Root Mean Squared Error (RMSE) for death counts
    using time series cross-validation.
    """
    cases, deaths = data['cases'], data['deaths']
    T = len(cases)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    errors = []
    
    print(f"Running Time Series Cross-Validation for {method_func.__name__}...")
    for split_idx, (train_idx, test_idx) in enumerate(tqdm(tscv.split(cases), total=n_splits)):
        data_train = {'cases': cases[train_idx], 'deaths': deaths[train_idx]}
        
        # Refit the model on the training data, handling different function signatures
        if method_func.__name__ == 'run_rjmcmc':
            fit_results = method_func(data_train, 
                                      p_geom=REAL_DATA_PRIOR_K_GEOMETRIC_P, 
                                      theta_sigma=REAL_DATA_PRIOR_THETA_SIGMA)
        else:
            cv_scenario_name = f'CV_fit_split_{split_idx}'
            fit_results = method_func(data_train, cv_scenario_name, 0)
            
        p_t_train_hat = fit_results['p_t_hat']
        
        # Forecast CFR on test set by holding the last value constant
        p_t_forecast = np.zeros(T)
        p_t_forecast[train_idx] = p_t_train_hat
        p_t_forecast[test_idx] = p_t_train_hat[-1]
        
        # Calculate expected deaths for the test set
        signal = cases * p_t_forecast
        delay_pmf = np.diff(delay_dist.cdf(np.arange(T + 1)))
        expected_deaths_full = np.convolve(signal, delay_pmf)[:T]
        expected_deaths_test = expected_deaths_full[test_idx]
        
        squared_error = (deaths[test_idx] - expected_deaths_test)**2
        errors.extend(squared_error)
        
    return np.sqrt(np.mean(errors))

# ==============================================================================
# 3. Alignment
# ==============================================================================

def load_event_list(filepath, series_start_date):
    """Loads a CSV of events and converts dates to time indices."""
    events_df = pd.read_csv(filepath)
    events_df['date'] = pd.to_datetime(events_df['date'])
    series_start_date = pd.to_datetime(series_start_date)
    
    # Convert event dates to integer indices
    events_df['time_idx'] = (events_df['date'] - series_start_date).dt.days
    return events_df

def calculate_hausdorff_alignment(detected_cps, event_indices):
    """
    Calculates the Hausdorff distance between detected changepoints and a list of event dates.
    A lower score indicates better alignment.
    """
    if not detected_cps or not event_indices:
        return np.nan

    u = np.array(detected_cps).reshape(-1, 1)
    v = np.array(event_indices).reshape(-1, 1)
    
    d_uv = directed_hausdorff(u, v)[0]
    d_vu = directed_hausdorff(v, u)[0]
    return max(d_uv, d_vu)

def calculate_event_hit_probability(pip_array, event_indices, window):
    """
    For each event, computes the posterior probability that there is at least one CP in the window.
    Then averages this probability across all events.
    """
    if pip_array is None:
        return np.nan
    
    T = len(pip_array)
    hit_probs = []
    
    for event_idx in event_indices:
        start = max(0, event_idx - window)
        end = min(T, event_idx + window + 1)
        
        window_pip = pip_array[start:end]
        
        # P(at least one CP) = 1 - P(no CPs)
        # P(no CPs) = product(1 - P(CP at t)) for t in window
        prob_no_cp_in_window = np.prod(1 - window_pip)
        prob_at_least_one_cp = 1 - prob_no_cp_in_window
        hit_probs.append(prob_at_least_one_cp)
        
    return np.mean(hit_probs)

def calculate_alignment_metrics(detected_cps, event_indices, window=14):
    """
    Calculates Precision, Recall, and F1 score for alignment with a list of events.
    The 'recall' is the deterministic equivalent of the event-window hit probability.
    """
    tp = 0
    fp = 0
    fn = 0
    
    detected_cps_matched = [False] * len(detected_cps)

    # Calculate True Positives and False Negatives
    for event_idx in event_indices:
        is_matched = False
        for i, cp in enumerate(detected_cps):
            if not detected_cps_matched[i] and abs(cp - event_idx) <= window:
                is_matched = True
                detected_cps_matched[i] = True
                break
    
        if is_matched:
            tp += 1
        else:
            fn += 1
            
    # Calculate False Positives
    fp = len(detected_cps) - sum(detected_cps_matched)
    
    # Calculate Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'tp': tp, 'fp': fp, 'fn': fn}
