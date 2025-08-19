# analysis.py

"""
This script handles the analysis of the simulation results. It loads the
checkpoint files, computes metrics, finds optimal hyperparameters, and
generates the final figures and tables.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from scipy.spatial.distance import directed_hausdorff
from collections import Counter

from config import (SENSITIVITY_RESULTS_DIR, MAIN_RESULTS_DIR, PLOTS_DIR,
                    SCENARIOS, T, SENSITIVITY_GRID_PRIORS, DELAY_DIST_SENSITIVITY,
                    SIGNAL_CACHE_DIR)

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def calculate_mae(p_t_true, p_t_hat):
    """Calculates the Mean Absolute Error, ignoring NaNs."""
    if p_t_hat is None: return np.nan
    return np.nanmean(np.abs(p_t_true - p_t_hat))

def calculate_accuracy(k_true, k_hat):
    """Calculates the accuracy of the number of changepoints."""
    if k_hat < 0: return np.nan
    return 1.0 if k_true == k_hat else 0.0

def calculate_hausdorff(true_cps, est_cps):
    """Calculates the Hausdorff distance between true and estimated changepoints."""
    if not true_cps: return 0.0 if not est_cps else np.nan
    if not est_cps: return np.nan
    u = np.array(true_cps).reshape(-1, 1)
    v = np.array(est_cps).reshape(-1, 1)
    d_uv = directed_hausdorff(u, v)[0]
    d_vu = directed_hausdorff(v, u)[0]
    return max(d_uv, d_vu)

# ==============================================================================
# DATA LOADING AND PROCESSING
# ==============================================================================

def load_and_process_sensitivity_results(results_dir):
    """Loads sensitivity results and computes metrics ONLY for RJMCMC."""
    all_results = []
    fnames = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
    for fname in tqdm(fnames, desc=f"Loading sensitivity results from: {results_dir}"):
        filepath = os.path.join(results_dir, fname)
        with open(filepath, 'rb') as f:
            res = pickle.load(f)
        params = res.get('params', {})
        rjmcmc_res = res.get('rjmcmc', {})
        p_t_hat = rjmcmc_res.get('p_t_hat', np.full(T, np.nan))
        k_est = rjmcmc_res.get('k_est', -1)
        all_results.append({
            'scenario': res['scenario'],
            'delay_setting': params.get('DELAY_DIST_NAME'),
            'p_geom': params.get('PRIOR_K_GEOMETRIC_P'),
            'theta_sigma': params.get('PRIOR_THETA_SIGMA'),
            'mae': calculate_mae(res['data']['true_p_t'], p_t_hat),
            'accuracy': calculate_accuracy(len(res['data']['true_cps']), k_est),
            'hausdorff': calculate_hausdorff(res['data']['true_cps'], rjmcmc_res.get('taus_est', []))
        })
    return pd.DataFrame(all_results)

def load_and_process_main_results(results_dir):
    """Loads main simulation results and computes metrics for ALL methods."""
    all_metrics = []
    fnames = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
    for fname in tqdm(fnames, desc=f"Loading main results from: {results_dir}"):
        filepath = os.path.join(results_dir, fname)
        with open(filepath, 'rb') as f:
            res = pickle.load(f)
        true_cps, true_k, true_p_t = res['data']['true_cps'], len(res['data']['true_cps']), res['data']['true_p_t']
        methods = {'RJMCMC': res.get('rjmcmc', {}), 'rtaCFR': res.get('rtacfr', {}), 'Pelt': res.get('pelt', {}), 'BinSeg': res.get('binseg', {})}
        for name, method_res in methods.items():
            all_metrics.append({
                'scenario': res['scenario'], 'method': name,
                'accuracy': calculate_accuracy(true_k, method_res.get('k_est', -1)),
                'hausdorff': calculate_hausdorff(true_cps, method_res.get('taus_est', [])),
                'mae': calculate_mae(true_p_t, method_res.get('p_t_hat'))
            })
    return pd.DataFrame(all_metrics)

# ==============================================================================
# HYPERPARAMETER OPTIMIZATION & TABLE GENERATION
# ==============================================================================

def find_optimal_hyperparameters(df_sens):
    """
    Finds the optimal hyperparameter combination from sensitivity results
    by ranking across Accuracy, Hausdorff Distance, and MAE.
    """
    df_perfect = df_sens[df_sens['delay_setting'] == "Perfectly Matched Delay (Mean=15.43)"].copy()
    df_perfect.dropna(subset=['mae', 'accuracy'], inplace=True)
    
    grouped = df_perfect.groupby(['p_geom', 'theta_sigma']).agg(
        mean_mae=('mae', 'mean'),
        mean_accuracy=('accuracy', 'mean'),
        mean_hd=('hausdorff', lambda x: x.mean(skipna=True))
    ).reset_index()

    grouped['mae_rank'] = grouped['mean_mae'].rank(method='min')
    grouped['accuracy_rank'] = grouped['mean_accuracy'].rank(method='min', ascending=False)
    grouped['hd_rank'] = grouped['mean_hd'].rank(method='min')
    
    grouped['overall_rank'] = grouped[['mae_rank', 'accuracy_rank', 'hd_rank']].mean(axis=1)

    optimal_row = grouped.loc[grouped['overall_rank'].idxmin()]
    optimal_params = {'PRIOR_K_GEOMETRIC_P': optimal_row['p_geom'], 'PRIOR_THETA_SIGMA': optimal_row['theta_sigma']}
    print(f"\n--- Optimal Hyperparameter Selection ---\nSelected from 'Perfectly Matched Delay' scenario.\nOptimal p_geom: {optimal_params['PRIOR_K_GEOMETRIC_P']}\nOptimal theta_sigma: {optimal_params['PRIOR_THETA_SIGMA']}\n--------------------------------------\n")
    return optimal_params

def generate_main_results_table(df_main):
    """Generates and saves a LaTeX table from the main simulation results."""
    summary = df_main.groupby(['scenario', 'method']).agg(
        accuracy_mean=('accuracy', 'mean'),
        hausdorff_mean=('hausdorff', lambda x: x.mean(skipna=True)),
        hausdorff_std=('hausdorff', lambda x: x.std(skipna=True)),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std')
    ).reset_index()
    save_path = os.path.join(PLOTS_DIR, "main_results_table.tex")
    # ... (file writing logic is unchanged)
    print(f"Main results LaTeX table saved at: {save_path}")

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def generate_publication_figure(results_dir, optimal_params):
    """Generates the 5x3 publication figure comparing RJMCMC and the original rtaCFR."""
    fig, axes = plt.subplots(len(SCENARIOS), 3, figsize=(18, 22), gridspec_kw={'width_ratios': [1, 1, 2]})
    scenario_order = list(SCENARIOS.keys())

    for i, scenario_name in enumerate(scenario_order):
        all_k_samples, all_tau_samples = [], []
        rjmcmc_p_t_runs, rtacfr_p_t_runs = [], []
        true_p_t, true_cps = None, None

        fnames = [f for f in os.listdir(results_dir) if scenario_name.replace(' ', '_') in f]
        for fname in fnames:
            p_opt = optimal_params['PRIOR_K_GEOMETRIC_P']
            s_opt = optimal_params['PRIOR_THETA_SIGMA']
            if f"p{p_opt}" in fname and f"s{s_opt}" in fname:
                with open(os.path.join(results_dir, fname), 'rb') as f:
                    res = pickle.load(f)
                
                rjmcmc_res = res.get('rjmcmc', {})
                if 'k_samples' in rjmcmc_res:
                    all_k_samples.extend(rjmcmc_res['k_samples'])
                    for k, taus in zip(rjmcmc_res['k_samples'], rjmcmc_res['taus_samples']):
                        if k > 0:
                            all_tau_samples.extend(taus[:k])
                if 'p_t_samples' in rjmcmc_res:
                    rjmcmc_p_t_runs.append(rjmcmc_res['p_t_samples'])
                
                # Load rtaCFR signal directly from cache
                match = re.search(r'_rep(\d+)_', fname)
                if match:
                    rep_idx = int(match.group(1))
                    scen_fname = scenario_name.replace(" ", "-")
                    cache_file = os.path.join(SIGNAL_CACHE_DIR, f"signal_scen={scen_fname}_rep={rep_idx}.npz")
                    if os.path.exists(cache_file):
                        with np.load(cache_file) as loaded_data:
                            rtacfr_p_t_runs.append(loaded_data['signal'])
                    else:
                        print(f"Warning: Cache file not found: {cache_file}")

                if true_p_t is None:
                    true_p_t, true_cps = res['data']['true_p_t'], res['data']['true_cps']
        
        ax = axes[i, 0]
        if all_k_samples:
            sns.histplot(all_k_samples, ax=ax, discrete=True, stat='probability', shrink=0.8)
        ax.set_title(f"Posterior k\n{scenario_name}"); ax.set_xlabel("Number of Changepoints (k)")
        ax.axvline(len(true_cps), color='red', linestyle='--', label=f'True k={len(true_cps)}'); ax.legend()

        ax = axes[i, 1]
        if true_cps:
            if all_tau_samples:
                sns.histplot(all_tau_samples, ax=ax, bins=T, kde=False)
            for cp_idx, cp in enumerate(true_cps):
                ax.axvline(cp, color='red', linestyle='--', label='True CP' if cp_idx == 0 else "")
            ax.legend()
        else:
            ax.set_xlim(0, T) # Keep plot blank for K=0 scenario
        ax.set_title("Posterior Changepoint Locations"); ax.set_xlabel("Time (t)"); ax.set_ylabel("Count")

        ax = axes[i, 2]
        if rjmcmc_p_t_runs:
            all_p_t_samples = np.vstack(rjmcmc_p_t_runs)
            p_t_mean = np.mean(all_p_t_samples, axis=0)
            p_t_lower = np.percentile(all_p_t_samples, 2.5, axis=0)
            p_t_upper = np.percentile(all_p_t_samples, 97.5, axis=0)
            ax.plot(p_t_mean, color='dodgerblue', label='RJMCMC Mean Estimate')
            ax.fill_between(range(T), p_t_lower, p_t_upper, color='skyblue', alpha=0.4, label='RJMCMC 95% CrI')

        if rtacfr_p_t_runs:
            avg_rtacfr_p_t = np.mean(np.vstack(rtacfr_p_t_runs), axis=0)
            ax.plot(avg_rtacfr_p_t, color='green', linestyle='--', lw=2, label='rtaCFR Mean Estimate')
        
        ax.plot(true_p_t, color='black', lw=2, label='True CFR')
        ax.set_title("Averaged CFR Estimate"); ax.set_xlabel("Time (t)"); ax.set_ylabel("Fatality Rate"); ax.legend(); ax.set_ylim(bottom=0)

    plt.tight_layout(h_pad=3)
    save_path = os.path.join(PLOTS_DIR, "publication_figure.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Publication figure saved at: {save_path}")

def generate_sensitivity_heatmap_grid(df_sens):
    """Generates a 3x3 grid of heatmaps for the sensitivity analysis."""
    delay_order = list(DELAY_DIST_SENSITIVITY.keys())
    metrics = ['accuracy', 'hausdorff', 'mae']
    metric_titles = ['Accuracy P(k_est=k_true)', 'Hausdorff Distance (HD)', 'Mean Absolute Error (MAE)']
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Sensitivity of RJMCMC Performance to Priors and Delay Misspecification', fontsize=20, y=1.02)

    p_geom_labels = SENSITIVITY_GRID_PRIORS['PRIOR_K_GEOMETRIC_P']
    theta_sigma_labels = SENSITIVITY_GRID_PRIORS['PRIOR_THETA_SIGMA']

    for row, delay_name in enumerate(delay_order):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            
            delay_df = df_sens[df_sens['delay_setting'] == delay_name]
            pivot_df = delay_df.groupby(['p_geom', 'theta_sigma'])[metric].mean().reset_index()
            pivot_table = pivot_df.pivot(index='theta_sigma', columns='p_geom', values=metric)
            
            sns.heatmap(pivot_table, ax=ax, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
            
            if row == 0:
                ax.set_title(metric_titles[col], fontsize=14)
            if col == 0:
                ax.set_ylabel(delay_name, fontsize=14)
            else:
                ax.set_ylabel('')

            ax.set_xlabel('p_geom')
            ax.set_xticklabels(p_geom_labels)
            ax.set_yticklabels(theta_sigma_labels, rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(PLOTS_DIR, "sensitivity_analysis_heatmap_grid.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Sensitivity analysis heatmap grid saved at: {save_path}")

def full_analysis_workflow():
    """The full, two-stage analysis workflow."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    df_sens = load_and_process_sensitivity_results(SENSITIVITY_RESULTS_DIR)
    generate_sensitivity_heatmap_grid(df_sens)
    optimal_params = find_optimal_hyperparameters(df_sens)
    
    df_main = load_and_process_main_results(MAIN_RESULTS_DIR)
    generate_main_results_table(df_main)
    generate_publication_figure(MAIN_RESULTS_DIR, optimal_params)

    print("\n--- Analysis Complete ---")
    return optimal_params

if __name__ == "__main__":
    full_analysis_workflow()
