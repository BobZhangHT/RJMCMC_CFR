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
from scipy.special import logit

from config import (SENSITIVITY_RESULTS_DIR, MAIN_RESULTS_DIR, PLOTS_DIR,
                    SCENARIOS, T, SENSITIVITY_GRID_PRIORS, DELAY_DIST_SENSITIVITY,
                    SIGNAL_CACHE_DIR, K_MAX)

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def _logit_safe(p, eps=1e-9):
    """Safely applies logit transformation by clamping values near 0 and 1."""
    p_safe = np.clip(p, eps, 1 - eps)
    return logit(p_safe)

def calculate_mae(p_t_true, p_t_hat):
    """
    Calculates the Mean Absolute Error on the LOGIT-TRANSFORMED CFR.
    This amplifies differences near 0 and 1.
    """
    if p_t_hat is None or p_t_true is None:
        return np.nan
    
    # Apply safe logit transformation to both true and estimated values
    true_logit = _logit_safe(p_t_true)
    hat_logit = _logit_safe(p_t_hat)
    
    return np.nanmean(np.abs(true_logit - hat_logit))

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

    pivot = summary.pivot(index='scenario', columns='method')
    scenario_order = list(SCENARIOS.keys())
    method_order = ['RJMCMC', 'Pelt', 'BinSeg']
    
    pivot = pivot.reindex(scenario_order)
    pivot = pivot.reindex(columns=[(level1, level2) for level1 in pivot.columns.levels[0] for level2 in method_order])

    latex_string = "\\begin{table}[ht]\n"
    latex_string += "\\centering\n"
    latex_string += "\\caption{Performance summary under optimal priors. Mean (std) reported for HD and Logit MAE.}\n"
    latex_string += "\\label{tab:main_results}\n"
    latex_string += "\\resizebox{\\textwidth}{!}{\n"
    latex_string += "\\begin{tabular}{l" + "ccc" * len(method_order) + "}\n"
    latex_string += "\\toprule\n"
    
    latex_string += "Scenario & "
    for method in method_order:
        method_label = "Proposed (RJMCMC)" if method == "RJMCMC" else method
        latex_string += f"\\multicolumn{{3}}{{c}}{{{method_label}}} & "
    latex_string = latex_string.rstrip(' &') + " \\\\\n"
    latex_string += "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n"
    
    latex_string += " & " + "Accuracy & Hausdorff (HD) & Logit MAE & " * len(method_order)
    latex_string = latex_string.rstrip(' &') + " \\\\\n"
    latex_string += "\\midrule\n"
    
    for scenario in scenario_order:
        latex_string += f"{scenario} & "
        for method in method_order:
            acc = pivot.loc[scenario, ('accuracy_mean', method)]
            hd_mean = pivot.loc[scenario, ('hausdorff_mean', method)]
            hd_std = pivot.loc[scenario, ('hausdorff_std', method)]
            mae_mean = pivot.loc[scenario, ('mae_mean', method)]
            mae_std = pivot.loc[scenario, ('mae_std', method)]
            
            hd_str = "NA" if np.isnan(hd_mean) else f"{hd_mean:.2f} ({hd_std:.2f})"
            mae_str = f"{mae_mean:.2f} ({mae_std:.2f})"
            
            latex_string += f"{acc:.2f} & {hd_str} & {mae_str} & "
        latex_string = latex_string.rstrip(' &') + " \\\\\n"
        
    latex_string += "\\bottomrule\n"
    latex_string += "\\end{tabular}}\n"
    latex_string += "\\end{table}"

    save_path = os.path.join(PLOTS_DIR, "main_results_table.tex")
    with open(save_path, 'w') as f:
        f.write(latex_string)
    
    print(f"Main results LaTeX table saved at: {save_path}")

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def generate_publication_figure(results_dir, optimal_params):
    """Generates the 5x3 publication figure by aggregating summarized results."""
    fig, axes = plt.subplots(len(SCENARIOS), 3, figsize=(18, 22), gridspec_kw={'width_ratios': [1, 1, 2]})
    scenario_order = list(SCENARIOS.keys())

    for i, scenario_name in enumerate(scenario_order):
        all_k_est, all_taus_est = [], []
        all_p_t_means, all_p_t_lowers, all_p_t_uppers = [], [], []
        rtacfr_p_t_runs = []
        true_p_t, true_cps = None, None

        fnames = [f for f in os.listdir(results_dir) if scenario_name.replace(' ', '_') in f]
        for fname in fnames:
            p_opt = optimal_params['PRIOR_K_GEOMETRIC_P']
            s_opt = optimal_params['PRIOR_THETA_SIGMA']
            if f"p{p_opt}" in fname and f"s{s_opt}" in fname:
                with open(os.path.join(results_dir, fname), 'rb') as f:
                    res = pickle.load(f)
                
                rjmcmc_res = res.get('rjmcmc', {})
                all_k_est.append(rjmcmc_res.get('k_est', -1))
                all_taus_est.extend(rjmcmc_res.get('taus_est', []))
                all_p_t_means.append(rjmcmc_res.get('p_t_hat', np.full(T, np.nan)))
                all_p_t_lowers.append(rjmcmc_res.get('p_t_lower_ci', np.full(T, np.nan)))
                all_p_t_uppers.append(rjmcmc_res.get('p_t_upper_ci', np.full(T, np.nan)))
                
                match = re.search(r'_rep(\d+)', fname)
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
        sns.histplot(all_k_est, ax=ax, discrete=True, stat='probability', shrink=0.8)
        ax.set_title(f"Histogram of $\\hat{{k}}$\n{scenario_name}")
        ax.set_xlim(0, K_MAX)
        ax.set_xlabel("Estimated Number of Changepoints ($\\hat{k}$)")
        ax.axvline(len(true_cps), color='red', linestyle='--', label=f'True k={len(true_cps)}'); ax.legend()

        ax = axes[i, 1]
        if true_cps:
            if all_taus_est:
                sns.histplot(all_taus_est, ax=ax, bins=T, kde=False)
            for cp_idx, cp in enumerate(true_cps):
                ax.axvline(cp, color='red', linestyle='--', label='True CP' if cp_idx == 0 else "")
            ax.set_xlim(0, T)
            ax.legend()
        else:
            ax.set_xlim(0, T) # Keep plot blank for K=0 scenario
        ax.set_title("Histogram of Estimated CPs"); ax.set_xlabel("Time (t)"); ax.set_ylabel("Count")

        ax = axes[i, 2]
        if all_p_t_means:
            avg_p_t_mean = np.mean(np.vstack(all_p_t_means), axis=0)
            avg_p_t_lower = np.mean(np.vstack(all_p_t_lowers), axis=0)
            avg_p_t_upper = np.mean(np.vstack(all_p_t_uppers), axis=0)
            ax.plot(avg_p_t_mean, color='dodgerblue', label='RJMCMC Avg. Mean Estimate')
            ax.fill_between(range(T), avg_p_t_lower, avg_p_t_upper, color='skyblue', alpha=0.4, label='RJMCMC Avg. 95% CrI')

        if rtacfr_p_t_runs:
            avg_rtacfr_p_t = np.mean(np.vstack(rtacfr_p_t_runs), axis=0)
            ax.plot(avg_rtacfr_p_t, color='green', linestyle='--', lw=2, label='rtaCFR Avg. Estimate')
        
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
