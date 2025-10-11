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

# --- Global plotting style for publication-quality figures ---
sns.set_theme(context='talk', style='whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 2.5,  # Increase border width
    'axes.edgecolor': 'black',  # Set border color to black
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def _logit_safe(p, eps=1e-9):
    """
    Safely applies the logit transformation by clamping input values to be
    within a small epsilon of the [0, 1] interval to avoid infinity.

    Args:
        p (np.ndarray): Array of probabilities.
        eps (float): A small epsilon value to prevent logit(0) or logit(1).

    Returns:
        np.ndarray: Logit-transformed values.
    """
    p_safe = np.clip(p, eps, 1 - eps)
    return logit(p_safe)

def calculate_mae(p_t_true, p_t_hat):
    """
    Calculates the Mean Absolute Error on the LOGIT-TRANSFORMED CFR.
    This transformation amplifies differences for CFR values near 0 and 1.

    Args:
        p_t_true (np.ndarray): The true CFR time series.
        p_t_hat (np.ndarray): The estimated CFR time series.

    Returns:
        float: The mean absolute error on the logit scale. Returns NaN if inputs are invalid.
    """
    if p_t_hat is None or p_t_true is None:
        return np.nan
    
    # Apply safe logit transformation to both true and estimated values
    true_logit = _logit_safe(p_t_true)
    hat_logit = _logit_safe(p_t_hat)
    
    return np.nanmean(np.abs(true_logit - hat_logit))

def calculate_accuracy(k_true, k_hat):
    """
    Calculates the accuracy of the estimated number of changepoints (k).
    Accuracy is 1 if k_est == k_true, and 0 otherwise.

    Args:
        k_true (int): The true number of changepoints.
        k_hat (int): The estimated number of changepoints.

    Returns:
        float: 1.0 for a correct estimate, 0.0 otherwise. NaN for invalid input.
    """
    if k_hat < 0: return np.nan
    return 1.0 if k_true == k_hat else 0.0

def calculate_hausdorff(true_cps, est_cps):
    """
    Calculates the Hausdorff distance between the true and estimated changepoint locations.
    This metric measures the worst-case discrepancy between the two sets of points.

    Args:
        true_cps (list): A list of true changepoint locations.
        est_cps (list): A list of estimated changepoint locations.

    Returns:
        float: The Hausdorff distance. Returns NaN or 0.0 for edge cases (e.g., empty sets).
    """
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
    """
    Loads and processes all result files from the sensitivity analysis directory.
    It computes evaluation metrics for each run and aggregates them into a DataFrame.

    Args:
        results_dir (str): Path to the directory containing sensitivity result .pkl files.

    Returns:
        pd.DataFrame: A DataFrame with metrics and parameters for each sensitivity run.
    """
    all_results = []
    fnames = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
    for fname in tqdm(fnames, desc=f"Loading sensitivity results from: {results_dir}"):
        filepath = os.path.join(results_dir, fname)
        with open(filepath, 'rb') as f:
            res = pickle.load(f)
        params = res.get('params', {})
        rjmcmc_res = res.get('rjmcmc', {})
        
        # Safely extract results, providing defaults for failed runs
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
    """
    Loads and processes all result files from the main simulation directory.
    It computes metrics for all methods (RJMCMC and benchmarks).

    Args:
        results_dir (str): Path to the directory containing main result .pkl files.

    Returns:
        pd.DataFrame: A DataFrame with metrics for each method and scenario.
    """
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

    Args:
        df_sens (pd.DataFrame): The DataFrame from load_and_process_sensitivity_results.

    Returns:
        dict: A dictionary with the optimal 'PRIOR_K_GEOMETRIC_P' and 'PRIOR_THETA_SIGMA'.
    """
    # Filter for the perfectly matched delay scenario to select the best parameters
    df_perfect = df_sens[df_sens['delay_setting'] == "Perfectly Matched Delay (Mean=15.43)"].copy()
    df_perfect.dropna(subset=['mae', 'accuracy'], inplace=True)
    
    # Aggregate metrics across all scenarios for each hyperparameter combination
    grouped = df_perfect.groupby(['p_geom', 'theta_sigma']).agg(
        mean_mae=('mae', 'mean'),
        mean_accuracy=('accuracy', 'mean'),
        mean_hd=('hausdorff', lambda x: x.mean(skipna=True)) # HD is NaN for "No Change"
    ).reset_index()

    # Rank each metric (lower is better for MAE/HD, higher is better for accuracy)
    grouped['mae_rank'] = grouped['mean_mae'].rank(method='min')
    grouped['accuracy_rank'] = grouped['mean_accuracy'].rank(method='min', ascending=False)
    grouped['hd_rank'] = grouped['mean_hd'].rank(method='min')
    
    # Calculate overall rank as the average of the three individual ranks
    grouped['overall_rank'] = grouped[['mae_rank', 'accuracy_rank', 'hd_rank']].mean(axis=1)

    # Find the combination with the best (lowest) overall rank
    optimal_row = grouped.loc[grouped['overall_rank'].idxmin()]
    optimal_params = {'PRIOR_K_GEOMETRIC_P': optimal_row['p_geom'], 'PRIOR_THETA_SIGMA': optimal_row['theta_sigma']}
    print(f"\n--- Optimal Hyperparameter Selection ---\nSelected from 'Perfectly Matched Delay' scenario.\nOptimal p_geom: {optimal_params['PRIOR_K_GEOMETRIC_P']}\nOptimal theta_sigma: {optimal_params['PRIOR_THETA_SIGMA']}\n--------------------------------------\n")
    return optimal_params

def generate_main_results_table(df_main):
    """
    Generates and saves a LaTeX table from the main simulation results.

    Args:
        df_main (pd.DataFrame): The DataFrame from load_and_process_main_results.
    """
    # Calculate mean and standard deviation for each metric, grouped by scenario and method
    summary = df_main.groupby(['scenario', 'method']).agg(
        accuracy_mean=('accuracy', 'mean'),
        hausdorff_mean=('hausdorff', lambda x: x.mean(skipna=True)),
        hausdorff_std=('hausdorff', lambda x: x.std(skipna=True)),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std')
    ).reset_index()

    # Pivot the table for LaTeX formatting
    pivot = summary.pivot(index='scenario', columns='method')
    scenario_order = list(SCENARIOS.keys())
    method_order = ['RJMCMC', 'Pelt', 'BinSeg']
    
    # Reorder rows and columns to match the desired table structure
    pivot = pivot.reindex(scenario_order)
    pivot = pivot.reindex(columns=[(level1, level2) for level1 in pivot.columns.levels[0] for level2 in method_order])

    # Build the LaTeX string dynamically
    latex_string = "\\begin{table}[ht]\n"
    latex_string += "\\centering\n"
    latex_string += "\\caption{Performance summary under optimal priors. Mean (std) reported for HD and Logit MAE.}\n"
    latex_string += "\\label{tab:main_results}\n"
    latex_string += "\\resizebox{\\textwidth}{!}{\n"
    latex_string += "\\begin{tabular}{l" + "ccc" * len(method_order) + "}\n"
    latex_string += "\\toprule\n"
    
    # Header row
    latex_string += "Scenario & "
    for method in method_order:
        method_label = "Proposed (RJMCMC)" if method == "RJMCMC" else method
        latex_string += f"\\multicolumn{{3}}{{c}}{{{method_label}}} & "
    latex_string = latex_string.rstrip(' &') + " \\\\\n"
    latex_string += "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n"
    
    # Sub-header row
    latex_string += " & " + "Accuracy & Hausdorff (HD) & Logit MAE & " * len(method_order)
    latex_string = latex_string.rstrip(' &') + " \\\\\n"
    latex_string += "\\midrule\n"
    
    # Data rows
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

    # Save the LaTeX string to a .tex file
    save_path = os.path.join(PLOTS_DIR, "main_results_table.tex")
    with open(save_path, 'w') as f:
        f.write(latex_string)
    
    print(f"Main results LaTeX table saved at: {save_path}")

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def generate_publication_figure(results_dir, optimal_params):
    """
    Generates the 5x3 publication figure by aggregating summarized results.
    Each row corresponds to a scenario, and columns show posterior k,
    posterior changepoint locations, and the averaged CFR estimate.

    Args:
        results_dir (str): Path to the main simulation results directory.
        optimal_params (dict): Dictionary of optimal hyperparameters.
    """
    # Larger canvas, clearer default styles for publication
    fig, axes = plt.subplots(
        len(SCENARIOS), 3,
        figsize=(24, 28),
        gridspec_kw={'width_ratios': [1, 1, 2]}
    )
    scenario_order = list(SCENARIOS.keys())

    for i, scenario_name in enumerate(scenario_order):
        # --- Aggregate summarized data for the current scenario ---
        all_k_est, all_taus_est = [], []
        all_p_t_means, all_p_t_lowers, all_p_t_uppers = [], [], []
        rtacfr_p_t_runs = []
        true_p_t, true_cps = None, None

        # Find all result files for the current scenario that used optimal parameters
        fnames = [f for f in os.listdir(results_dir) if scenario_name.replace(' ', '_') in f]
        for fname in fnames:
            p_opt = optimal_params['PRIOR_K_GEOMETRIC_P']
            s_opt = optimal_params['PRIOR_THETA_SIGMA']
            if f"p{p_opt}" in fname and f"s{s_opt}" in fname:
                with open(os.path.join(results_dir, fname), 'rb') as f:
                    res = pickle.load(f)
                
                # Collect summarized RJMCMC statistics from each run
                rjmcmc_res = res.get('rjmcmc', {})
                all_k_est.append(rjmcmc_res.get('k_est', -1))
                all_taus_est.extend(rjmcmc_res.get('taus_est', []))
                all_p_t_means.append(rjmcmc_res.get('p_t_hat', np.full(T, np.nan)))
                all_p_t_lowers.append(rjmcmc_res.get('p_t_lower_ci', np.full(T, np.nan)))
                all_p_t_uppers.append(rjmcmc_res.get('p_t_upper_ci', np.full(T, np.nan)))
                
                # Load the corresponding raw rtaCFR signal from its cache
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
        
        # --- Column 1: Histogram of Posterior Mode k ---
        ax = axes[i, 0]
        if all_k_est:  # Only plot if we have data
            sns.histplot(
                all_k_est, ax=ax, discrete=True, stat='probability', shrink=0.75,
                edgecolor='black', linewidth=2.0, color='tab:blue',
                alpha=0.9  # Higher opacity for better visibility
            )
        ax.set_title(f"Histogram of $\\hat{{k}}$\n{scenario_name}", fontsize=18, fontweight='bold')
        ax.set_xlim(0, K_MAX)
        ax.set_xlabel("Estimated Number of Changepoints ($\\hat{k}$)", fontsize=16)
        ax.set_ylabel("Probability", fontsize=16)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        ax.axvline(len(true_cps), color='red', linestyle='--', linewidth=2.5,
                   label=f'True k={len(true_cps)}')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=14)
        # Set prominent black borders for all spines
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        # --- Column 2: Histogram of Estimated Changepoint Locations ---
        ax = axes[i, 1]
        if true_cps:
            if all_taus_est:
                # Use fixed bin width for consistent appearance across all scenarios
                # Bin width of 5 time units provides good visibility for all scenarios
                bin_width = 5
                min_tau = min(all_taus_est)
                max_tau = max(all_taus_est)
                
                # Create bins with consistent width across the x-axis range (0-200)
                # Center bins around the data range for better visualization
                bins = np.arange(0, 205, bin_width)  # 0, 5, 10, ..., 200
                
                sns.histplot(
                    all_taus_est, ax=ax, bins=bins, kde=False,
                    edgecolor='black', linewidth=2.5, color='tab:purple',
                    alpha=0.95,  # Even higher opacity for maximum visibility
                    stat='count'  # Use count instead of density
                )
            for cp_idx, cp in enumerate(true_cps):
                ax.axvline(cp, color='red', linestyle='--', linewidth=2.5,
                           label='True CP' if cp_idx == 0 else "")
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=14)
        else:
            ax.set_xlim(0, 200) # Keep plot blank for K=0 scenario, fixed range
        
        # Always set x-axis range to 0-200 regardless of scenario
        ax.set_xlim(0, 200)
        ax.set_title("Histogram of Estimated CPs", fontsize=18, fontweight='bold')
        ax.set_xlabel("Time (t)", fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        
        # Set prominent black borders for all spines
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        # --- Column 3: Averaged CFR Estimate ---
        ax = axes[i, 2]
        if all_p_t_means:
            avg_p_t_mean = np.mean(np.vstack(all_p_t_means), axis=0)
            avg_p_t_lower = np.mean(np.vstack(all_p_t_lowers), axis=0)
            avg_p_t_upper = np.mean(np.vstack(all_p_t_uppers), axis=0)
            ax.plot(avg_p_t_mean, color='dodgerblue', linewidth=3, label='RJMCMC Avg. Mean Estimate')
            ax.fill_between(range(T), avg_p_t_lower, avg_p_t_upper,
                            color='skyblue', alpha=0.35,
                            label='RJMCMC Avg. 95% CrI')

        if rtacfr_p_t_runs:
            avg_rtacfr_p_t = np.mean(np.vstack(rtacfr_p_t_runs), axis=0)
            ax.plot(avg_rtacfr_p_t, color='green', linestyle='--', lw=3,
                    label='rtaCFR Avg. Estimate')
        
        ax.plot(true_p_t, color='black', lw=3, label='True CFR')
        ax.set_title("Averaged CFR Estimate", fontsize=18, fontweight='bold')
        ax.set_xlabel("Time (t)", fontsize=16)
        ax.set_ylabel("Fatality Rate", fontsize=16)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=14)
        # Set prominent black borders for all spines
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.5)
            spine.set_visible(True)

    plt.tight_layout(h_pad=3)
    save_path = os.path.join(PLOTS_DIR, "publication_figure.pdf")
    # Save higher-resolution versions
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    png_path = os.path.join(PLOTS_DIR, "publication_figure.png")
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Publication figure saved at: {save_path}")

def generate_sensitivity_line_plots(df_sens):
    """
    Generates line plots showing sensitivity analysis results.
    Each subplot shows how metrics vary with hyperparameters for different delay settings.
    
    Args:
        df_sens (pd.DataFrame): The DataFrame from load_and_process_sensitivity_results.
    """
    delay_order = list(DELAY_DIST_SENSITIVITY.keys())
    metrics = ['accuracy', 'hausdorff', 'mae']
    metric_titles = ['Accuracy P(k_est = k_true)', 'Hausdorff Distance (HD)', 'Mean Absolute Error (MAE)']
    
    # Create a 3x5 subplot grid (3 metrics x 5 p_geom values)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(
        'Sensitivity Analysis: Performance vs Hyperparameters',
        fontsize=22,
        fontweight='bold',
        y=0.96
    )
    
    p_geom_values = SENSITIVITY_GRID_PRIORS['PRIOR_K_GEOMETRIC_P']
    theta_sigma_values = SENSITIVITY_GRID_PRIORS['PRIOR_THETA_SIGMA']
    
    # Define colors and markers for different delay settings
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']  # circle, square, triangle
    
    # Store handles and labels for a single legend
    legend_handles = []
    legend_labels = []
    
    for row, metric in enumerate(metrics):
        for col, p_geom in enumerate(p_geom_values):
            ax = axes[row, col]
            
            # Plot lines for each delay setting
            for i, delay_name in enumerate(delay_order):
                delay_df = df_sens[df_sens['delay_setting'] == delay_name]
                
                # Filter for current p_geom and get means across scenarios
                p_geom_df = delay_df[delay_df['p_geom'] == p_geom]
                metric_means = p_geom_df.groupby('theta_sigma')[metric].mean()
                
                line, = ax.plot(theta_sigma_values, 
                       [metric_means.get(val, np.nan) for val in theta_sigma_values],
                       marker=markers[i], linestyle='-', color=colors[i], 
                       linewidth=2.5, markersize=7,
                       label=delay_name, alpha=0.8)
                
                # Collect legend info only from first subplot
                if row == 0 and col == 0:
                    legend_handles.append(line)
                    legend_labels.append(delay_name)
            
            # Only add title to top row (p_geom labels)
            if row == 0:
                ax.set_title(f'p_geom = {p_geom}', fontsize=14, fontweight='bold', pad=8)
            
            # Only add y-label to leftmost column
            if col == 0:
                ax.set_ylabel(metric_titles[row], fontsize=13, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            # Only add x-label to bottom row
            if row == 2:
                ax.set_xlabel('θ_σ', fontsize=13, fontweight='bold')
            else:
                ax.set_xlabel('')
            
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=1.0)
            ax.tick_params(labelsize=11)
            
            # Set prominent black borders
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2.0)
                spine.set_visible(True)
    
    # Add a single legend outside the plot area
    fig.legend(legend_handles, legend_labels, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 0.91),
              ncol=3, 
              fontsize=13, 
              frameon=True, 
              fancybox=True, 
              shadow=True,
              edgecolor='black',
              framealpha=1.0)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    save_path = os.path.join(PLOTS_DIR, "sensitivity_line_plots.pdf")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    png_path = os.path.join(PLOTS_DIR, "sensitivity_line_plots.png")
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Sensitivity line plots saved at: {save_path}")

def generate_sensitivity_comparison_plot(df_sens):
    """
    Generates a comprehensive comparison plot showing the impact of delay misspecification.
    
    Args:
        df_sens (pd.DataFrame): The DataFrame from load_and_process_sensitivity_results.
    """
    delay_order = list(DELAY_DIST_SENSITIVITY.keys())
    metrics = ['accuracy', 'hausdorff', 'mae']
    metric_titles = ['Accuracy', 'Hausdorff Distance', 'Mean Absolute Error']
    
    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'Impact of Delay Misspecification on RJMCMC Performance',
        fontsize=20,
        fontweight='bold',
        y=0.96
    )
    
    # Define colors for different delay settings
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    for i, delay_name in enumerate(delay_order):
        delay_df = df_sens[df_sens['delay_setting'] == delay_name]
        accuracy_means = delay_df.groupby(['p_geom', 'theta_sigma'])['accuracy'].mean().reset_index()
        mean_accuracy = accuracy_means['accuracy'].mean()
        
        ax.bar(i, mean_accuracy, color=colors[i], alpha=0.8, 
               edgecolor='black', linewidth=1.5, 
               label=f'{delay_name}\n(mean: {mean_accuracy:.3f})')
    
    ax.set_title('Mean Accuracy by Delay Setting', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(delay_order)))
    ax.set_xticklabels([name.split()[0] for name in delay_order], rotation=45)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Plot 2: Hausdorff Distance comparison
    ax = axes[0, 1]
    for i, delay_name in enumerate(delay_order):
        delay_df = df_sens[df_sens['delay_setting'] == delay_name]
        hd_means = delay_df.groupby(['p_geom', 'theta_sigma'])['hausdorff'].mean().reset_index()
        mean_hd = hd_means['hausdorff'].mean()
        
        ax.bar(i, mean_hd, color=colors[i], alpha=0.8, 
               edgecolor='black', linewidth=1.5,
               label=f'{delay_name}\n(mean: {mean_hd:.2f})')
    
    ax.set_title('Mean Hausdorff Distance by Delay Setting', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hausdorff Distance', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(delay_order)))
    ax.set_xticklabels([name.split()[0] for name in delay_order], rotation=45)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Plot 3: MAE comparison
    ax = axes[1, 0]
    for i, delay_name in enumerate(delay_order):
        delay_df = df_sens[df_sens['delay_setting'] == delay_name]
        mae_means = delay_df.groupby(['p_geom', 'theta_sigma'])['mae'].mean().reset_index()
        mean_mae = mae_means['mae'].mean()
        
        ax.bar(i, mean_mae, color=colors[i], alpha=0.8, 
               edgecolor='black', linewidth=1.5,
               label=f'{delay_name}\n(mean: {mean_mae:.3f})')
    
    ax.set_title('Mean Absolute Error by Delay Setting', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(delay_order)))
    ax.set_xticklabels([name.split()[0] for name in delay_order], rotation=45)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Plot 4: Combined performance score
    ax = axes[1, 1]
    for i, delay_name in enumerate(delay_order):
        delay_df = df_sens[df_sens['delay_setting'] == delay_name]
        
        # Normalize metrics to 0-1 scale and create combined score
        accuracy_means = delay_df.groupby(['p_geom', 'theta_sigma'])['accuracy'].mean().reset_index()
        hd_means = delay_df.groupby(['p_geom', 'theta_sigma'])['hausdorff'].mean().reset_index()
        mae_means = delay_df.groupby(['p_geom', 'theta_sigma'])['mae'].mean().reset_index()
        
        # Normalize (higher is better for accuracy, lower is better for others)
        mean_acc = accuracy_means['accuracy'].mean()
        mean_hd = hd_means['hausdorff'].mean()
        mean_mae = mae_means['mae'].mean()
        
        # Create combined score (higher is better)
        combined_score = mean_acc - (mean_hd / 100) - (mean_mae / 10)
        
        ax.bar(i, combined_score, color=colors[i], alpha=0.8, 
               edgecolor='black', linewidth=1.5,
               label=f'{delay_name}\n(score: {combined_score:.3f})')
    
    ax.set_title('Combined Performance Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Combined Score', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(delay_order)))
    ax.set_xticklabels([name.split()[0] for name in delay_order], rotation=45)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Set prominent black borders for all subplots
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.0)
            spine.set_visible(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(PLOTS_DIR, "sensitivity_delay_comparison.pdf")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    png_path = os.path.join(PLOTS_DIR, "sensitivity_delay_comparison.png")
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Sensitivity delay comparison saved at: {save_path}")

def generate_sensitivity_bar_plots(df_sens):
    """
    Generates bar plots showing the best hyperparameter combinations for each delay setting.
    
    Args:
        df_sens (pd.DataFrame): The DataFrame from load_and_process_sensitivity_results.
    """
    delay_order = list(DELAY_DIST_SENSITIVITY.keys())
    metrics = ['accuracy', 'hausdorff', 'mae']
    
    # Create a 1x3 subplot for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Best Hyperparameter Combinations by Delay Setting',
        fontsize=20,
        fontweight='bold',
        y=0.98
    )
    
    for col, metric in enumerate(metrics):
        ax = axes[col]
        
        best_combinations = []
        delay_names = []
        
        for delay_name in delay_order:
            delay_df = df_sens[df_sens['delay_setting'] == delay_name]
            
            # Group by hyperparameters and average across scenarios
            grouped = delay_df.groupby(['p_geom', 'theta_sigma'])[metric].mean().reset_index()
            
            # Find best combination (highest for accuracy, lowest for others)
            if metric == 'accuracy':
                best_idx = grouped[metric].idxmax()
            else:
                best_idx = grouped[metric].idxmin()
            
            best_combo = grouped.loc[best_idx]
            best_combinations.append(best_combo[metric])
            delay_names.append(delay_name)
        
        # Create bar plot
        bars = ax.bar(delay_names, best_combinations, 
                     color=['tab:blue', 'tab:orange', 'tab:green'], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, best_combinations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        # Customize subplot
        ax.set_title(f'Best {metric.title()} by Delay Setting', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.title(), fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')
        ax.tick_params(axis='x', rotation=45)
        
        # Set prominent black borders
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.0)
            spine.set_visible(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(PLOTS_DIR, "sensitivity_best_combinations.pdf")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    png_path = os.path.join(PLOTS_DIR, "sensitivity_best_combinations.png")
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Sensitivity best combinations saved at: {save_path}")

def generate_sensitivity_heatmap_grid(df_sens):
    """
    Generates a 3x3 grid of heatmaps for the sensitivity analysis.
    Each subplot shows a different metric vs. delay assumption.

    Args:
        df_sens (pd.DataFrame): The DataFrame from load_and_process_sensitivity_results.
    """
    delay_order = list(DELAY_DIST_SENSITIVITY.keys())
    metrics = ['accuracy', 'hausdorff', 'mae']
    metric_titles = ['Accuracy P(k_est = k_true)', 'Hausdorff Distance (HD)', 'Mean Absolute Error (MAE)']

    # Determine consistent color scale per-metric for readability/comparability
    metric_vmin = {m: np.nanmin(df_sens[m]) for m in metrics}
    metric_vmax = {m: np.nanmax(df_sens[m]) for m in metrics}

    # Larger canvas for publication and bigger base fonts
    fig, axes = plt.subplots(3, 3, figsize=(26, 22))
    fig.suptitle(
        'Sensitivity of RJMCMC Performance to Priors and Delay Misspecification',
        fontsize=28,
        fontweight='bold',
        y=0.995
    )

    p_geom_labels = SENSITIVITY_GRID_PRIORS['PRIOR_K_GEOMETRIC_P']
    theta_sigma_labels = SENSITIVITY_GRID_PRIORS['PRIOR_THETA_SIGMA']

    for row, delay_name in enumerate(delay_order):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            
            # Filter data for the current delay setting
            delay_df = df_sens[df_sens['delay_setting'] == delay_name]
            
            # Group by hyperparameters and average the metric over all 5 scenarios
            pivot_df = delay_df.groupby(['p_geom', 'theta_sigma'])[metric].mean().reset_index()
            pivot_table = pivot_df.pivot(index='theta_sigma', columns='p_geom', values=metric)
            
            # Plot the heatmap with larger text and clearer styling
            heat = sns.heatmap(
                pivot_table,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                linewidths=1.5,
                linecolor='white',
                cbar_kws={'shrink': 0.9},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                vmin=metric_vmin[metric],
                vmax=metric_vmax[metric],
                square=True
            )
            
            # Set titles and labels for clarity
            if row == 0:
                ax.set_title(metric_titles[col], fontsize=20, fontweight='bold', pad=10)
            if col == 0:
                ax.set_ylabel(delay_name, fontsize=18, fontweight='bold')
            else:
                ax.set_ylabel('')

            ax.set_xlabel('p_geom', fontsize=18, fontweight='bold')
            ax.set_xticklabels(p_geom_labels, rotation=0, fontsize=14)
            ax.set_yticklabels(theta_sigma_labels, rotation=0, fontsize=14)
            ax.tick_params(axis='both', labelsize=14)

            # Set prominent black borders for all spines
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2.5)
                spine.set_visible(True)

            # Enlarge colorbar tick labels
            if heat.collections:
                cbar = heat.collections[0].colorbar
                if cbar is not None:
                    cbar.ax.tick_params(labelsize=14)
                    # Add border to colorbar
                    cbar.outline.set_edgecolor('black')
                    cbar.outline.set_linewidth(2.5)

    # Improve layout spacing; export high-res PDF and PNG
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    save_path = os.path.join(PLOTS_DIR, "sensitivity_analysis_heatmap_grid.pdf")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    png_path = os.path.join(PLOTS_DIR, "sensitivity_analysis_heatmap_grid.png")
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Sensitivity analysis heatmap grid saved at: {save_path}")

def full_analysis_workflow():
    """The full, two-stage analysis workflow."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Stage 1: Analyze sensitivity results
    df_sens = load_and_process_sensitivity_results(SENSITIVITY_RESULTS_DIR)
    
    # Generate multiple sensitivity visualizations
    print("Generating sensitivity analysis visualizations...")
    # generate_sensitivity_heatmap_grid(df_sens)      # Heatmap generation disabled
    generate_sensitivity_line_plots(df_sens)        # Line plots
    
    optimal_params = find_optimal_hyperparameters(df_sens)
    
    # Stage 2: Analyze main simulation results
    df_main = load_and_process_main_results(MAIN_RESULTS_DIR)
    generate_main_results_table(df_main)
    generate_publication_figure(MAIN_RESULTS_DIR, optimal_params)

    print("\n--- Analysis Complete ---")
    print("Generated sensitivity visualizations:")
    print("  1. sensitivity_line_plots.pdf (line plots)")
    return optimal_params

if __name__ == "__main__":
    full_analysis_workflow()
