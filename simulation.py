# simulation.py

import os
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from config import SCENARIOS, N_REPLICATIONS, RESULTS_DIR
from data_generation import generate_dataset
from methods import run_rjmcmc, run_pelt, run_binseg

def setup_directories():
    """Create directories for results if they don't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def run_simulation_worker(task):
    """
    Main worker function to be run on each CPU core.
    Handles data generation, running a method, and saving the result.
    """
    scenario_name, rep_idx, method_name = task
    
    # Define checkpoint file path
    result_file = os.path.join(RESULTS_DIR, f"res_{scenario_name.replace(' ', '_')}_rep{rep_idx}_{method_name}.pkl")
    
    # --- Checkpoint: Skip if already completed ---
    if os.path.exists(result_file):
        return {"task": task, "status": "skipped"}

    try:
        # Generate data
        scenario_config = SCENARIOS[scenario_name]
        data = generate_dataset(scenario_config, rep_idx)
        
        # Run the specified method
        if method_name == "RJMCMC":
            result = run_rjmcmc(data)
        elif method_name == "PELT":
            result = run_pelt(data)
        elif method_name == "BinSeg":
            result = run_binseg(data)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Add true values for later analysis
        result['true_cps'] = data['true_cps']
        
        # Save the result to a checkpoint file
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
            
        return {"task": task, "status": "completed"}
    except Exception as e:
        return {"task": task, "status": "failed", "error": str(e)}

def main():
    """Main function to run the entire simulation study in parallel."""
    setup_directories()
    
    methods_to_run = ["RJMCMC", "PELT", "BinSeg"]
    
    # Create a list of all tasks (all combinations of scenarios, reps, methods)
    tasks = [
        (scenario_name, rep_idx, method_name)
        for scenario_name in SCENARIOS
        for rep_idx in range(N_REPLICATIONS)
        for method_name in methods_to_run
    ]
    
    print(f"Starting simulation study with {len(tasks)} total tasks.")
    print(f"Using {cpu_count()} CPU cores.")
    
    # --- Parallel Execution with Progress Bar ---
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(run_simulation_worker, tasks), total=len(tasks)))
    
    # --- Report Summary ---
    completed = sum(1 for r in results if r["status"] == "completed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print("\n--- Simulation Finished ---")
    print(f"Completed: {completed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\n--- Failed Tasks ---")
        for r in results:
            if r["status"] == "failed":
                print(f"Task: {r['task']}, Error: {r['error']}")

if __name__ == "__main__":
    main()
