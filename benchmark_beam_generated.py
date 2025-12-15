import subprocess
import pandas as pd
import re
import os
import sys

PYTHON_EXEC = sys.executable
INSTANCES = ['c101', 'r101', 'rc101', 'pr01', 't101']
MODELS = ['gat_transformer_bench'] # Only GAT-Trans for now as requested? Or keep all to compare. Let's keep all.
MODELS = ['gat_transformer_bench']
BEAM_SIZE = 10
RESULTS_FILE = 'results/generated_beam_results.csv'
INFE_TYPE = 'bs'

import glob

def get_latest_epoch(instance, model_name):
    # Path: results/{instance}/model_w/model_{model_name}_uni_samp/model_{epoch}.pkl
    model_dir = f"results/{instance}/model_w/model_{model_name}_uni_samp"
    files = glob.glob(f"{model_dir}/model_*.pkl")
    
    if not files:
        return None
        
    max_epoch = 0
    for f in files:
        match = re.search(r'model_(\d+).pkl', f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
    return max_epoch

def run_inference(instance, model_name):
    epoch = get_latest_epoch(instance, model_name)
    if epoch is None:
        print(f"No checkpoint found for {instance} {model_name}")
        return None, None

    cmd = [
        PYTHON_EXEC, 'inference_optw_rl.py',
        '--instance', instance,
        '--model_name', model_name,
        '--infe_type', INFE_TYPE,
        '--max_beam_number', str(BEAM_SIZE),
        '--saved_model_epoch', str(epoch),
        '--device', 'cpu',
        '--generated' # Added flag
    ]

    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stderr # Logging usually goes to stderr
        
        # Parse output for AVERAGES
        # "average total score: 257.5"
        # "average inference time: 1234 ms"
        score_match = re.search(r'average total score:\s+([\d\.]+)', output)
        time_match = re.search(r'average inference time:\s+(\d+)\s+ms', output)
        
        score = float(score_match.group(1)) if score_match else None
        time_ms = int(time_match.group(1)) if time_match else None
        
        return score, time_ms
        
    except subprocess.CalledProcessError as e:
        print(f"Error running inference for {instance} {model_name}")
        print(e.stderr)
        return None, None

def main():
    results = []
    
    print(f"{'Instance':<10} {'Model':<20} {'Score':<10} {'Time (ms)':<10}")
    print("-" * 55)
    
    for inst in INSTANCES:
        for model in MODELS:
            score, time_ms = run_inference(inst, model)
            
            if score is not None:
                print(f"{inst:<10} {model:<20} {score:<10} {time_ms:<10}")
                results.append({
                    'Instance': inst,
                    'Model': model,
                    'Beam_Score': score,
                    'Beam_Time_ms': time_ms
                })
            else:
                print(f"{inst:<10} {model:<20} {'FAILED':<10} {'FAILED':<10}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_FILE, index=False)
        print(f"\nSaved results to {RESULTS_FILE}")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    main()
