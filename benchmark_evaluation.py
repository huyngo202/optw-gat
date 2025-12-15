#!/usr/bin/env python
"""
Benchmark Evaluation Script
Compares performance of Baseline, Transformer, and Transformer PPO models.
"""
import subprocess
import sys
import time
import re
import csv
import os
from datetime import datetime

# Configuration
INSTANCES = ['c101', 'r101', 'rc101', 'pr01', 't101']
MODELS = {
    'Baseline': 'baseline_bench',
    'Transformer': 'transformer_bench'
}
SAMPLE_TYPE = 'uni_samp'
INFERENCE_TYPE = 'bs' # Beam Search
PYTHON = sys.executable
OUTPUT_CSV = 'benchmark_results_comparison.csv'

def parse_output(output_str):
    """Extracts score and inference time from stdout."""
    score_match = re.search(r'total score:\s+(\d+)', output_str)
    time_match = re.search(r'inference time:\s+(\d+)\s+ms', output_str)
    
    score = int(score_match.group(1)) if score_match else None
    inf_time = int(time_match.group(1)) if time_match else None
    
    return score, inf_time

def get_latest_epoch(instance, model_name):
    """Finds the latest model epoch in the results directory."""
    base_dir = f'results/{instance}/model_w/model_{model_name}_{SAMPLE_TYPE}'
    if not os.path.exists(base_dir):
        return None
    
    files = os.listdir(base_dir)
    epochs = []
    for f in files:
        match = re.search(r'model_(\d+).pkl', f)
        if match:
            epochs.append(int(match.group(1)))
            
    if not epochs:
        return None
        
    return max(epochs)

def run_benchmark(instance, model_label, model_name):
    """Runs inference for a specific instance and model."""
    print(f"Running {model_label} on {instance}...")
    
    # Determine epoch
    epoch = 500000 # Default for pre-trained models
    
    # Check if default exists, if not find latest
    default_path = f'results/{instance}/model_w/model_{model_name}_{SAMPLE_TYPE}/model_{epoch}.pkl'
    if not os.path.exists(default_path):
        latest_epoch = get_latest_epoch(instance, model_name)
        if latest_epoch:
            print(f"  Default model_{epoch}.pkl not found. Using latest: model_{latest_epoch}.pkl")
            epoch = latest_epoch
        else:
            print(f"  No model files found for {model_label} on {instance}")
            return None, None

    cmd = [
        PYTHON, 'inference_optw_rl.py',
        '--instance', instance,
        '--model_name', model_name,
        '--sample_type', SAMPLE_TYPE,
        '--infe_type', INFERENCE_TYPE,
        '--device', 'cpu',
        '--batch_size', '1',
        '--saved_model_epoch', str(epoch)
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        score, inf_time = parse_output(result.stderr)
        
        if score is None:
            score, inf_time = parse_output(result.stdout)
            
        return score, inf_time
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {model_label} on {instance}: {e}")
        # print(e.stderr) # Reduce noise
        return None, None

def main():
    print("="*60)
    print("BENCHMARK EVALUATION STARTED")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = []
    
    # Prepare CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['Instance', 'Model', 'Score', 'Inference Time (ms)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for instance in INSTANCES:
            print(f"\nProcessing Instance: {instance}")
            print("-" * 40)
            
            for model_label, model_name in MODELS.items():
                score, inf_time = run_benchmark(instance, model_label, model_name)
                
                if score is not None:
                    print(f"  -> {model_label}: Score={score}, Time={inf_time}ms")
                    writer.writerow({
                        'Instance': instance,
                        'Model': model_label,
                        'Score': score,
                        'Inference Time (ms)': inf_time
                    })
                    csvfile.flush()
                else:
                    print(f"  -> {model_label}: FAILED")

    print("\n" + "="*60)
    print(f"Benchmark completed. Results saved to {OUTPUT_CSV}")
    print("="*60)

if __name__ == "__main__":
    main()
