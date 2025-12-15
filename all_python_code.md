# Gom mã nguồn Python từ `/home/huyngo/Project/ML/optw_rl`

Generated: 2025-12-14 09:55:44

## `analyze_logs.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/analyze_logs.py`
- **Size**: 4357 bytes
- **Last modified**: 2025-11-24 22:13:40

```python
import os
import glob
import pandas as pd
import numpy as np
import re

RESULTS_DIR = './results'

def get_training_speed(instance, model_name):
    """
    Estimates seconds per epoch based on model checkpoint timestamps.
    Returns median seconds per epoch.
    """
    model_dir = f"{RESULTS_DIR}/{instance}/model_w/model_{model_name}_uni_samp"
    if not os.path.exists(model_dir):
        return None

    files = glob.glob(f"{model_dir}/model_*.pkl")
    checkpoints = []
    
    for f in files:
        match = re.search(r'model_(\d+).pkl', f)
        if match:
            epoch = int(match.group(1))
            mtime = os.path.getmtime(f)
            checkpoints.append({'epoch': epoch, 'mtime': mtime})
    
    if len(checkpoints) < 2:
        return None
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x['epoch'])
    
    speeds = []
    for i in range(len(checkpoints) - 1):
        c1 = checkpoints[i]
        c2 = checkpoints[i+1]
        
        delta_epoch = c2['epoch'] - c1['epoch']
        delta_time = c2['mtime'] - c1['mtime']
        
        if delta_epoch > 0 and delta_time > 0:
            speed = delta_time / delta_epoch
            # Filter out unreasonable speeds (e.g. if files were just copied/touched instantly, speed would be ~0)
            # Also filter out if delta_time is huge (e.g. days later)
            if speed > 0.01 and speed < 100: # Assuming 0.01s to 100s per epoch is reasonable range
                speeds.append(speed)
                
    if not speeds:
        return None
        
    return np.median(speeds)

def get_convergence_metrics(instance, model_name):
    """
    Returns convergence epoch and max reward.
    """
    csv_path = f"{RESULTS_DIR}/{instance}/outputs/model_{model_name}_uni_samp/training_history.csv"
    if not os.path.exists(csv_path):
        return None
        
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
            
        # Find row with max validation reward
        # Note: using 'avg_reward_val_uni_samp' as the metric
        best_row = df.loc[df['avg_reward_val_uni_samp'].idxmax()]
        
        return {
            'convergence_epoch': int(best_row['epoch']),
            'max_reward': best_row['avg_reward_val_uni_samp'],
            'final_epoch': int(df['epoch'].iloc[-1])
        }
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def main():
    instances = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    instances = sorted(instances)
    
    results = []
    
    print(f"{'Instance':<10} {'Model':<12} {'Speed (s/ep)':<15} {'Conv. Epoch':<12} {'Time to Conv (m)':<18} {'Max Reward':<12}")
    print("-" * 85)
    
    for inst in instances:
        for model_type in ['baseline', 'transformer']:
            model_name = f"{model_type}_bench"
            
            speed = get_training_speed(inst, model_name)
            metrics = get_convergence_metrics(inst, model_name)
            
            if metrics:
                conv_epoch = metrics['convergence_epoch']
                max_reward = metrics['max_reward']
                
                time_to_conv_min = "N/A"
                speed_str = "N/A"
                
                if speed:
                    speed_str = f"{speed:.3f}"
                    time_to_conv_sec = speed * conv_epoch
                    time_to_conv_min = f"{time_to_conv_sec / 60:.1f}"
                
                print(f"{inst:<10} {model_type:<12} {speed_str:<15} {conv_epoch:<12} {time_to_conv_min:<18} {max_reward:.2f}")
                
                results.append({
                    'Instance': inst,
                    'Model': model_type,
                    'Speed_s_per_epoch': speed,
                    'Convergence_Epoch': conv_epoch,
                    'Time_to_Convergence_min': time_to_conv_min,
                    'Max_Reward': max_reward
                })
            else:
                 print(f"{inst:<10} {model_type:<12} {'N/A':<15} {'N/A':<12} {'N/A':<18} {'N/A':<12}")

    # Save detailed results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv('training_analysis.csv', index=False)
        print("\nSaved analysis to training_analysis.csv")

if __name__ == "__main__":
    main()

```

## `benchmark_beam_search.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/benchmark_beam_search.py`
- **Size**: 3074 bytes
- **Last modified**: 2025-11-24 22:44:23

```python
import subprocess
import pandas as pd
import re
import os
import sys

PYTHON_EXEC = sys.executable
INSTANCES = ['c101', 'r101', 'rc101', 'pr01', 't101']
MODELS = ['baseline_bench', 'transformer_bench']
BEAM_SIZE = 128
RESULTS_FILE = 'results/greedy_results.csv'
INFE_TYPE = 'gr'

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
        # '--max_beam_number', str(BEAM_SIZE), # Not needed for greedy but harmless
        '--saved_model_epoch', str(epoch),
        '--device', 'cpu'
    ]

    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stderr # Logging usually goes to stderr
        
        # Parse output
        # Looking for: "total score: 257" and "inference time: 1234 ms"
        score_match = re.search(r'total score:\s+(\d+)', output)
        time_match = re.search(r'inference time:\s+(\d+)\s+ms', output)
        
        score = int(score_match.group(1)) if score_match else None
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

```

## `benchmark_evaluation.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/benchmark_evaluation.py`
- **Size**: 4411 bytes
- **Last modified**: 2025-11-27 19:58:01

```python
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

```

## `benchmark_gat_runner.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/benchmark_gat_runner.py`
- **Size**: 4484 bytes
- **Last modified**: 2025-11-27 00:06:42

```python
import os
import subprocess
import argparse
import time
import sys

# Configuration
PYTHON_EXEC = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'benchmark')

# Tập instance mẫu dùng để benchmark nhanh
SUBSET_INSTANCES = ['c101', 'r101', 'rc101', 'pr01', 't101']

def get_all_instances():
    instances = []
    if not os.path.exists(DATA_DIR):
        # Fallback nếu không tìm thấy thư mục data, dùng subset
        return SUBSET_INSTANCES
    
    for f in os.listdir(DATA_DIR):
        if f.endswith('.txt') and f not in ['TOPTW_format.txt', 'format_description_TTDP.txt']:
            instances.append(f.replace('.txt', ''))
    return sorted(instances)

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(e)
        return False

def benchmark_gat(instance, model_type, epochs=1000, batch_size=16, device='cpu'):
    """
    model_type: 'gat_lstm' hoặc 'gat_transformer'
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking GAT ({model_type}) - Instance: {instance}")
    print(f"{'='*60}\n")

    # 1. Đảm bảo dữ liệu validation đã được sinh ra
    # Kiểm tra xem file validation đã tồn tại chưa để tránh sinh lại nhiều lần không cần thiết
    val_path = os.path.join(BASE_DIR, 'data', 'generated_instances', instance, 'inp_val_uni_samp.pt')
    if not os.path.exists(val_path):
        print(f"--- Generating Validation Data for {instance} ---")
        # Dùng generate_instances_transformer.py vì nó tương thích tốt với format mới
        cmd_gen = f"{PYTHON_EXEC} generate_instances_transformer.py --instance {instance} --ni 64 --device {device}"
        run_command(cmd_gen)
    else:
        print(f"--- Validation Data for {instance} already exists. Skipping generation. ---")

    # 2. Train GAT Model
    script_name = f"train_optw_{model_type}.py"
    model_name_suffix = f"{model_type}_bench"
    
    # Mapping args đặc thù nếu cần (hiện tại logic giống nhau)
    print(f"\n--- Training {model_type} for {instance} ---")
    
    cmd_train = (
        f"{PYTHON_EXEC} {script_name} "
        f"--instance {instance} "
        f"--model_name {model_name_suffix} "
        f"--nepocs {epochs} "
        f"--batch_size {batch_size} "
        f"--nprint 100 "
        f"--nsave 100 " # Save checkpoint mỗi 2000 epoch
        f"--device {device} "
        f"--n_gat_layers 3" # Cấu hình số lớp GAT, có thể thay đổi
    )
    
    return run_command(cmd_train)

def main():
    parser = argparse.ArgumentParser(description='GAT Benchmark Runner')
    parser.add_argument('--mode', choices=['subset', 'all'], default='subset', help='Run on subset or all instances')
    parser.add_argument('--type', choices=['gat_lstm', 'gat_transformer', 'both'], default='both', help='Which GAT architecture to run')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs per training run')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()

    if args.mode == 'subset':
        instances_to_run = SUBSET_INSTANCES
    else:
        instances_to_run = get_all_instances()

    # Xác định các model cần chạy
    models_to_run = []
    if args.type == 'both':
        models_to_run = ['gat_lstm', 'gat_transformer']
    else:
        models_to_run = [args.type]

    print(f"Starting GAT benchmark on {len(instances_to_run)} instances...")
    print(f"Models: {models_to_run}")
    print(f"Device: {args.device}")
    
    start_time = time.time()
    results = {}

    for inst in instances_to_run:
        for model in models_to_run:
            success = benchmark_gat(inst, model, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
            key = f"{inst}_{model}"
            results[key] = "SUCCESS" if success else "FAILED"

    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("GAT BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total time: {elapsed/60:.2f} minutes")
    for k, v in results.items():
        print(f"{k:<30}: {v}")
    print("="*60)

if __name__ == "__main__":
    main()
```

## `benchmark_ils.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/benchmark_ils.py`
- **Size**: 1292 bytes
- **Last modified**: 2025-11-24 22:28:23

```python
import pandas as pd
import time
from src.ils import OPTW_ILS
import os

INSTANCES = ['c101', 'r101', 'rc101', 'pr01', 't101']
RESULTS_FILE = 'benchmark_summary.csv' # We will append or merge to this

def run_benchmark():
    results = []
    
    print(f"{'Instance':<10} {'ILS Profit':<10} {'Time (s)':<10}")
    print("-" * 35)
    
    for inst in INSTANCES:
        # Run ILS
        # 1000 iterations or 60 seconds max
        ils = OPTW_ILS(inst, max_iter=2000, time_limit=30) 
        start = time.time()
        profit, route = ils.solve()
        elapsed = time.time() - start
        
        print(f"{inst:<10} {profit:<10.2f} {elapsed:<10.2f}")
        
        results.append({
            'Instance': inst,
            'ILS_Max_Reward': profit,
            'ILS_Time': elapsed
        })
        
    df_ils = pd.DataFrame(results)
    
    # Merge with existing benchmark summary if exists
    if os.path.exists(RESULTS_FILE):
        df_existing = pd.read_csv(RESULTS_FILE)
        # Merge on Instance
        df_final = pd.merge(df_existing, df_ils, on='Instance', how='outer')
    else:
        df_final = df_ils
        
    df_final.to_csv(RESULTS_FILE, index=False)
    print(f"\nUpdated {RESULTS_FILE} with ILS results.")

if __name__ == "__main__":
    run_benchmark()

```

## `benchmark_inference_gat.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/benchmark_inference_gat.py`
- **Size**: 6141 bytes
- **Last modified**: 2025-11-27 20:05:13

```python
#!/usr/bin/env python
"""
Inference Benchmark for GAT Models
Compares inference performance (score and time) of Baseline, Transformer, GAT-LSTM, and GAT-Transformer models.
"""
import subprocess
import sys
import os
import re
import csv
import argparse
from datetime import datetime

# Configuration
INSTANCES = ['c101', 'r101', 'rc101', 'pr01', 't101']
MODELS = {
    'Baseline': 'baseline_bench',
    'Transformer': 'transformer_bench',
    'GAT-LSTM': 'gat_lstm',
    'GAT-Transformer': 'gat_transformer_bench'
}
SAMPLE_TYPE = 'uni_samp'
PYTHON = sys.executable
RESULTS_DIR = 'results'

def find_latest_epoch(instance, model_name):
    """Find the latest available model epoch for a given instance and model."""
    model_dir = f'{RESULTS_DIR}/{instance}/model_w/model_{model_name}_{SAMPLE_TYPE}'
    
    if not os.path.exists(model_dir):
        return None
    
    # Find all .pkl files
    pkl_files = []
    for f in os.listdir(model_dir):
        if f.endswith('.pkl') and f.startswith('model_'):
            match = re.search(r'model_(\d+)\.pkl', f)
            if match:
                pkl_files.append(int(match.group(1)))
    
    if not pkl_files:
        return None
    
    return max(pkl_files)

def parse_inference_output(output_str):
    """Parse score and inference time from inference output."""
    score_match = re.search(r'total score:\s+(\d+)', output_str)
    time_match = re.search(r'inference time:\s+(\d+)\s+ms', output_str)
    
    score = int(score_match.group(1)) if score_match else None
    inf_time = int(time_match.group(1)) if time_match else None
    
    return score, inf_time

def run_inference(instance, model_label, model_name, epoch, infe_type='bs', device='cpu'):
    """Run inference for a specific instance and model."""
    print(f"\n{'='*60}")
    print(f"Running {model_label} on {instance} (epoch {epoch})...")
    print(f"{'='*60}")
    
    cmd = [
        PYTHON, 'inference_optw_rl.py',
        '--instance', instance,
        '--model_name', model_name,
        '--sample_type', SAMPLE_TYPE,
        '--infe_type', infe_type,
        '--device', device,
        '--batch_size', '1',
        '--saved_model_epoch', str(epoch)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse output from both stdout and stderr
        score, inf_time = parse_inference_output(result.stderr)
        if score is None:
            score, inf_time = parse_inference_output(result.stdout)
        
        if score is not None:
            print(f"✓ Score: {score}, Time: {inf_time}ms")
            return score, inf_time
        else:
            print(f"✗ Failed to parse output")
            return None, None
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout (>5 minutes)")
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        return None, None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='GAT Inference Benchmark')
    parser.add_argument('--instances', nargs='+', default=INSTANCES, 
                        help='Instances to benchmark')
    parser.add_argument('--infe_type', choices=['gr', 'bs'], default='bs',
                        help='Inference type: greedy (gr) or beam search (bs)')
    parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--output', default='inference_benchmark_results.csv',
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GAT INFERENCE BENCHMARK")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Inference Type: {args.infe_type}")
    print(f"Device: {args.device}")
    print("="*60)
    
    results = []
    
    for instance in args.instances:
        print(f"\n{'#'*60}")
        print(f"# Instance: {instance}")
        print(f"{'#'*60}")
        
        for model_label, model_name in MODELS.items():
            # Find latest epoch
            epoch = find_latest_epoch(instance, model_name)
            
            if epoch is None:
                print(f"\n{model_label}: No model found - SKIPPED")
                results.append({
                    'Instance': instance,
                    'Model': model_label,
                    'Epoch': 'N/A',
                    'Score': 'N/A',
                    'Inference_Time_ms': 'N/A'
                })
                continue
            
            # Run inference
            score, inf_time = run_inference(
                instance, model_label, model_name, epoch,
                args.infe_type, args.device
            )
            
            results.append({
                'Instance': instance,
                'Model': model_label,
                'Epoch': epoch if score is not None else 'N/A',
                'Score': score if score is not None else 'FAILED',
                'Inference_Time_ms': inf_time if inf_time is not None else 'FAILED'
            })
    
    # Save to CSV
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['Instance', 'Model', 'Epoch', 'Score', 'Inference_Time_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Results saved to {args.output}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Instance':<10} | {'Model':<16} | {'Epoch':<8} | {'Score':<8} | {'Time(ms)':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['Instance']:<10} | {r['Model']:<16} | {str(r['Epoch']):<8} | "
              f"{str(r['Score']):<8} | {str(r['Inference_Time_ms']):<10}")
    
    print("="*60)

if __name__ == "__main__":
    main()

```

## `benchmark_report.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/benchmark_report.py`
- **Size**: 4698 bytes
- **Last modified**: 2025-11-27 19:58:08

```python
import os
import pandas as pd
import glob
import argparse
import matplotlib.pyplot as plt

RESULTS_DIR = './results'

def get_latest_metrics(instance, model_name):
    # Path pattern: results/{instance}/outputs/model_{model_name}_uni_samp/training_history.csv
    path = f"{RESULTS_DIR}/{instance}/outputs/model_{model_name}_uni_samp/training_history.csv"
    
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        
        # Get max reward and final loss
        max_reward = df['avg_reward_val_uni_samp'].max()
        final_loss = df['tloss_train'].iloc[-1]
        return {'max_reward': max_reward, 'final_loss': final_loss}
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def plot_training_curves(df_baseline, df_transformer, instance):
    try:
        plt.figure(figsize=(10, 6))
        
        if df_baseline is not None:
            plt.plot(df_baseline['epoch'], df_baseline['avg_reward_val_uni_samp'], label='Baseline', linestyle='-', alpha=0.8)
        if df_transformer is not None:
            plt.plot(df_transformer['epoch'], df_transformer['avg_reward_val_uni_samp'], label='Transformer', linestyle='-', alpha=0.8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Avg Validation Reward')
        plt.title(f'Training Progress: {instance}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f'training_curve_{instance}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Saved training curve to {filename}")
        return filename
    except Exception as e:
        print(f"Error plotting curves for {instance}: {e}")
        return None

def generate_report():
    instances = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    instances = sorted(instances)
    
    data = []
    
    for inst in instances:
        # Get latest metrics
        baseline_metrics = get_latest_metrics(inst, 'baseline_bench')
        transformer_metrics = get_latest_metrics(inst, 'transformer_bench')
        
        # Load full dataframes for plotting
        path_base = f"{RESULTS_DIR}/{inst}/outputs/model_baseline_bench_uni_samp/training_history.csv"
        path_trans = f"{RESULTS_DIR}/{inst}/outputs/model_transformer_bench_uni_samp/training_history.csv"
        
        df_base = pd.read_csv(path_base) if os.path.exists(path_base) else None
        df_trans = pd.read_csv(path_trans) if os.path.exists(path_trans) else None
        
        if any(df is not None for df in [df_base, df_trans]):
            plot_training_curves(df_base, df_trans, inst)

        row = {'Instance': inst}
        
        if baseline_metrics:
            row['Baseline_Max_Reward'] = baseline_metrics['max_reward']
        else:
            row['Baseline_Max_Reward'] = 0
            
        if transformer_metrics:
            row['Transformer_Max_Reward'] = transformer_metrics['max_reward']
        else:
            row['Transformer_Max_Reward'] = 0
            
        # Calculate Gaps
        if row['Baseline_Max_Reward'] > 0:
            row['Gap_Trans_Base'] = ((row['Transformer_Max_Reward'] - row['Baseline_Max_Reward']) / row['Baseline_Max_Reward']) * 100
        else:
            row['Gap_Trans_Base'] = 0
            
        data.append(row)
            
    if not data:
        print("No complete benchmark data found.")
        return

    df_report = pd.DataFrame(data)
    
    print("\nBenchmark Report Summary:")
    print(df_report[['Instance', 'Baseline_Max_Reward', 'Transformer_Max_Reward']].to_string(index=False))
    
    # Save to CSV
    df_report.to_csv('benchmark_summary.csv', index=False)
    print("\nSaved summary to benchmark_summary.csv")
    
    # Plotting Summary Bar Chart
    try:
        plt.figure(figsize=(12, 6))
        
        # Bar chart for Rewards
        x = range(len(df_report))
        width = 0.25
        
        plt.bar([i - width/2 for i in x], df_report['Baseline_Max_Reward'], width, label='Baseline')
        plt.bar([i + width/2 for i in x], df_report['Transformer_Max_Reward'], width, label='Transformer')
        
        plt.xlabel('Instance')
        plt.ylabel('Max Validation Reward')
        plt.title('Benchmark: Baseline vs Transformer (Max Reward)')
        plt.xticks(x, df_report['Instance'])
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('benchmark_plot.png')
        print("Saved plot to benchmark_plot.png")
    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")

if __name__ == "__main__":
    generate_report()

```

## `benchmark_report_all.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/benchmark_report_all.py`
- **Size**: 5031 bytes
- **Last modified**: 2025-11-27 19:58:32

```python
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = './results'

def get_latest_metrics(instance, model_name):
    # Path pattern: results/{instance}/outputs/model_{model_name}_uni_samp/training_history.csv
    path = f"{RESULTS_DIR}/{instance}/outputs/model_{model_name}_uni_samp/training_history.csv"
    
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        
        # Get max reward and final loss
        max_reward = df['avg_reward_val_uni_samp'].max()
        final_loss = df['tloss_train'].iloc[-1] if 'tloss_train' in df.columns else 0
        return {'max_reward': max_reward, 'final_loss': final_loss}
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def plot_training_curves(dataframes, instance):
    try:
        plt.figure(figsize=(12, 7))
        
        colors = {
            'Baseline': 'gray',
            'Transformer': 'blue',
            'GAT-LSTM': 'orange',
            'GAT-Trans': 'red'
        }

        for name, df in dataframes.items():
            if df is not None:
                plt.plot(df['epoch'], df['avg_reward_val_uni_samp'], 
                         label=name, linestyle='-', alpha=0.8, color=colors.get(name))
        
        plt.xlabel('Epoch')
        plt.ylabel('Avg Validation Reward')
        plt.title(f'Training Progress Comparison: {instance}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f'training_curve_all_{instance}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Saved comprehensive training curve to {filename}")
    except Exception as e:
        print(f"Error plotting curves for {instance}: {e}")

def generate_report():
    instances = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    instances = sorted(instances)
    
    summary_data = []
    
    # Định nghĩa các model cần so sánh và tên file suffix tương ứng
    # Key: Tên hiển thị, Value: tên suffix model (model_{suffix}_uni_samp)
    models_map = {
        'Baseline': 'baseline_bench',
        'Transformer': 'transformer_bench',
        'GAT-LSTM': 'gat_lstm_bench',
        'GAT-Trans': 'gat_transformer_bench'
    }

    print(f"{'Instance':<10} | {'Base':<8} | {'Trans':<8} | {'GAT-L':<8} | {'GAT-T':<8}")
    print("-" * 65)

    for inst in instances:
        row = {'Instance': inst}
        dfs = {}
        
        print_row = [f"{inst:<10}"]
        
        for display_name, model_suffix in models_map.items():
            # 1. Get Metrics
            metrics = get_latest_metrics(inst, model_suffix)
            metric_key = f"{display_name}_Max"
            
            if metrics:
                row[metric_key] = metrics['max_reward']
                print_row.append(f"{metrics['max_reward']:<8.2f}")
            else:
                row[metric_key] = 0
                print_row.append(f"{'N/A':<8}")
            
            # 2. Load DataFrame for plotting
            csv_path = f"{RESULTS_DIR}/{inst}/outputs/model_{model_suffix}_uni_samp/training_history.csv"
            if os.path.exists(csv_path):
                dfs[display_name] = pd.read_csv(csv_path)
            else:
                dfs[display_name] = None
        
        print(" | ".join(print_row))
        summary_data.append(row)
        
        # Vẽ biểu đồ nếu có ít nhất 1 model có dữ liệu
        if any(v is not None for v in dfs.values()):
            plot_training_curves(dfs, inst)

    if not summary_data:
        print("No benchmark data found.")
        return

    df_report = pd.DataFrame(summary_data)
    df_report.to_csv('benchmark_summary_all.csv', index=False)
    print("\nSaved full summary to benchmark_summary_all.csv")
    
    # Vẽ biểu đồ cột tổng hợp
    try:
        plt.figure(figsize=(14, 6))
        x = range(len(df_report))
        width = 0.15 # Độ rộng cột
        
        model_names = list(models_map.keys())
        # Tạo offset để vẽ nhiều cột cạnh nhau
        offsets = [i * width for i in range(len(model_names))]
        center_offset = (len(model_names) - 1) * width / 2
        
        for i, name in enumerate(model_names):
            col_name = f"{name}_Max"
            plt.bar([pos + offsets[i] - center_offset for pos in x], 
                    df_report[col_name], 
                    width, 
                    label=name)
        
        plt.xlabel('Instance')
        plt.ylabel('Max Validation Reward')
        plt.title('Comprehensive Benchmark: Max Reward by Model Architecture')
        plt.xticks(x, df_report['Instance'])
        plt.legend()
        plt.tight_layout()
        plt.savefig('benchmark_plot_all.png')
        print("Saved comparison plot to benchmark_plot_all.png")
    except Exception as e:
        print(f"Skipping summary plot: {e}")

if __name__ == "__main__":
    generate_report()
```

## `benchmark_runner.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/benchmark_runner.py`
- **Size**: 2964 bytes
- **Last modified**: 2025-11-21 23:38:37

```python
import os
import subprocess
import argparse
import time
import sys

# Configuration
PYTHON_EXEC = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'benchmark')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Representative subset for quick benchmarking
SUBSET_INSTANCES = ['c101', 'r101', 'rc101', 'pr01', 't101']

def get_all_instances():
    instances = []
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        return []
    
    for f in os.listdir(DATA_DIR):
        if f.endswith('.txt') and f not in ['TOPTW_format.txt', 'format_description_TTDP.txt']:
            instances.append(f.replace('.txt', ''))
    return sorted(instances)

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(e)

def benchmark_instance(instance, epochs=1000, batch_size=16):
    print(f"\n{'='*50}")
    print(f"Benchmarking Instance: {instance}")
    print(f"{'='*50}\n")

    # 1. Generate Validation Data
    print(f"--- Generating Validation Data for {instance} ---")
    cmd_gen = f"{PYTHON_EXEC} generate_instances_transformer.py --instance {instance} --ni 64 --device cpu"
    run_command(cmd_gen)

    # 2. Train Baseline
    print(f"\n--- Training Baseline for {instance} ---")
    cmd_base = f"{PYTHON_EXEC} train_optw_baseline.py --instance {instance} --model_name baseline_bench --nepocs {epochs} --batch_size {batch_size} --nprint 100 --nsave 2000 --device cpu"
    run_command(cmd_base)

    # 3. Train Transformer
    print(f"\n--- Training Transformer for {instance} ---")
    cmd_trans = f"{PYTHON_EXEC} train_optw_transformer.py --instance {instance} --model_name transformer_bench --nepocs {epochs} --batch_size {batch_size} --nprint 100 --nsave 2000 --device cpu"
    run_command(cmd_trans)

def main():
    parser = argparse.ArgumentParser(description='Benchmark Runner')
    parser.add_argument('--mode', choices=['subset', 'all'], default='subset', help='Run on subset or all instances')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs per training run')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    if args.mode == 'subset':
        instances_to_run = SUBSET_INSTANCES
    else:
        instances_to_run = get_all_instances()

    print(f"Starting benchmark on {len(instances_to_run)} instances...")
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    
    start_time = time.time()
    
    for inst in instances_to_run:
        benchmark_instance(inst, epochs=args.epochs, batch_size=args.batch_size)

    elapsed = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    main()

```

## `draft.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/draft.py`
- **Size**: 38 bytes
- **Last modified**: 2025-11-15 09:16:10

```python
import torch

print(torch.__version__)
```

## `generate_instances.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/generate_instances.py`
- **Size**: 2857 bytes
- **Last modified**: 2025-11-03 21:32:31

```python
#!/usr/bin/env python
# coding: utf-8

import math, operator
import numpy as np
import pandas as pd
import argparse

import os,time

from src.utils import get_instance_type, get_instance_df, get_distance_matrix
import src.config as cf
from src.sampling_norm_utils import sample_new_instance

import random
import torch

random.seed(2925)

#------------------------------------------------------------------------------------------

def setup_args_parser():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--instance', help='which instance to run')
    parser.add_argument('--ni', help='number of generated instances', default=64, type=int)

    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: \n \
                                uniformly sampled (uni_samp), \
                                score proportional to each point of interest\'s duration of visit (corr_samp)' ,
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')

    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    return parser


def parse_args_further(args):

    args.instance_type = get_instance_type(args.instance)
    args.output_directory = cf.GENERATED_INSTANCES_PATH+args.instance
    args.output_filename = 'inp_val_{sample_type}.pt'.format(sample_type=args.sample_type)
    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    return args



if __name__ == "__main__":

    # parse arguments and setup logger
    parser = setup_args_parser()
    args_temp = parser.parse_args()
    args = parse_args_further(args_temp)

    # for reproducibility"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_directory) if not os.path.exists(args.output_directory) else None

    df_inst = get_instance_df(args.instance, cf.BENCHMARK_INSTANCES_PATH, instance_type=args.instance_type)
    D = get_distance_matrix(df_inst, instance_type=args.instance_type)

    raw_data = df_inst[['x','y','duration','ti','tf','prof','Total Time']].values

    raw_data = torch.FloatTensor(raw_data).to(args.device)
    raw_distm =  torch.FloatTensor(D).to(args.device)

    inp_val = [sample_new_instance(raw_data, raw_distm, args) for x in range(args.ni)]

    output_path = '{output_dir}/{output_file}'.format(output_dir=args.output_directory,
                                                      output_file=args.output_filename)
    torch.save(inp_val, open(output_path, 'wb'))

```

## `generate_instances_transformer.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/generate_instances_transformer.py`
- **Size**: 2975 bytes
- **Last modified**: 2025-11-19 20:04:49

```python
#!/usr/bin/env python
# coding: utf-8

import math, operator
import numpy as np
import pandas as pd
import argparse

import os,time

# CHANGED: Import from src.utils_transformer
from src.utils_transformer import get_instance_type, get_instance_df, get_distance_matrix
import src.config as cf
from src.sampling_norm_utils import sample_new_instance

import random
import torch

random.seed(2925)

#------------------------------------------------------------------------------------------

def setup_args_parser():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--instance', help='which instance to run')
    parser.add_argument('--ni', help='number of generated instances', default=64, type=int)

    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: \n \
                                uniformly sampled (uni_samp), \
                                score proportional to each point of interest\'s duration of visit (corr_samp)' ,
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')

    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    return parser


def parse_args_further(args):

    args.instance_type = get_instance_type(args.instance)
    args.output_directory = cf.GENERATED_INSTANCES_PATH+args.instance
    args.output_filename = 'inp_val_{sample_type}.pt'.format(sample_type=args.sample_type)
    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    return args



if __name__ == "__main__":

    # parse arguments and setup logger
    parser = setup_args_parser()
    args_temp = parser.parse_args()
    args = parse_args_further(args_temp)

    # for reproducibility"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_directory) if not os.path.exists(args.output_directory) else None

    df_inst = get_instance_df(args.instance, cf.BENCHMARK_INSTANCES_PATH, instance_type=args.instance_type)
    D = get_distance_matrix(df_inst, instance_type=args.instance_type)

    raw_data = df_inst[['x','y','duration','ti','tf','prof','Total Time']].values

    raw_data = torch.FloatTensor(raw_data).to(args.device)
    raw_distm =  torch.FloatTensor(D).to(args.device)

    inp_val = [sample_new_instance(raw_data, raw_distm, args) for x in range(args.ni)]

    output_path = '{output_dir}/{output_file}'.format(output_dir=args.output_directory,
                                                      output_file=args.output_filename)
    torch.save(inp_val, open(output_path, 'wb'))

```

## `inference_optw_rl.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/inference_optw_rl.py`
- **Size**: 10560 bytes
- **Last modified**: 2025-11-26 21:26:35

```python
import os,time
import logging
import argparse
import json
import pandas as pd
import numpy as np

import torch
from torch import optim

import src.inference_utils as iu
import src.utils as u
import src.sampling_norm_utils as snu

import src.config as cf
import src.problem_config as pcf

from src.neural_net import RecPointerNetwork
from src.neural_net import RecPointerNetwork
try:
    from src.hybrid_neural_net import HybridPointerNetwork
except ImportError:
    HybridPointerNetwork = None
from src.neural_net_transformer import TransformerPointerNetwork
from src.neural_net_gat_lstm import GATLSTMPointerNetwork
from src.neural_net_gat_transformer import GATTransformerPointerNetwork

# for logging
N_DASHES = 40
SAVE_H_FILE = 'performance_scores.csv'




def setup_args_parser():
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--instance', help='which instance to run')
    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--use_checkpoint', help='use checkpoint (see https://pytorch.org/docs/stable/checkpoint.html)', action='store_true')
    parser.add_argument('--infe_type', help='which inference to run: \n \
                                greedy (gr), \
                                beam search (bs) or \
                                active search with beam search (as_bs)',
                                choices=['gr', 'bs', 'as_bs'],
                                default='bs')

    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: \n \
                                uniformly sampled (uni_samp), \
                                score proportional to each point of interest\'s duration of visit (corr_samp)' ,
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')

    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--saved_model_epoch', help='epoch number which the pre-trained model was saved', default=500000, type=int)
    parser.add_argument('--model_name', help='model name', default='default', type=str)
    parser.add_argument('--nprint', help='epoch frequency for printing and saving in training history generated/benchmark instance reward', default=1, type=int)
    parser.add_argument('--nepocs', help='number of epochs for active search training', default=128, type=int)
    parser.add_argument('--batch_size', help='traing batch size', default = 32, type=int)
    parser.add_argument('--max_beam_number', help='-max number of beams in beam search inference', default = 128, type=int)
    parser.add_argument('--max_grad_norm', help='maximum norm value for gradient value clipping', default=1, type=int)
    parser.add_argument('--lr', help='learning rate for active search training', default=1e-5, type=float)
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    parser.add_argument('--beta', help='entropy term coefficient', default=0.01, type=float)
    parser.add_argument('--generated', help='run on the generated instances of the validation set\
                                             instead of on the benchmark instance', action='store_true')

    return parser




def parse_args_further(args):

    LOAD_W_DIR_STRING = '{results_path}/{benchmark_instance}/model_w/model_{model_name}_{sample_type}'
    GENERATED_STRING = '{generated_path}/{benchmark_instance}'

    VAL_SET_PT_FILE = 'inp_val_{sample_type}.pt'

    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    args.instance_type = u.get_instance_type(args.instance)
    args.map_location =  {'cpu': args.device_name}

    args.val_dir = GENERATED_STRING.format(generated_path=cf.GENERATED_INSTANCES_PATH,
                                           benchmark_instance=args.instance)

    args.load_w_dir = LOAD_W_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)

    args.val_set_pt_file = VAL_SET_PT_FILE.format(sample_type=args.sample_type)

    return args


def load_saved_args(args):

    with open(args.load_w_dir+'/model_'+args.model_name+'_training_args.txt') as json_file:
        data = json.load(json_file)
        args.model_type = data.get('model_type', 'original')
        args.n_gat_layers = data.get('n_gat_layers', 0)

        args.n_layers = data['n_layers']
        args.n_heads = data['n_heads']
        args.ff_dim = data['ff_dim']
        args.nfeatures = data['nfeatures']
        args.ndfeatures = data['ndfeatures']
        args.rnn_hidden = data['rnn_hidden']

    return args


def log_args(args):
    logger.info(N_DASHES*'-')
    logger.info('Running test_optw_rl.py')
    logger.info(N_DASHES*'-')
    logger.info('model name:  %s' % args.model_name)
    logger.info(N_DASHES*'-')
    logger.info('device: %s' % args.device_name)
    logger.info('instance: %s' % args.instance)
    logger.info('instance type: %s' % args.instance_type)
    logger.info('infe_type: %s' % args.infe_type)
    logger.info('use_checkpoint: %s' % args.use_checkpoint)
    logger.info('sample type: %s' % args.sample_type)
    # logger.info('sample_prof: %s' % args.sample_prof) # Removed as it's not in args
    logger.info('debug mode: %s' % args.debug)
    logger.info('nprint: %s' % args.nprint)
    logger.info('nepocs: %s' % args.nepocs)
    logger.info('seed: %s' % args.seed)
    logger.info(N_DASHES*'-')
    # logger.info('max square length (Xmax): %s' % args.Xmax) # Removed
    logger.info('batch_size: %s' % args.batch_size)
    logger.info('max_grad_norm: %s' % args.max_grad_norm)
    logger.info('learning rate (lr): %s' % args.lr)
    logger.info('entropy term coefficient (beta): %s' % args.beta)
    logger.info('hidden size of RNN (hidden): %s' % args.rnn_hidden)
    logger.info('number of attention layers in the Encoder: %s' % args.n_layers)
    logger.info('number of features: %s' % args.nfeatures)
    logger.info('number of dynamic features: %s' % args.ndfeatures)
    logger.info(N_DASHES*'-')
    logger.info(args.instance)



if __name__ == "__main__":

    # ---------------------------------
    #  parse arguments and setup logger
    # ---------------------------------

    parser = setup_args_parser()
    args_temp = parser.parse_args()

    args = parse_args_further(args_temp)
    args = load_saved_args(args)

    logger = u.setup_logger(args.debug)
    if args.debug:
        log_args(args)


    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(args.seed)


    # ---------------------------------
    #  load data
    # ---------------------------------
    inp_real = u.get_real_data(args, phase='inference')
    raw_data, raw_distm = inp_real[0]

    start_time = raw_data[0, pcf.OPENING_TIME_WINDOW_IDX]

    # get Tmax and Smax
    norm_dic = {}
    Tmax, Smax = snu.instance_dependent_norm_const(raw_data)
    norm_dic = {'Tmax': Tmax, 'Smax': Smax}

    # ---------------------------------
    # load model
    # ---------------------------------

    logger.info('Loading model for instance {instance} ...'.format(instance=args.instance))
    performance_scores = []

    if args.model_type == 'hybrid':
        logger.info(f"Loading HYBRID model with {args.n_gat_layers} GAT layer(s).")
        pointer_net = HybridPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device).eval()
    elif args.model_type == 'transformer':
        logger.info(f"Loading TRANSFORMER model.")
        pointer_net = TransformerPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device).eval()
    elif args.model_type == 'gat_lstm':
        logger.info(f"Loading GAT-LSTM model with {args.n_gat_layers} GAT layer(s).")
        pointer_net = GATLSTMPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device).eval()
    elif args.model_type == 'gat_transformer':
        logger.info(f"Loading GAT-Transformer model with {args.n_gat_layers} GAT layer(s).")
        pointer_net = GATTransformerPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device).eval()
    else: 
        logger.info(f"Loading ORIGINAL model.") 
        pointer_net = RecPointerNetwork(args.nfeatures, args.ndfeatures,
                              args.rnn_hidden, args).to(args.device).eval()


    # ---------------------------------
    # inference
    # ---------------------------------

    if not args.generated:
        logger.info('Infering route for benchmark instance...')
        output =  iu.run_single(raw_data, norm_dic, start_time, raw_distm, args,
                                pointer_net, which_inf=args.infe_type)

    else:
        inp_val = u.get_val_data(args, phase='inference')
        logger.info('Infering routes for {num_inst} generated instances...' \
                    .format(num_inst=len(inp_val)))
        outputs =  iu.run_multiple(inp_val, norm_dic, args, pointer_net,
                                   which_inf=args.infe_type)

    # ---------------------------------
    # Log results
    # ---------------------------------
    if args.infe_type in ['gr', 'bs', 'as_bs']:

        logger.info(N_DASHES*'-')
        if not args.generated:
            logger.info('route: {route}'.format(route=output['route']))
            logger.info('total score: {total_score}'\
                        .format(total_score=int(output['score'])))
            inference_time_ms = int(1000*output['inf_time'])
            logger.info('inference time: {inference_time} ms'\
                        .format(inference_time=inference_time_ms))

        else:
            df_out = pd.DataFrame(outputs)
            average_total_score = round(df_out.score.mean(), 2)
            average_inf_time_ms = int(1000*df_out.inf_time.mean())
            logger.info('average total score: {average_total_score}' \
                        .format(average_total_score=average_total_score))
            logger.info('average inference time: {average_inference_time} ms' \
                        .format(average_inference_time=average_inf_time_ms))
        logger.info(N_DASHES*'-')

```

## `scan_code.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/scan_code.py`
- **Size**: 3961 bytes
- **Last modified**: 2025-11-26 23:55:40

```python
#!/usr/bin/env python3
"""
scan_code.py

Gom tất cả file .py trong một thư mục thành 1 file markdown.
"""
from pathlib import Path
import argparse
import datetime
import sys
import io
import os

DEFAULT_EXCLUDE_DIRS = {"__pycache__", ".git", "venv", "env", ".venv", "node_modules", "build", "dist"}

def find_py_files(root: Path, recursive: bool, exclude_dirs: set):
    if not root.exists():
        return []
    py_files = []
    if recursive:
        for p in root.rglob("*.py"):
            # skip if any parent directory in exclude_dirs
            if any(part in exclude_dirs for part in p.parts):
                continue
            # skip hidden files? (optional) keep them
            py_files.append(p)
    else:
        for p in root.glob("*.py"):
            if any(part in exclude_dirs for part in p.parts):
                continue
            py_files.append(p)
    # sort by relative path for stable order
    py_files.sort(key=lambda p: str(p.relative_to(root)))
    return py_files

def safe_read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:
            return f"# Could not read file {path}: {e}\n"

def render_file_md(path: Path, root: Path):
    rel = path.relative_to(root)
    stat = path.stat()
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" ", timespec="seconds")
    header = f"## `{rel}`\n\n"
    meta = f"- **Path**: `{path}`\n- **Size**: {stat.st_size} bytes\n- **Last modified**: {mtime}\n\n"
    content = safe_read_text(path)
    # ensure code fence doesn't break if triple backticks are inside file:
    # use ```​python and if file contains ```​ we can use ```​\u200b`` with zero-width space inside fence marker? Simpler: use triple backticks and escape by replacing any occurrence of ```​ with ```​\u200b
    content_sanitized = content.replace("```​", "```​​")  # adds ZWSP preventing fence break
    code_block = "```​python\n" + content_sanitized + "\n```​\n\n"
    return header + meta + code_block

def main():
    parser = argparse.ArgumentParser(description="Gom tất cả file .py thành 1 file markdown")
    parser.add_argument("-d", "--dir", default=".", help="Thư mục gốc để tìm file .py (default: current dir)")
    parser.add_argument("-o", "--output", default="all_python_code.md", help="File markdown đầu ra")
    parser.add_argument("-r", "--recursive",default=True, action="store_true", help="Đệ quy (tìm trong subfolders)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Tên các thư mục cần loại trừ (space-separated)")
    parser.add_argument("--prepend-toc", action="store_true", help="Thêm mục lục (TOC) phía trên")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    exclude_set = set(DEFAULT_EXCLUDE_DIRS) | set(args.exclude)

    py_files = find_py_files(root, args.recursive, exclude_set)
    if not py_files:
        print("Không tìm thấy file .py nào.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    parts = []
    title = f"# Gom mã nguồn Python từ `{root}`\n\nGenerated: {datetime.datetime.now().isoformat(sep=' ', timespec='seconds')}\n\n"
    parts.append(title)

    if args.prepend_toc:
        parts.append("## Mục lục\n\n")
        for p in py_files:
            rel = p.relative_to(root)
            anchor = str(rel).replace("/", "-").replace(" ", "-")
            parts.append(f"- [{rel}](#{anchor})\n")
        parts.append("\n")

    for p in py_files:
        parts.append(render_file_md(p, root))

    # write atomically
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(parts)
    tmp.replace(out_path)
    print(f"Wrote {len(py_files)} files to {out_path}")

if __name__ == "__main__":
    main()

```

## `src/__init__.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/__init__.py`
- **Size**: 0 bytes
- **Last modified**: 2025-11-03 21:32:31

```python

```

## `src/config.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/config.py`
- **Size**: 120 bytes
- **Last modified**: 2025-11-03 21:32:31

```python
BENCHMARK_INSTANCES_PATH = './data/benchmark/'
GENERATED_INSTANCES_PATH = './data/generated/'
RESULTS_PATH = './results'
```

## `src/features_utils.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/features_utils.py`
- **Size**: 1756 bytes
- **Last modified**: 2025-11-03 21:32:31

```python
import torch
import src.problem_config as pcf



class DynamicFeatures():

    def __init__(self, args):
        super(DynamicFeatures, self).__init__()

        self.arrival_time_idx = pcf.ARRIVAL_TIME_IDX
        self.opening_time_window_idx = pcf.OPENING_TIME_WINDOW_IDX
        self.closing_time_window_idx = pcf.CLOSING_TIME_WINDOW_IDX
        self.device = args.device

    def make_dynamic_feat(self, data, current_time, current_poi_idx, dist_mat, batch_idx):

        num_dyn_feat = 8
        _ , sequence_size, input_size  = data.size()
        batch_size = batch_idx.shape[0]

        dyn_feat = torch.ones(batch_size, sequence_size, num_dyn_feat).to(self.device)

        tour_start_time = data[0, 0, self.opening_time_window_idx]
        max_tour_duration = data[0, 0, self.arrival_time_idx] - tour_start_time
        arrive_j_times = current_time + dist_mat[current_poi_idx]

        dyn_feat[:, :, 0] = (data[batch_idx, :, self.opening_time_window_idx] - current_time) / max_tour_duration
        dyn_feat[:, :, 1] = (data[batch_idx, :, self.closing_time_window_idx] - current_time) / max_tour_duration
        dyn_feat[:, :, 2] = (data[batch_idx, :, self.arrival_time_idx] - current_time) / max_tour_duration
        dyn_feat[:, :, 3] = (current_time - tour_start_time) / max_tour_duration


        dyn_feat[:, :, 4] = (arrive_j_times - tour_start_time) / max_tour_duration
        dyn_feat[:, :, 5] = (data[batch_idx, :, self.opening_time_window_idx] - arrive_j_times) / max_tour_duration
        dyn_feat[:, :, 6] = (data[batch_idx, :, self.closing_time_window_idx] - arrive_j_times) / max_tour_duration
        dyn_feat[:, :, 7] = (data[batch_idx, :, self.arrival_time_idx] - arrive_j_times) / max_tour_duration

        return dyn_feat

```

## `src/hybrid_neural_net.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/hybrid_neural_net.py`
- **Size**: 3885 bytes
- **Last modified**: 2025-11-15 09:13:29

```python
# src/hybrid_neural_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import các thành phần có thể tái sử dụng từ mô hình gốc
from src.neural_net import Decoder, EncoderLayer, RecPointerNetwork

# Import thư viện GNN
from torch_geometric.nn import GATConv

def dense_to_sparse_edge_index(adj):
    """Chuyển đổi ma trận kề dày [N, N] sang định dạng edge_index [2, num_edges] của PyG."""
    # Đảm bảo ma trận không có gradient để tránh lỗi
    edge_index = torch.nonzero(adj.detach(), as_tuple=False).t()
    return edge_index

class HybridEncoder(nn.Module):
    """
    Bộ mã hóa lai kết hợp GAT (xử lý cục bộ) và Transformer (xử lý toàn cục).
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_size, args):
        super(HybridEncoder, self).__init__()

        n_heads = args.n_heads
        d_ff = args.ff_dim
        n_layers = args.n_layers
        n_gat_layers = args.n_gat_layers

        # Các lớp chiếu feature ban đầu, giữ nguyên
        self.L1 = nn.Linear(features_dim, hidden_size // 2)
        self.L2 = nn.Linear(dfeatures_dim, hidden_size // 2)

        # --- TÍCH HỢP GAT ---
        self.n_gat_layers = n_gat_layers
        if self.n_gat_layers > 0:
            self.gat_layers = nn.ModuleList([
                GATConv(in_channels=hidden_size, out_channels=hidden_size, heads=n_heads, concat=False, dropout=0.1)
                for _ in range(n_gat_layers)
            ])
            self.gat_layer_norm = nn.LayerNorm(hidden_size)

        # --- CÁC LỚP TRANSFORMER ---
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, emb_inp, rec_inp, mask, dummy_arg):
        batch_size = emb_inp.size(0)

        # --- Giai đoạn 1: Xử lý cục bộ bằng GAT ---
        if self.n_gat_layers > 0:
            gat_outputs = []
            for i in range(batch_size):
                x_i = emb_inp[i]
                adj_i = mask[i]
                edge_index_i = dense_to_sparse_edge_index(adj_i)
                
                # Truyền qua các lớp GAT
                for gat_layer in self.gat_layers:
                    # GATConv trả về embedding mới, không cần F.relu ngay lập tức
                    # F.relu thường được dùng giữa các layer
                    x_i = gat_layer(x_i, edge_index_i)

                gat_outputs.append(x_i)
            
            gat_processed_emb = torch.stack(gat_outputs, dim=0)
            emb_inp = self.gat_layer_norm(emb_inp + F.relu(gat_processed_emb))

        # --- Giai đoạn 2: Xử lý toàn cục bằng Transformer ---
        for layer in self.layers:
            emb_inp, _ = layer(emb_inp, rec_inp, mask)

        return emb_inp

class HybridPointerNetwork(RecPointerNetwork):
    """
    Lớp Pointer Network mới sử dụng HybridEncoder.
    Chúng ta kế thừa từ RecPointerNetwork để tái sử dụng các phương thức khác.
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_dim, args):
        # Gọi __init__ của lớp cha (nn.Module)
        super(RecPointerNetwork, self).__init__()

        self.features_dim = features_dim
        self.dfeatures_dim = dfeatures_dim
        self.use_checkpoint = args.use_checkpoint
        self.hidden_dim = hidden_dim
        self.decoder = Decoder(hidden_dim) # Tái sử dụng Decoder từ file gốc
        
        # SỬ DỤNG BỘ MÃ HÓA LAI MỚI
        self.encoder = HybridEncoder(features_dim, dfeatures_dim, hidden_dim, args)
        
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._initialize_parameters()
        # Các phương thức còn lại (forward, _one_step, etc.) được kế thừa từ RecPointerNetwork
```

## `src/ils.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/ils.py`
- **Size**: 14248 bytes
- **Last modified**: 2025-11-24 22:29:02

```python
import numpy as np
import pandas as pd
import time
import random
import copy
import src.config as cf

# --- Data Loading Utils (Copied from src/utils_transformer.py to avoid torch dependency) ---

def read_instance_data(instance_name, path):
    """reads instance data"""
    PATH_TO_BENCHMARK_INSTANCES = path
    benchmark_file = '{path_to_benchmark_instances}/{instance}.txt'.format(
        path_to_benchmark_instances=PATH_TO_BENCHMARK_INSTANCES,
        instance=instance_name)
    dfile = open(benchmark_file)
    data = [[float(x) for x in line.split()] for line in dfile]
    dfile.close()
    return data

def eliminate_extra_cordeau_columns(instance_data):
    DATA_INIT_ROW = 2
    N_RELEVANT_FIRST_COLUMNS = 8
    N_RELEVANT_LAST_COLUMNS = 2
    return [s[:N_RELEVANT_FIRST_COLUMNS]+s[-N_RELEVANT_LAST_COLUMNS:] for s in instance_data[DATA_INIT_ROW :]]

def parse_instance_data_Gavalas(instance_data):
    N_DAYS_INDEX = 1
    START_DAY_INDEX = 2
    M = instance_data[0][N_DAYS_INDEX]
    SD =  int(instance_data[0][START_DAY_INDEX])
    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 't',
                           'open_0', 'close_0', 'open_1', 'close_1',
                           'open_2', 'close_2', 'open_3', 'close_3',
                           'open_4', 'close_4', 'open_5', 'close_5',
                           'open_6', 'close_6', 'b']
    df = pd.DataFrame(instance_data[2:], columns=COLUMN_NAMES_ABBREV)
    df_ = df[['i', 'x', 'y', 'd', 'S', 't']+[c for c in df.columns if c[-1]==str(SD)]]
    columns = ['i', 'x', 'y', 'd', 'S', 't', OPENING_TIME_WINDOW_ABBREV_KEY, CLOSING_TIME_WINDOW_ABBREV_KEY]
    df_.columns=columns
    aux = pd.DataFrame([instance_data[1]], columns = ['i', 'x', 'y', 'd', 'S', 'O', 'C'])
    df = pd.concat([aux, df_], axis=0, sort=True).reset_index(drop=True)
    df[TOTAL_TIME_KEY] = 0
    df[TOTAL_TIME_KEY] = df.loc[0][CLOSING_TIME_WINDOW_ABBREV_KEY]
    return df

def parse_instance_data(instance_data):
    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 'f', 'a', 'list', 'O', 'C']
    instance_data_clean = eliminate_extra_cordeau_columns(instance_data)
    df = pd.DataFrame(instance_data_clean, columns=COLUMN_NAMES_ABBREV)
    df[TOTAL_TIME_KEY] = 0
    df[TOTAL_TIME_KEY] = df.loc[0][CLOSING_TIME_WINDOW_ABBREV_KEY]
    return df

def get_instance_type(instance_name):
    if instance_name[:2]=='pr': return 'Cordeau'
    elif instance_name[0] in ['r', 'c']: return 'Solomon'
    elif instance_name[0] in ['t']: return 'Gavalas'
    raise Exception('weird instance')

def get_instance_df(instance_name, path, instance_type):
    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'
    COLS_OF_INT = ['i', 'x', 'y', 'd', OPENING_TIME_WINDOW_ABBREV_KEY,
                   CLOSING_TIME_WINDOW_ABBREV_KEY, 'S', TOTAL_TIME_KEY]
    COLS_OF_INT_NEW_NAMES = ['i', 'x', 'y', 'duration', 'ti', 'tf', 'prof', TOTAL_TIME_KEY]
    standard2newnames_dict =  dict(((c, ca) for c, ca in zip(COLS_OF_INT, COLS_OF_INT_NEW_NAMES)))
    
    instance_data = read_instance_data(instance_name, path)
    
    if instance_type=='Gavalas':
        df = parse_instance_data_Gavalas(instance_data)
    else:
        df = parse_instance_data(instance_data)
        
    COLS_OF_INT_NEW_NAMES = [standard2newnames_dict[s] for s in COLS_OF_INT]
    df_ = df[COLS_OF_INT].copy()
    df_.columns = COLS_OF_INT_NEW_NAMES
    df_['inst_name'] = instance_name
    df_['real_or_val'] = 'real'
    df_ = pd.concat([df_, df_.loc[[0]]], ignore_index=True)
    return df_

def get_distance_matrix(instance_df, instance_type):
    if instance_type in ['Solomon']: n_digits = 10.0
    elif instance_type in ['Cordeau', 'Gavalas']: n_digits = 100.0
    n = instance_df.shape[0]
    distm = np.zeros((n,n))
    x = instance_df.x.values
    y = instance_df.y.values
    for i in range(0, n-1):
        for j in range(i+1, n):
            distm[i,j] = np.floor(n_digits*(np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)))/n_digits
            distm[j,i] = distm[i,j]
    return distm

# --- End of Data Loading Utils ---

class OPTW_ILS:
    def __init__(self, instance_name, max_iter=1000, time_limit=None):
        self.instance_name = instance_name
        self.max_iter = max_iter
        self.time_limit = time_limit
        
        # Load Data
        self.instance_type = get_instance_type(instance_name)
        self.df = get_instance_df(instance_name, cf.BENCHMARK_INSTANCES_PATH, self.instance_type)
        self.dist_mat = get_distance_matrix(self.df, self.instance_type)
        
        # Extract arrays for faster access
        self.n_nodes = len(self.df)
        self.profits = self.df['prof'].values
        self.durations = self.df['duration'].values
        self.opens = self.df['ti'].values
        self.closes = self.df['tf'].values
        self.max_time = self.df['Total Time'].iloc[0]
        
        # Node 0 is start/end
        self.start_node = 0
        
    def check_feasibility(self, route):
        """
        Checks if a route is feasible (time windows and max time).
        Returns (is_feasible, total_time, total_profit)
        """
        current_time = 0.0
        total_profit = 0.0
        
        prev_node = self.start_node
        
        # Route should not include start/end node explicitly in the middle, 
        # but we assume route is list of visited customer nodes.
        # We assume start -> route -> start
        
        # Start at depot
        current_time = self.opens[self.start_node] # Usually 0
        
        for node in route:
            # Travel time
            dist = self.dist_mat[prev_node, node]
            arrival = current_time + dist
            
            # Wait if early
            start_service = max(arrival, self.opens[node])
            
            # Check late
            if start_service > self.closes[node]:
                return False, float('inf'), 0
            
            # Service
            current_time = start_service + self.durations[node]
            total_profit += self.profits[node]
            prev_node = node
            
        # Return to depot
        dist_to_depot = self.dist_mat[prev_node, self.start_node]
        arrival_depot = current_time + dist_to_depot
        
        if arrival_depot > self.max_time:
             return False, float('inf'), 0
             
        return True, arrival_depot, total_profit

    def calculate_profit(self, route):
        return np.sum(self.profits[route])

    def insert_greedy(self, route):
        """
        Tries to insert unvisited nodes into the route greedily.
        Best insertion position based on min time increase or max profit/time ratio.
        """
        unvisited = [i for i in range(1, self.n_nodes) if i not in route]
        
        while unvisited:
            best_node = -1
            best_pos = -1
            best_ratio = -1.0
            
            for node in unvisited:
                # Try inserting node at every position
                for i in range(len(route) + 1):
                    new_route = route[:i] + [node] + route[i:]
                    feasible, time_cost, _ = self.check_feasibility(new_route)
                    
                    if feasible:
                        # Heuristic: Maximize Profit / Time_Increase
                        # But simpler: Just maximize profit, tie-break with min time
                        # For OPTW, profit is paramount.
                        
                        # Let's calculate marginal cost
                        # We need the time of the route BEFORE insertion to calculate increase?
                        # Or just pick the one that leaves most remaining time?
                        # Actually, we want to pick the move that gives highest profit.
                        # If profits are equal, pick min time.
                        
                        # Since we iterate all nodes, we can find max profit feasible node.
                        # If multiple positions feasible for same node, pick min total time.
                        
                        ratio = self.profits[node] # / time_cost (maybe?)
                        # Let's stick to: Maximize Profit, then Minimize Total Time
                        
                        # To make it comparable across nodes with different profits:
                        # Score = Profit - alpha * Total_Time (alpha small)
                        score = self.profits[node] - 0.0001 * time_cost
                        
                        if score > best_ratio:
                            best_ratio = score
                            best_node = node
                            best_pos = i
            
            if best_node != -1:
                route.insert(best_pos, best_node)
                unvisited.remove(best_node)
            else:
                break # No more nodes can be inserted
                
        return route

    def local_search(self, route):
        """
        Applies 2-opt, Swap, and Relocate to reduce time, potentially enabling more insertions.
        """
        improved = True
        while improved:
            improved = False
            
            # 1. Relocate (Shift)
            # Move node i to position j
            for i in range(len(route)):
                for j in range(len(route)):
                    if i == j: continue
                    
                    node = route[i]
                    new_route = route[:i] + route[i+1:] # Remove
                    new_route.insert(j, node) # Insert
                    
                    feasible, new_time, _ = self.check_feasibility(new_route)
                    _, old_time, _ = self.check_feasibility(route)
                    
                    if feasible and new_time < old_time - 1e-4: # Tolerance
                        route = new_route
                        improved = True
                        break
                if improved: break
            if improved: continue

            # 2. Swap
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    new_route = route[:]
                    new_route[i], new_route[j] = new_route[j], new_route[i]
                    
                    feasible, new_time, _ = self.check_feasibility(new_route)
                    _, old_time, _ = self.check_feasibility(route)
                    
                    if feasible and new_time < old_time - 1e-4:
                        route = new_route
                        improved = True
                        break
                if improved: break
            if improved: continue
            
            # 3. 2-opt
            for i in range(len(route)):
                for j in range(i + 2, len(route) + 1):
                    # Reverse segment [i:j]
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    
                    feasible, new_time, _ = self.check_feasibility(new_route)
                    _, old_time, _ = self.check_feasibility(route)
                    
                    if feasible and new_time < old_time - 1e-4:
                        route = new_route
                        improved = True
                        break
                if improved: break
                
        return route

    def perturb(self, route):
        """
        Removes k random nodes.
        """
        if not route: return route
        
        k = min(len(route), random.randint(1, min(5, len(route))))
        for _ in range(k):
            if not route: break
            idx = random.randint(0, len(route)-1)
            route.pop(idx)
            
        return route

    def solve(self):
        start_time = time.time()
        
        # Initial Solution
        current_route = []
        current_route = self.insert_greedy(current_route)
        current_route = self.local_search(current_route)
        # Try inserting again after LS reduced time
        current_route = self.insert_greedy(current_route)
        
        best_route = list(current_route)
        best_profit = self.calculate_profit(best_route)
        
        iter_count = 0
        while iter_count < self.max_iter:
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
                
            # Perturbation
            new_route = list(current_route)
            new_route = self.perturb(new_route)
            
            # Local Search + Greedy Insertion
            new_route = self.insert_greedy(new_route)
            new_route = self.local_search(new_route)
            new_route = self.insert_greedy(new_route)
            
            new_profit = self.calculate_profit(new_route)
            
            # Acceptance Criterion (Simple Improvement)
            if new_profit > best_profit:
                best_profit = new_profit
                best_route = list(new_route)
                current_route = list(new_route)
            elif new_profit == best_profit:
                 # If equal profit, maybe accept if time is lower?
                 # For now, just keep best.
                 # To escape local optima, we might accept worse solutions (Simulated Annealing),
                 # but standard ILS often just accepts improvements or uses a restart.
                 # Let's accept if profit is same but different structure (random walk on plateau)
                 if random.random() < 0.1:
                     current_route = list(new_route)
            else:
                # Restart from best sometimes?
                if random.random() < 0.05:
                     current_route = list(best_route)
            
            iter_count += 1
            
        return best_profit, best_route

if __name__ == "__main__":
    # Test
    ils = OPTW_ILS('c101', max_iter=100)
    profit, route = ils.solve()
    print(f"Instance: c101, Profit: {profit}, Route: {route}")

```

## `src/inference_utils.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/inference_utils.py`
- **Size**: 4959 bytes
- **Last modified**: 2025-11-03 21:32:31

```python
from tqdm import tnrange, tqdm
import time


import torch
from torch import optim
import src.train_utils as tu
import src.sampling_norm_utils as snu
from src.solution_construction import RunEpisode, BeamSearch

def gr_inference(inst_data, norm_dic, start_time, dist_mat, args, run_episode):

    data_scaled = snu.data_scaler(inst_data, norm_dic)
    binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

    with torch.no_grad():
        seq, _ , _, _ = run_episode(binst_data, bdata_scaled, start_time, dist_mat, 'greedy')

    rewards = tu.reward_fn(inst_data, seq, args.device)
    maxrew, idx_max = torch.max(rewards, 0)
    score = maxrew.item()

    route =  [act.item() for act in seq]
    route[-1] = 0
    return route, score


def bs_inference(inst_data, norm_dic, start_time, dist_mat, args, run_episode):

    data_scaled = snu.data_scaler(inst_data, norm_dic)
    binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

    nb = args.max_beam_number
    with torch.no_grad():
        seq, _ = run_episode(binst_data, bdata_scaled, start_time, dist_mat, 'greedy', nb)

    seq_list = [ seq[:,k] for k in range(seq.shape[1])]
    rewards = tu.reward_fn(inst_data, seq_list, args.device)
    maxrew, idx_max = torch.max(rewards, 0)
    score = maxrew.item()

    route =  [0] + [val.item() for val in seq[idx_max] if val.item() != 0]
    route[-1] = 0
    return route, score


def as_bs_inference(inp_data, norm_dic, args, run_episode, run_episode_bs):

    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=args.lr)

    inst_data, start_time, dist_mat = inp_data

    data_scaled = snu.data_scaler(inst_data, norm_dic)

    for epoch in tqdm(range(args.nepocs)):

        active_search_train_model(inst_data, data_scaled, start_time, dist_mat, run_episode, model_opt, args)

    # .. to load your previously training model:
    run_episode_bs.neuralnet.load_state_dict(run_episode.neuralnet.state_dict())
    binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

    with torch.no_grad():  
        seq, _ = run_episode_bs(binst_data, bdata_scaled, start_time, dist_mat, 'greedy', args.max_beam_number)

    seq_list = [ seq[:,k] for k in range(seq.shape[1])]
    rewards = tu.reward_fn(inst_data, seq_list, args.device)

    maxreward, idx_max = torch.max(rewards, 0)

    score = maxreward.item()

    route =  [0]+[val.item() for val in seq[idx_max] if val.item() != 0]
    route[-1] = 0

    return route, score

def run_single(inst_data, norm_dic, start_time, dist_mat, args, model,
               which_inf=None):


    saved_model_path = args.load_w_dir +'/model_' + str(args.saved_model_epoch) + '.pkl'
    model._load_model_weights(saved_model_path, args.device)


    tic = time.time()
    if which_inf=='bs':
        run_episode_inf = BeamSearch(model, args).eval()
        route, score = bs_inference(inst_data, norm_dic, start_time, dist_mat,
                                    args, run_episode_inf)

    elif which_inf=='gr':
        run_episode_inf = RunEpisode(model, args).eval()
        route, score = gr_inference(inst_data, norm_dic, start_time, dist_mat,
                                    args, run_episode_inf)

    elif which_inf=='as_bs':

        saved_model_path = args.load_w_dir +'/model_' + str(args.saved_model_epoch) + '.pkl'
        model._load_model_weights(saved_model_path, args.device)
        run_episode_train = RunEpisode(model, args)

        run_episode_inf = BeamSearch(model, args).eval()
        inp_data = (inst_data, start_time, dist_mat)
        route, score = as_bs_inference(inp_data, norm_dic, args,
                                       run_episode_train, run_episode_inf)
    toc = time.time()

    output = dict([('score', score), ('route', route), ('inf_time', toc-tic)])

    return output

def run_multiple(inp_val, norm_dic, args, model, which_inf=None):

    outputs = list()
    for k, (inst_data, start_time, dist_mat) in enumerate(tqdm(inp_val)):
        output = run_single(inst_data, norm_dic, start_time, dist_mat, args,
                               model, which_inf=which_inf)
        outputs.append(output)

    return outputs

def active_search_train_model(inst_data, data_scaled, inp_t_init_val, dist_mat, run_episode, model_opt, args):

    run_episode.train()

    binst_data, bdata_scaled = tu.samples2batch(inst_data, data_scaled, args.batch_size)

    actions, log_prob, entropy, step_mask = run_episode(binst_data, bdata_scaled, inp_t_init_val, dist_mat, 'stochastic')

    rewards = tu.reward_fn(inst_data, actions, args.device)

    av_rew = rewards.mean()

    advantage = (rewards - av_rew)

    res = advantage.unsqueeze(1)*log_prob + args.beta*entropy

    loss = -res[step_mask].sum()/args.batch_size

    model_opt.zero_grad()
    loss.backward(retain_graph=False)
    torch.nn.utils.clip_grad_norm_(run_episode.neuralnet.parameters(), args.max_grad_norm)
    model_opt.step()

```

## `src/neural_net.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/neural_net.py`
- **Size**: 9268 bytes
- **Last modified**: 2025-11-19 20:01:42

```python
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import math
import numpy as np

# ------------------------------------------------------------------------------
# Transformer model from: https://github.com/JayParks/transformer
# and https://github.com/jadore801120/attention-is-all-you-need-pytorch


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()                                                                                                                                                                                                                                                                                                        
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
        #    assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask==0, -1e32)

        attn = self.softmax(attn )
        outputs = torch.bmm(attn, v) # outputs: [b_size x len_q x d_v]
        return outputs, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(_MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_v))

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, q, k, v, attn_mask=None, is_adj=True):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)  # [b_size * n_heads x len_v x d_v]

        if attn_mask is not None:
            if is_adj:
                outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_heads, 1, 1))
            else:
                outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.unsqueeze(1).repeat(n_heads, 1, 1))
        else:
            outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=None)

        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.attention = _MultiHeadAttention(d_model, n_heads)
        self.proj = nn.Linear(n_heads * self.d_k, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask = None, is_adj = True):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask, is_adj=is_adj)
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1)
        # project back to residual size, result_size = [b_size x len_q x d_model]
        outputs = self.proj(outputs)

        return self.layer_norm(residual + outputs), attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x # inputs: [b_size x len_q x d_model]
        outputs = self.w_2(F.relu(self.w_1(x)))
        return self.layer_norm(residual + outputs)


#----------- Pointer models common blocks ---------------------

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()

        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Parameter(torch.zeros((hidden_size, 1), requires_grad=True))

        self.first_h_0 = nn.Parameter(torch.FloatTensor(1, hidden_size), requires_grad=True)
        self.first_h_0.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

        self.c0 = nn.Parameter(torch.FloatTensor( 1, hidden_size),requires_grad=True)
        self.c0.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

        self.hidden_0 = (self.first_h_0, self.c0)

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)


    def forward(self, input, hidden, enc_outputs, mask):
        hidden = self.lstm(input, hidden)
        w1e = self.W1(enc_outputs)
        w2h = self.W2(hidden[0]).unsqueeze(1)
        u = torch.tanh(w1e + w2h)
        a = u.matmul(self.V)
        a = 10*torch.tanh(a).squeeze(2)

        policy = F.softmax(a + mask.float().log(), dim=1)

        return policy, hidden


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, enc_inp, rec_enc_inp, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inp, rec_enc_inp, enc_inp, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, features_dim, dfeatures_dim, hidden_size, args):
        super(Encoder, self).__init__()

        n_heads = args.n_heads # number of heads
        d_ff = args.ff_dim # feed_forward_hidden
        n_layers = args.n_layers # number of Layers

        self.L1 = nn.Linear(features_dim, hidden_size//2) # for static features
        self.L2 = nn.Linear(dfeatures_dim, hidden_size//2) # for dynamic features

        self.layers = nn.ModuleList([EncoderLayer(hidden_size, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, emb_inp, rec_inp, mask, dummy_arg):
        for layer in self.layers:
            emb_inp, _ = layer(emb_inp, rec_inp, mask)

        return emb_inp


class RecPointerNetwork(nn.Module):

    def __init__(self, features_dim, dfeatures_dim, hidden_dim, args):
        super(RecPointerNetwork, self).__init__()

        self.features_dim = features_dim
        self.dfeatures_dim = dfeatures_dim
        self.use_checkpoint = args.use_checkpoint
        self.hidden_dim = hidden_dim
        self.decoder = Decoder(hidden_dim)
        self.encoder = Encoder(features_dim, dfeatures_dim, hidden_dim, args)
        # see https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._initialize_parameters()

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def _load_model_weights(self, path_string, device):
        self.load_state_dict(torch.load(path_string, map_location=device))


    def forward(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step=False):
        policy, dec_hidden, enc_outputs = self._one_step(enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step)
        return policy, dec_hidden, enc_outputs

    def _one_step(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step):
        if self.use_checkpoint:
            enc_outputs = checkpoint(self.encoder, enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)
        else:
            enc_outputs = self.encoder(enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)

        if first_step:
            return  None, None, enc_outputs
        else:
            policy, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs, mask)
            return policy, dec_hidden, enc_outputs

    def sta_emb(self, sta_inp):
        return torch.tanh(self.encoder.L1(sta_inp))

    def dyn_emb(self, dyn_inp):
        return torch.tanh(self.encoder.L2(dyn_inp))

```

## `src/neural_net_gat_lstm.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/neural_net_gat_lstm.py`
- **Size**: 5005 bytes
- **Last modified**: 2025-11-27 00:01:05

```python
# src/neural_net_gat_lstm.py
# GAT-LSTM Hybrid Model: GAT Encoder + LSTM Decoder (from baseline)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# Import baseline components
from src.neural_net import Decoder, EncoderLayer

# Import GAT from PyTorch Geometric
from torch_geometric.nn import GATConv

def dense_to_sparse_edge_index(adj):
    """Convert dense adjacency matrix [N, N] to sparse edge_index [2, num_edges] format for PyG."""
    edge_index = torch.nonzero(adj.detach(), as_tuple=False).t()
    return edge_index

class GATEncoder(nn.Module):
    """
    GAT-based Encoder combining GAT layers (local) with Transformer attention layers (global).
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_size, args):
        super(GATEncoder, self).__init__()

        n_heads = args.n_heads
        d_ff = args.ff_dim
        n_layers = args.n_layers
        n_gat_layers = args.n_gat_layers

        # Feature projection layers (same as baseline)
        self.L1 = nn.Linear(features_dim, hidden_size // 2)
        self.L2 = nn.Linear(dfeatures_dim, hidden_size // 2)

        # GAT layers for local graph processing
        self.n_gat_layers = n_gat_layers
        if self.n_gat_layers > 0:
            self.gat_layers = nn.ModuleList([
                GATConv(in_channels=hidden_size, out_channels=hidden_size, 
                       heads=n_heads, concat=False, dropout=0.1)
                for _ in range(n_gat_layers)
            ])
            self.gat_layer_norm = nn.LayerNorm(hidden_size)

        # Transformer attention layers for global context
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, d_ff, n_heads) 
            for _ in range(n_layers)
        ])

    def forward(self, emb_inp, rec_inp, mask, dummy_arg):
        batch_size = emb_inp.size(0)

        # Stage 1: Local graph processing with GAT
        if self.n_gat_layers > 0:
            gat_outputs = []
            for i in range(batch_size):
                x_i = emb_inp[i]
                adj_i = mask[i]
                edge_index_i = dense_to_sparse_edge_index(adj_i)
                
                # Pass through GAT layers
                for gat_layer in self.gat_layers:
                    identity = x_i
                    out = gat_layer(x_i, edge_index_i)
                    out = F.relu(out)
                    x_i = identity + out

                gat_outputs.append(x_i)
            
            gat_processed_emb = torch.stack(gat_outputs, dim=0)
            # Residual connection + layer norm
            emb_inp = self.gat_layer_norm(emb_inp + F.relu(gat_processed_emb))

        # Stage 2: Global context with Transformer attention
        for layer in self.layers:
            emb_inp, _ = layer(emb_inp, rec_inp, mask)

        return emb_inp


class GATLSTMPointerNetwork(nn.Module):
    """
    Hybrid Pointer Network with GAT Encoder and LSTM Decoder (baseline).
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_dim, args):
        super(GATLSTMPointerNetwork, self).__init__()

        self.features_dim = features_dim
        self.dfeatures_dim = dfeatures_dim
        self.use_checkpoint = args.use_checkpoint
        self.hidden_dim = hidden_dim
        
        # LSTM Decoder from baseline
        self.decoder = Decoder(hidden_dim)
        
        # GAT-based Encoder
        self.encoder = GATEncoder(features_dim, dfeatures_dim, hidden_dim, args)
        
        # Dummy tensor for checkpointing
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._initialize_parameters()

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def _load_model_weights(self, path_string, device):
        self.load_state_dict(torch.load(path_string, map_location=device))

    def forward(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step=False):
        policy, dec_hidden, enc_outputs = self._one_step(enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step)
        return policy, dec_hidden, enc_outputs

    def _one_step(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step):
        if self.use_checkpoint:
            enc_outputs = checkpoint.checkpoint(self.encoder, enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)
        else:
            enc_outputs = self.encoder(enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)

        if first_step:
            return None, None, enc_outputs
        else:
            policy, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs, mask)
            return policy, dec_hidden, enc_outputs

    def sta_emb(self, sta_inp):
        return torch.tanh(self.encoder.L1(sta_inp))

    def dyn_emb(self, dyn_inp):
        return torch.tanh(self.encoder.L2(dyn_inp))

```

## `src/neural_net_gat_transformer.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/neural_net_gat_transformer.py`
- **Size**: 5376 bytes
- **Last modified**: 2025-11-26 23:59:34

```python
# src/neural_net_gat_transformer.py
# GAT-Transformer Hybrid Model: GAT Encoder + Transformer Decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

# Import baseline components
from src.neural_net import EncoderLayer

# Import Transformer decoder
from src.neural_net_transformer import TransformerDecoder

# Import GAT from PyTorch Geometric
from torch_geometric.nn import GATConv

def dense_to_sparse_edge_index(adj):
    """Convert dense adjacency matrix [N, N] to sparse edge_index [2, num_edges] format for PyG."""
    edge_index = torch.nonzero(adj.detach(), as_tuple=False).t()
    return edge_index

class GATEncoder(nn.Module):
    """
    GAT-based Encoder combining GAT layers (local) with Transformer attention layers (global).
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_size, args):
        super(GATEncoder, self).__init__()

        n_heads = args.n_heads
        d_ff = args.ff_dim
        n_layers = args.n_layers
        n_gat_layers = args.n_gat_layers

        # Feature projection layers (same as baseline)
        self.L1 = nn.Linear(features_dim, hidden_size // 2)
        self.L2 = nn.Linear(dfeatures_dim, hidden_size // 2)

        # GAT layers for local graph processing
        self.n_gat_layers = n_gat_layers
        if self.n_gat_layers > 0:
            self.gat_layers = nn.ModuleList([
                GATConv(in_channels=hidden_size, out_channels=hidden_size, 
                       heads=n_heads, concat=False, dropout=0.1)
                for _ in range(n_gat_layers)
            ])
            self.gat_layer_norm = nn.LayerNorm(hidden_size)

        # Transformer attention layers for global context
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, d_ff, n_heads) 
            for _ in range(n_layers)
        ])

    def forward(self, emb_inp, rec_inp, mask, dummy_arg):
        batch_size = emb_inp.size(0)

        # Stage 1: Local graph processing with GAT
        if self.n_gat_layers > 0:
            gat_outputs = []
            for i in range(batch_size):
                x_i = emb_inp[i]
                adj_i = mask[i]
                edge_index_i = dense_to_sparse_edge_index(adj_i)
                
                # Pass through GAT layers
                for gat_layer in self.gat_layers:
                    identity = x_i
                    out = gat_layer(x_i, edge_index_i)
                    out = F.relu(out)
                    x_i = identity + out

                gat_outputs.append(x_i)
            
            gat_processed_emb = torch.stack(gat_outputs, dim=0)
            # Residual connection + layer norm
            emb_inp = self.gat_layer_norm(emb_inp + F.relu(gat_processed_emb))

        # Stage 2: Global context with Transformer attention
        for layer in self.layers:
            emb_inp, _ = layer(emb_inp, rec_inp, mask)

        return emb_inp


class GATTransformerPointerNetwork(nn.Module):
    """
    Hybrid Pointer Network with GAT Encoder and Transformer Decoder.
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_dim, args):
        super(GATTransformerPointerNetwork, self).__init__()

        self.features_dim = features_dim
        self.dfeatures_dim = dfeatures_dim
        self.use_checkpoint = args.use_checkpoint
        self.hidden_dim = hidden_dim
        
        # Transformer Decoder
        self.decoder = TransformerDecoder(hidden_dim, args)
        
        # GAT-based Encoder
        self.encoder = GATEncoder(features_dim, dfeatures_dim, hidden_dim, args)
        
        # Dummy tensor for checkpointing
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._initialize_parameters()

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def _load_model_weights(self, path_string, device):
        self.load_state_dict(torch.load(path_string, map_location=device))

    def forward(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step=False):
        policy, dec_hidden, enc_outputs = self._one_step(enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step)
        return policy, dec_hidden, enc_outputs

    def _one_step(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step):
        if self.use_checkpoint:
            enc_outputs = checkpoint.checkpoint(self.encoder, enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)
        else:
            enc_outputs = self.encoder(enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)

        if first_step:
            return None, None, enc_outputs
        else:
            # dec_input is [b_size x hidden_size] (embedding)
            # We need to unsqueeze to [b_size x 1 x hidden_size] for the decoder
            if dec_input.dim() == 2:
                dec_input = dec_input.unsqueeze(1)
                
            policy, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs, mask)
            return policy, dec_hidden, enc_outputs

    def sta_emb(self, sta_inp):
        return torch.tanh(self.encoder.L1(sta_inp))

    def dyn_emb(self, dyn_inp):
        return torch.tanh(self.encoder.L2(dyn_inp))

```

## `src/neural_net_transformer.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/neural_net_transformer.py`
- **Size**: 12485 bytes
- **Last modified**: 2025-11-26 23:43:53

```python
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import math
import numpy as np

# ------------------------------------------------------------------------------
# Transformer model components
# Reusing and adapting from src/neural_net.py

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v]
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask==0, -1e32)

        attn = self.softmax(attn)
        outputs = torch.bmm(attn, v) # outputs: [b_size x len_q x d_v]
        return outputs, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(_MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_v))

        self.attention = ScaledDotProductAttention(self.d_k)
        
        init.xavier_uniform_(self.w_q)
        init.xavier_uniform_(self.w_k)
        init.xavier_uniform_(self.w_v)

    def forward(self, q, k, v, attn_mask=None, is_adj=True):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)

        if attn_mask is not None:
            if is_adj:
                outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_heads, 1, 1))
            else:
                outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.unsqueeze(1).repeat(n_heads, 1, 1))
        else:
            outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=None)

        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.attention = _MultiHeadAttention(d_model, n_heads)
        self.proj = nn.Linear(n_heads * self.d_k, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask = None, is_adj = True):
        residual = q
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask, is_adj=is_adj)
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.proj(outputs)
        return self.layer_norm(residual + outputs), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        outputs = self.w_2(F.relu(self.w_1(x)))
        return self.layer_norm(residual + outputs)


# ------------------------------------------------------------------------------
# New Transformer Decoder Components

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super(TransformerDecoderLayer, self).__init__()
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, dec_input, enc_outputs, mask):
        # dec_input: [b_size x 1 x d_model] (Query)
        # enc_outputs: [b_size x seq_len x d_model] (Key, Value)
        # mask: [b_size x seq_len]
        
        # Cross Attention
        # Note: is_adj=False because mask is [b_size x seq_len], not [b_size x seq_len x seq_len]
        # We want to mask positions in enc_outputs that are invalid/visited
        
        # The existing MultiHeadAttention logic for `is_adj=False` expects mask to be [b_size x seq_len] 
        # and expands it to [n_heads x b_size x 1 x seq_len] inside _MultiHeadAttention if we pass it correctly.
        # Let's check _MultiHeadAttention.forward:
        # if is_adj: ... attn_mask.repeat(n_heads, 1, 1) -> [n_heads*b_size x seq_len x seq_len]
        # else: ... attn_mask.unsqueeze(1).repeat(n_heads, 1, 1) -> [n_heads*b_size x 1 x seq_len]
        
        # Our mask is [b_size x seq_len].
        # We need to pass it such that it broadcasts correctly.
        
        out, attn = self.cross_attn(dec_input, enc_outputs, enc_outputs, attn_mask=mask, is_adj=False)
        out = self.pos_ffn(out)
        return out, attn

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, args):
        super(TransformerDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = args.n_layers # Reuse n_layers or add a new arg if needed
        self.n_heads = args.n_heads
        self.ff_dim = args.ff_dim
        
        # Input projection to combine graph context + current node
        # We assume input is [current_node_emb] + [graph_emb]
        self.input_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, self.ff_dim, self.n_heads) 
            for _ in range(self.n_layers)
        ])
        
        # Final Pointer Attention (Single Head to get logits)
        # We calculate logits = (Q @ K^T) / sqrt(d_k)
        # We can use a linear projection for Q and K before dot product
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.scale_factor = np.sqrt(hidden_size)
        
        # Value head for PPO
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Dummy hidden state for compatibility with RunEpisode
        self.dummy_h = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)
        self.dummy_c = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)
        self.hidden_0 = (self.dummy_h, self.dummy_c)

    def forward(self, dec_input, hidden, enc_outputs, mask, return_value=False):
        # dec_input: [b_size x 1 x hidden_size] (Current Node Embedding)
        # enc_outputs: [b_size x seq_len x hidden_size]
        # mask: [b_size x seq_len] (1 for valid, 0 for invalid/visited)
        # return_value: If True, also return value prediction
        
        b_size = dec_input.size(0)
        
        # 1. Construct Context
        # Graph Embedding: Mean of encoder outputs (simple version)
        # [b_size x 1 x hidden_size]
        graph_emb = enc_outputs.mean(dim=1, keepdim=True)
        
        # Concatenate [graph_emb; dec_input]
        # [b_size x 1 x 2*hidden_size]
        combined_input = torch.cat([graph_emb, dec_input], dim=2)
        
        # Project to hidden_size
        # [b_size x 1 x hidden_size]
        query = self.input_proj(combined_input)
        
        # 2. Pass through Decoder Layers
        for layer in self.layers:
            query, _ = layer(query, enc_outputs, mask)
            
        # 3. Compute Pointer Logits
        # Q: [b_size x 1 x hidden_size]
        # K: [b_size x seq_len x hidden_size]
        
        Q = self.W_q(query)
        K = self.W_k(enc_outputs)
        
        # Dot product
        # [b_size x 1 x seq_len]
        logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale_factor
        
        # Squeeze to [b_size x seq_len]
        logits = logits.squeeze(1)
        
        policy = F.softmax(logits + mask.float().log(), dim=1)
        
        # Return dummy hidden state
        dummy_hidden = (self.dummy_h.expand(b_size, -1), self.dummy_c.expand(b_size, -1))
        
        if return_value:
            # Compute value prediction from query representation
            # query: [b_size x 1 x hidden_size]
            value = self.value_head(query)  # [b_size x 1 x 1]
            value = value.squeeze(-1)  # [b_size x 1]
            return policy, dummy_hidden, value
        else:
            return policy, dummy_hidden


# ------------------------------------------------------------------------------
# Encoder (Same as original)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, enc_inp, rec_enc_inp, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inp, rec_enc_inp, enc_inp, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, features_dim, dfeatures_dim, hidden_size, args):
        super(Encoder, self).__init__()

        n_heads = args.n_heads
        d_ff = args.ff_dim
        n_layers = args.n_layers

        self.L1 = nn.Linear(features_dim, hidden_size//2)
        self.L2 = nn.Linear(dfeatures_dim, hidden_size//2)

        self.layers = nn.ModuleList([EncoderLayer(hidden_size, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, emb_inp, rec_inp, mask, dummy_arg):
        for layer in self.layers:
            emb_inp, _ = layer(emb_inp, rec_inp, mask)

        return emb_inp


# ------------------------------------------------------------------------------
# TransformerPointerNetwork

class TransformerPointerNetwork(nn.Module):

    def __init__(self, features_dim, dfeatures_dim, hidden_dim, args):
        super(TransformerPointerNetwork, self).__init__()

        self.features_dim = features_dim
        self.dfeatures_dim = dfeatures_dim
        self.use_checkpoint = args.use_checkpoint
        self.hidden_dim = hidden_dim
        
        # CHANGED: Use TransformerDecoder
        self.decoder = TransformerDecoder(hidden_dim, args)
        
        self.encoder = Encoder(features_dim, dfeatures_dim, hidden_dim, args)
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._initialize_parameters()

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def _load_model_weights(self, path_string, device):
        self.load_state_dict(torch.load(path_string, map_location=device), strict=False)


    def forward(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step=False):
        policy, dec_hidden, enc_outputs = self._one_step(enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step)
        return policy, dec_hidden, enc_outputs

    def _one_step(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step):
        if self.use_checkpoint:
            enc_outputs = checkpoint.checkpoint(self.encoder, enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)
        else:
            enc_outputs = self.encoder(enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)

        if first_step:
            return None, None, enc_outputs
        else:
            # dec_input is [b_size x hidden_size] (embedding)
            # We need to unsqueeze to [b_size x 1 x hidden_size] for the decoder
            if dec_input.dim() == 2:
                dec_input = dec_input.unsqueeze(1)
                
            policy, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs, mask)
            return policy, dec_hidden, enc_outputs

    def sta_emb(self, sta_inp):
        return torch.tanh(self.encoder.L1(sta_inp))

    def dyn_emb(self, dyn_inp):
        return torch.tanh(self.encoder.L2(dyn_inp))

```

## `src/problem_config.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/problem_config.py`
- **Size**: 372 bytes
- **Last modified**: 2025-11-03 21:32:31

```python
# Indices of instance data
X_COORDINATE_IDX = 0
Y_COORDINATE_IDX = 1
VIS_DURATION_TIME_IDX = 2
OPENING_TIME_WINDOW_IDX = 3
CLOSING_TIME_WINDOW_IDX = 4
REWARD_IDX = 5
ARRIVAL_TIME_IDX = 6

# For generating instances
SAMP_DAY_FRAC_INF = 4/24.
UB_T_INIT_FRAC = 15/24.
LB_T_MAX_FRAC = 12/24.
CORR_SCORE_STD = 10

MULTIPLE_SCORE = 1.1

X_MAX = 100. # max square length (X_MAX)

```

## `src/sampling_norm_utils.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/sampling_norm_utils.py`
- **Size**: 4559 bytes
- **Last modified**: 2025-11-03 21:32:31

```python
import torch

import src.config as cf
import src.problem_config as pcf


def sample_new_instance(inst_data, dist_mat, args):
    instance_type = args.instance_type
    sample_type  = args.sample_type

    if instance_type=='Solomon':
        n_digits = 10.0
        xy_inf = 0.
        xy_delta = 100.
    elif instance_type=='Cordeau':
        n_digits = 100.0
        xy_inf = -100.
        xy_delta = 200.
    elif instance_type=='Gavalas':
        n_digits = 100.0
        xy_inf = 0.
        xy_delta = 100.

    poit = inst_data.clone()
    n = inst_data.shape[0]

    prof = inst_data[1:-1, pcf.REWARD_IDX]
    durat_max = int(inst_data[1:-1, pcf.VIS_DURATION_TIME_IDX].max().item())

    day_duration = int(inst_data[:, pcf.CLOSING_TIME_WINDOW_IDX].max().item())

    t_init_real = int(inst_data[0, pcf.OPENING_TIME_WINDOW_IDX].item()) # starting time
    t_max_real = int(inst_data[0, pcf.ARRIVAL_TIME_IDX].item()) # max arrival time

    day_fract_inf = pcf.SAMP_DAY_FRAC_INF
    t_min = int(pcf.SAMP_DAY_FRAC_INF*day_duration)
    ub_t_init_val = pcf.UB_T_INIT_FRAC*day_duration
    lb_t_max_val = pcf.LB_T_MAX_FRAC*day_duration

    ub = int(torch.min(torch.tensor([ub_t_init_val,
                                     t_max_real+int(day_fract_inf*day_duration)])))
    t_init_val = torch.randint(int(t_init_real)-int(day_fract_inf*day_duration),
                               ub,
                               (1,))

    lb = int(torch.max(torch.tensor([lb_t_max_val, int(t_init_val)+t_min])))
    t_max_val = torch.randint(lb,
                              t_max_real+int(day_fract_inf*day_duration),
                              (1,))

    Smax = int(torch.round(pcf.MULTIPLE_SCORE*prof.max()).item())
    if sample_type == 'uni_samp':
        new_scores = torch.randint(1, Smax, (n,))
    elif sample_type == 'corr_samp':
        new_scores_unbound = (Smax/durat_max)*inst_data[:, pcf.VIS_DURATION_TIME_IDX] + pcf.CORR_SCORE_STD*torch.randn(n, device=args.device)
        new_scores = torch.round(torch.min(Smax*torch.ones(1, device=args.device),
                                           torch.max(torch.ones(n, device=args.device),
                                                     new_scores_unbound)))

    poit[:, pcf.REWARD_IDX] = new_scores

    #------------ correct first/last----------

    poit[0, pcf.REWARD_IDX] = 0 # starting point
    poit[n-1, pcf.REWARD_IDX] = 0 # ending point

    poit[0, pcf.X_COORDINATE_IDX] = float(xy_inf+xy_delta*torch.rand(1)) # starting point
    poit[n-1, pcf.X_COORDINATE_IDX] = poit[0, pcf.X_COORDINATE_IDX] # ending point

    poit[0, pcf.Y_COORDINATE_IDX] = float(xy_inf+xy_delta*torch.rand(1)) # starting point
    poit[n-1, pcf.Y_COORDINATE_IDX] = poit[0, pcf.Y_COORDINATE_IDX] # ending point

    poit[:, pcf.ARRIVAL_TIME_IDX] = t_max_val*torch.ones(n)
    poit[0, pcf.OPENING_TIME_WINDOW_IDX] = t_init_val*torch.ones(1)
    poit[n-1, pcf.OPENING_TIME_WINDOW_IDX] = t_init_val*torch.ones(1)
    poit[0, pcf.CLOSING_TIME_WINDOW_IDX] = t_max_val*torch.ones(1)
    poit[n-1, pcf.CLOSING_TIME_WINDOW_IDX] = t_max_val*torch.ones(1)

    start_time = t_init_val.clone().detach().to(args.device)
    dist_matn = dist_mat.clone()

    for j in range(1, n-1):
        dist_matn[0, j] = float(torch.floor(n_digits*(torch.sqrt((poit[0,0]-poit[j,0])**2+(poit[0,1]-poit[j,1])**2))).item()/n_digits)
        dist_matn[n-1, j] = dist_matn[0, j]

        dist_matn[j, 0] = dist_matn[0, j]
        dist_matn[j, n-1] = dist_matn[n-1, j]

    return poit, start_time, dist_matn


def data_scaler(data, norm_dic):
    datan = data.clone()
    datan[:, pcf.X_COORDINATE_IDX] /= pcf.X_MAX
    datan[:, pcf.Y_COORDINATE_IDX] /= pcf.X_MAX
    datan[:, pcf.VIS_DURATION_TIME_IDX] /= (datan[:, pcf.VIS_DURATION_TIME_IDX].max())
    datan[:, pcf.OPENING_TIME_WINDOW_IDX] /= norm_dic['Tmax']
    datan[:, pcf.CLOSING_TIME_WINDOW_IDX ] /= norm_dic['Tmax']
    datan[:, pcf.REWARD_IDX] /= norm_dic['Smax']
    datan[:, pcf.ARRIVAL_TIME_IDX] /= norm_dic['Tmax']

    return datan


def instance_dependent_norm_const(instance_raw_data):
    day_duration = int(instance_raw_data[:, pcf.CLOSING_TIME_WINDOW_IDX].max().item())
    t_max_real = int(instance_raw_data[0, pcf.ARRIVAL_TIME_IDX].item()) # max instance arrival time
    arrival_time_val_ub = t_max_real+int(pcf.SAMP_DAY_FRAC_INF*day_duration)
    Tmax = int(max(day_duration, arrival_time_val_ub)) # max possible time value
    Smax = int(torch.round(pcf.MULTIPLE_SCORE*instance_raw_data[1:-1, pcf.REWARD_IDX].max()).item()) # max score

    return Tmax, Smax

```

## `src/solution_construction.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/solution_construction.py`
- **Size**: 19480 bytes
- **Last modified**: 2025-11-22 22:56:12

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.distributions import Categorical

import sys

from src.features_utils import DynamicFeatures
import src.problem_config as pcf

ourlogzero = sys.float_info.min


class Lookahead():
    def __init__(self, args):
        super(Lookahead, self).__init__()

        self.device = args.device
        self.opening_time_window_idx = pcf.OPENING_TIME_WINDOW_IDX
        self.closing_time_window_idx = pcf.CLOSING_TIME_WINDOW_IDX
        self.vis_duration_time_idx = pcf.VIS_DURATION_TIME_IDX
        self.arrival_time_idx = pcf.ARRIVAL_TIME_IDX

    def adjacency_matrix(self, braw_inputs, mask, dist_mat, pres_act, present_time):
        # feasible neighborhood for each node
        maskk = mask.clone()
        step_batch_size, npoints = mask.shape

        #one step forward update
        arrivej = dist_mat[pres_act] + present_time
        farrivej = arrivej.view(step_batch_size, npoints)
        tw_start = braw_inputs[:, :, self.opening_time_window_idx]
        waitj = torch.max(torch.FloatTensor([0.0]).to(self.device), tw_start-farrivej)
        durat = braw_inputs[:, : , self.vis_duration_time_idx]

        fpresent_time = farrivej + waitj + durat
        fpres_act = torch.arange(0, npoints, device=self.device).expand(step_batch_size, -1)

        # feasible neighborhood for each node
        adj_mask = maskk.unsqueeze(1).repeat(1, npoints, 1)
        arrivej = dist_mat.expand(step_batch_size, -1, -1) + fpresent_time.unsqueeze(2)
        waitj = torch.max(torch.FloatTensor([0.0]).to(self.device), tw_start.unsqueeze(2)-arrivej)

        tw_end = braw_inputs[:, :, self.closing_time_window_idx]
        ttime = braw_inputs[:, 0, self.arrival_time_idx]

        dlast = dist_mat[:, -1].unsqueeze(0).expand(step_batch_size, -1)

        c1 = arrivej + waitj <= tw_end.unsqueeze(1)
        c2 = arrivej + waitj + durat.unsqueeze(1) + dlast.unsqueeze(1) <= ttime.unsqueeze(1).unsqueeze(1).expand(-1, npoints, npoints)
        adj_mask = adj_mask * c1 * c2

        # self-loop
        idx = torch.arange(0, npoints, device=self.device).expand(step_batch_size, -1)
        adj_mask[:, idx, idx] = 1

        return adj_mask



class ModelUtils():
    def __init__(self, args):
        super(ModelUtils, self).__init__()

        self.device = args.device
        self.opening_time_window_idx = pcf.OPENING_TIME_WINDOW_IDX
        self.closing_time_window_idx = pcf.CLOSING_TIME_WINDOW_IDX
        self.vis_duration_time_idx = pcf.VIS_DURATION_TIME_IDX
        self.arrival_time_idx = pcf.ARRIVAL_TIME_IDX

    def feasibility_control(self, braw_inputs, mask, dist_mat, pres_act, present_time, batch_idx, first_step=False):

        done = False
        maskk = mask.clone()
        step_batch_size = batch_idx.shape[0]

        arrivej = dist_mat[pres_act] + present_time
        waitj = torch.max(torch.FloatTensor([0.0]).to(self.device), braw_inputs[:, :, self.opening_time_window_idx]-arrivej)

        c1 = arrivej + waitj <= braw_inputs[:, :, self.closing_time_window_idx]
        c2 = arrivej + waitj + braw_inputs[:, :, self.vis_duration_time_idx] + dist_mat[:, -1] <= braw_inputs[0, 0, self.arrival_time_idx]

        if not first_step:
            maskk[batch_idx, pres_act] = 0

        maskk[batch_idx] = maskk[batch_idx] * c1 * c2

        if maskk[:, -1].any() == 0:
            done = True
        return done, maskk


    def one_step_update(self, raw_inputs_b, dist_mat, pres_action, future_action, present_time, batch_idx, batch_size):

        present_time_b = torch.zeros(batch_size, 1, device=self.device)
        pres_actions_b = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        step_mask_b = torch.zeros(batch_size, 1, device=self.device, requires_grad=False, dtype=torch.bool)

        arrive_j = dist_mat[pres_action, future_action].unsqueeze(1) + present_time
        wait_j = torch.max(torch.FloatTensor([0.0]).to(self.device),
                           raw_inputs_b[batch_idx, future_action, self.opening_time_window_idx].unsqueeze(1)-arrive_j)
        present_time = arrive_j + wait_j + raw_inputs_b[batch_idx, future_action, self.vis_duration_time_idx].unsqueeze(1)

        present_time_b[batch_idx] = present_time

        pres_actions_b[batch_idx] = future_action
        step_mask_b[batch_idx] = 1

        return pres_actions_b, present_time_b, step_mask_b



class RunEpisode(nn.Module):

    def __init__(self, neuralnet, args):
        super(RunEpisode, self).__init__()

        self.device = args.device
        self.neuralnet = neuralnet
        self.dyn_feat = DynamicFeatures(args)
        self.lookahead = Lookahead(args)
        self.mu = ModelUtils(args)

    def forward(self, binputs, bdata_scaled, start_time, dist_mat, infer_type):

        self.batch_size, sequence_size, input_size = binputs.size()

        h_0, c_0 = self.neuralnet.decoder.hidden_0

        dec_hidden = (h_0.expand(self.batch_size, -1), c_0.expand(self.batch_size, -1))

        mask = torch.ones(self.batch_size, sequence_size, device=self.device, requires_grad=False, dtype = torch.uint8)

        bpresent_time = start_time*torch.ones(self.batch_size, 1, device=self.device)

        llog_probs, lactions, lstep_mask, lentropy = [], [], [], []

        bpres_actions = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)

        batch_idx = torch.arange(0, self.batch_size, device=self.device)

        done, mask = self.mu.feasibility_control(binputs[batch_idx], mask, dist_mat, bpres_actions,
                                                 bpresent_time, batch_idx, first_step=True)

        adj_mask = self.lookahead.adjacency_matrix(binputs[batch_idx], mask, dist_mat, bpres_actions, bpresent_time)

        # encoder first forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(binputs, bpresent_time, bpres_actions, dist_mat, batch_idx)
        emb1 = self.neuralnet.sta_emb(bdata_scaled)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1,emb2), dim=2)

        _, _, enc_outputs = self.neuralnet(enc_inputs, enc_inputs, adj_mask, enc_inputs, dec_hidden, mask, first_step=True)

        decoder_input = enc_outputs[batch_idx, bpres_actions]

        done, mask = self.mu.feasibility_control(binputs[batch_idx], mask, dist_mat, bpres_actions, bpresent_time, batch_idx)
        adj_mask = self.lookahead.adjacency_matrix(binputs[batch_idx], mask, dist_mat, bpres_actions, bpresent_time)

        # encoder/decoder forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(binputs, bpresent_time,
                                                      bpres_actions, dist_mat, batch_idx)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1,emb2), dim=2)

        policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden, mask)

        lactions.append(bpres_actions)

        # Starting the trip
        while not done:

            future_actions, log_probs, entropy = self.select_actions(policy, infer_type)

            bpres_actions, bpresent_time, bstep_mask = self.mu.one_step_update(binputs, dist_mat, bpres_actions[batch_idx],
                                                                               future_actions, bpresent_time[batch_idx],
                                                                               batch_idx, self.batch_size)

            blog_probs = torch.zeros(self.batch_size, 1, dtype=torch.float32).to(self.device)
            blog_probs[batch_idx] = log_probs.unsqueeze(1)

            bentropy = torch.zeros(self.batch_size,1,dtype=torch.float32).to(self.device)
            bentropy[batch_idx] = entropy.unsqueeze(1)

            llog_probs.append(blog_probs)
            lactions.append(bpres_actions)
            lstep_mask.append(bstep_mask)
            lentropy.append(bentropy)

            done, mask = self.mu.feasibility_control(binputs[batch_idx], mask, dist_mat,
                                                     bpres_actions[batch_idx], bpresent_time[batch_idx],
                                                     batch_idx)

            if done: break
            sub_batch_idx = torch.nonzero(mask[batch_idx][:,-1], as_tuple=False).squeeze(1)

            batch_idx = torch.nonzero(mask[:,-1], as_tuple=False).squeeze(1)

            adj_mask = self.lookahead.adjacency_matrix(binputs[batch_idx], mask[batch_idx], dist_mat, bpres_actions[batch_idx], bpresent_time[batch_idx])

            #update decoder input and hidden
            decoder_input = enc_outputs[sub_batch_idx, bpres_actions[sub_batch_idx]]
            dec_hidden = (dec_hidden[0][sub_batch_idx], dec_hidden[1][sub_batch_idx])

            # encoder/decoder forward pass
            bdyn_inputs = self.dyn_feat.make_dynamic_feat(binputs, bpresent_time[batch_idx], bpres_actions[batch_idx], dist_mat, batch_idx)
            emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
            enc_inputs = torch.cat((emb1[batch_idx],emb2), dim=2)

            policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs[sub_batch_idx], adj_mask, decoder_input, dec_hidden, mask[batch_idx])

        return lactions, torch.cat(llog_probs, dim=1), torch.cat(lentropy, dim=1), torch.cat(lstep_mask, dim=1)


    def select_actions(self, policy, infer_type):

        if infer_type == 'stochastic':
            m = Categorical(policy)
            act_ind = m.sample()
            log_select =  m.log_prob(act_ind)
            poli_entro = m.entropy()
        elif infer_type == 'greedy':
            prob, act_ind = torch.max(policy, 1)
            log_select =  prob.log()
            poli_entro =  torch.zeros(self.batch_size, requires_grad=False).to(self.device)

        return act_ind, log_select, poli_entro



class BeamSearch(nn.Module):
    def __init__(self, neuralnet, args):
        super(BeamSearch, self).__init__()

        self.device = args.device
        self.neuralnet = neuralnet
        self.dyn_feat = DynamicFeatures(args)
        self.lookahead = Lookahead(args)
        self.mu = ModelUtils(args)

    def forward(self, inputs, data_scaled, start_time, dist_mat, infer_type, beam_size):
        self.beam_size = beam_size
        _, sequence_size, input_size = inputs.size()

        # first step  - node 0
        bpresent_time = start_time*torch.ones(1, 1, device=self.device)

        mask = torch.ones(1, sequence_size, device=self.device, requires_grad=False, dtype= torch.uint8)
        bpres_actions = torch.zeros(1, dtype=torch.int64,device=self.device)
        beam_idx = torch.arange(0, 1, device=self.device)

        done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                 mask, dist_mat, bpres_actions, bpresent_time,
                                                 torch.arange(0, mask.shape[0], device=self.device),
                                                 first_step=True)
        adj_mask = self.lookahead.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                   mask, dist_mat, bpres_actions, bpresent_time)

        h_0, c_0 = self.neuralnet.decoder.hidden_0
        dec_hidden = (h_0.expand(1, -1), c_0.expand(1, -1))

        step = 0

        # encoder first forward pass
        bdata_scaled = data_scaled.expand(1,-1,-1)
        sum_log_probs = torch.zeros(1, device=self.device).float()

        bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(1,-1,-1), bpresent_time, bpres_actions, dist_mat, beam_idx)
        emb1 = self.neuralnet.sta_emb(bdata_scaled)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1, emb2), dim=2)

        _, _, enc_outputs = self.neuralnet(enc_inputs, enc_inputs, adj_mask, enc_inputs, dec_hidden, mask, first_step=True)

        decoder_input = enc_outputs[beam_idx, bpres_actions]

        done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                 mask, dist_mat, bpres_actions, bpresent_time,
                                                 torch.arange(0, mask.shape[0], device=self.device))
        adj_mask = self.lookahead.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                   mask, dist_mat, bpres_actions, bpresent_time)

        # encoder/decoder forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(beam_idx.shape[0], -1, -1), bpresent_time, bpres_actions, dist_mat, beam_idx)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1,emb2), dim=2)

        policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden, mask)

        future_actions, log_probs, beam_idx = self.select_actions(policy, sum_log_probs, mask, infer_type)
        # info update
        h_step = torch.index_select(dec_hidden[0], dim=0, index = beam_idx)
        c_step = torch.index_select(dec_hidden[1], dim=0, index = beam_idx)
        dec_hidden = (h_step,c_step)

        mask = torch.index_select(mask, dim=0, index=beam_idx)
        bpresent_time = torch.index_select(bpresent_time, dim=0, index=beam_idx)
        bpres_actions = torch.index_select(bpres_actions, dim=0, index=beam_idx)
        enc_outputs  = torch.index_select(enc_outputs, dim=0, index=beam_idx)
        sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=beam_idx)

        emb1 = torch.index_select(emb1, dim=0, index=beam_idx)

        # initialize buffers
        bllog_probs = torch.zeros(bpres_actions.shape[0], sequence_size, device=self.device).float()
        blactions = torch.zeros(bpres_actions.shape[0], sequence_size, device=self.device).long()

        sum_log_probs += log_probs.squeeze(0).detach()

        blactions[:, step] = bpres_actions

        final_log_probs, final_actions, lstep_mask = [], [], []

        # Starting the trip
        while not done:

            future_actions = future_actions.squeeze(0)

            beam_size = bpres_actions.shape[0]
            bpres_actions, bpresent_time, bstep_mask = \
                self.mu.one_step_update(inputs.expand(beam_size, -1, -1), dist_mat,
                                        bpres_actions, future_actions, bpresent_time,
                                        torch.arange(0,beam_size,device=self.device),
                                        beam_size)

            bllog_probs[:, step] = log_probs
            blactions[:, step+1] = bpres_actions
            step+=1

            done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                     mask, dist_mat, bpres_actions, bpresent_time,
                                                     torch.arange(0, mask.shape[0], device=self.device))
            adj_mask = self.lookahead.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                       mask, dist_mat, bpres_actions, bpresent_time)

            active_beam_idx = torch.nonzero(mask[:, -1], as_tuple=False).squeeze(1)
            end_beam_idx = torch.nonzero((mask[:, -1]==0), as_tuple=False).squeeze(1)

            if end_beam_idx.shape[0]>0:

                final_log_probs.append(torch.index_select(bllog_probs, dim=0, index=end_beam_idx))
                final_actions.append(torch.index_select(blactions, dim=0, index=end_beam_idx))

                # ending seq info update
                h_step = torch.index_select(dec_hidden[0], dim=0, index = active_beam_idx)
                c_step = torch.index_select(dec_hidden[1], dim=0, index = active_beam_idx)
                dec_hidden = (h_step,c_step)

                mask = torch.index_select(mask, dim=0, index=active_beam_idx)
                adj_mask = torch.index_select(adj_mask, dim=0, index=active_beam_idx)

                bpresent_time = torch.index_select(bpresent_time, dim=0, index=active_beam_idx)
                bpres_actions = torch.index_select(bpres_actions, dim=0, index=active_beam_idx)
                enc_outputs  = torch.index_select(enc_outputs, dim=0, index=active_beam_idx)

                emb1 = torch.index_select(emb1, dim=0, index=active_beam_idx)

                blactions = torch.index_select(blactions, dim=0, index=active_beam_idx)
                bllog_probs = torch.index_select(bllog_probs, dim=0, index=active_beam_idx)
                sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=active_beam_idx)

            if done: break
            decoder_input = enc_outputs[torch.arange(0, bpres_actions.shape[0], device=self.device), bpres_actions]

            bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(beam_idx.shape[0], -1, -1), bpresent_time, bpres_actions, dist_mat, active_beam_idx)
            emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
            enc_inputs = torch.cat((emb1,emb2), dim=2)

            policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden, mask)

            future_actions, log_probs, beam_idx = self.select_actions(policy, sum_log_probs, mask, infer_type)

            # info update
            h_step = torch.index_select(dec_hidden[0], dim=0, index = beam_idx)
            c_step = torch.index_select(dec_hidden[1], dim=0, index = beam_idx)
            dec_hidden = (h_step,c_step)

            mask = torch.index_select(mask, dim=0, index=beam_idx)
            adj_mask = torch.index_select(adj_mask, dim=0, index=beam_idx)

            bpresent_time = torch.index_select(bpresent_time, dim=0, index=beam_idx)
            bpres_actions = torch.index_select(bpres_actions, dim=0, index=beam_idx)

            enc_outputs  = torch.index_select(enc_outputs, dim=0, index=beam_idx)

            emb1 = torch.index_select(emb1, dim=0, index=beam_idx)

            blactions = torch.index_select(blactions, dim=0, index=beam_idx)
            bllog_probs = torch.index_select(bllog_probs, dim=0, index=beam_idx)
            sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=beam_idx)

            sum_log_probs += log_probs.squeeze(0).detach()

        return torch.cat(final_actions, dim=0), torch.cat(final_log_probs, dim=0)



    def select_actions(self, policy, sum_log_probs, mask, infer_type = 'stochastic'):

        beam_size, seq_size = policy.size()
        nzn  = torch.nonzero(mask, as_tuple=False).shape[0]
        sample_size = min(nzn,self.beam_size)

        ourlogzero = sys.float_info.min
        lpolicy = policy.masked_fill(mask == 0, ourlogzero).log()
        npolicy = sum_log_probs.unsqueeze(1) + lpolicy
        if infer_type == 'stochastic':
            nnpolicy = npolicy.exp().masked_fill(mask == 0, 0).view(1, -1)

            m = Categorical(nnpolicy)
            gact_ind = torch.multinomial(nnpolicy, sample_size)
            log_select =  m.log_prob(gact_ind)

        elif infer_type == 'greedy':
            nnpolicy = npolicy.exp().masked_fill(mask == 0, 0).view(1, -1)

            _ , gact_ind = nnpolicy.topk(sample_size, dim = 1)
            prob = policy.view(-1)[gact_ind]
            log_select =  prob.log()

        beam_id = torch.floor_divide(gact_ind, seq_size).squeeze(0)
        act_ind = torch.fmod(gact_ind, seq_size)

        return act_ind, log_select, beam_id

```

## `src/train_utils.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/train_utils.py`
- **Size**: 3120 bytes
- **Last modified**: 2025-11-22 22:56:12

```python
import torch
import random

from src.sampling_norm_utils import sample_new_instance, data_scaler
import src.problem_config as pcf


def reward_fn(data, sample_solution, device):
    """
    Returns:
        Tensor of shape [batch_size] containing rewards
    """

    batch_size = sample_solution[0].shape[0]
    tour_reward = torch.zeros(batch_size, device=device)

    for act_id in sample_solution:
        tour_reward += data[act_id, pcf.REWARD_IDX].squeeze(0)

    return tour_reward


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_step=5000):
    """Decay learning rate by a factor of 0.96 every lr_decay_epoch epochs.
       Lower_bounded at 0.00001"""
    lr = init_lr * (0.96**(epoch // lr_decay_step))
    if lr < 0.00001:
        lr = 0.00001

    if epoch % lr_decay_step == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def samples2batch(new_data, new_data_scaled, batch_size):
    bnew_data = new_data.expand(batch_size, -1, -1)
    bnew_data_scaled = new_data_scaled.expand(batch_size, -1, -1)
    return bnew_data, bnew_data_scaled


def train_model(raw_data, raw_dist_mat, norm_dic, run_episode, opt, args):

    new_data, start_time, dist_mat = sample_new_instance(raw_data, raw_dist_mat, args)
    new_data_scaled = data_scaler(new_data, norm_dic[args.instance])
    bnew_data, bnew_data_scaled = samples2batch(new_data, new_data_scaled, args.batch_size)

    run_episode.train()
    opt.zero_grad()
    actions, log_prob, entropy, step_mask = run_episode(bnew_data, bnew_data_scaled, start_time, dist_mat, 'stochastic')

    rewards = reward_fn(new_data, actions, args.device)

    loss = 0

    av_rew = rewards.mean()
    min_rew = rewards.min()
    max_rew = rewards.max()

    advantage = (rewards - av_rew) #advantage

    res = advantage.unsqueeze(1)*log_prob + args.beta*entropy

    loss = -res[step_mask].sum()/args.batch_size

    loss.backward(retain_graph=False)
    torch.nn.utils.clip_grad_norm_(run_episode.neuralnet.parameters(), args.max_grad_norm)
    opt.step()

    return av_rew.item(), min_rew.item(), max_rew.item(), loss.item()


def test_model(data, start_time, dist_mat, inst, inst_norm_dic, run_episode, device):
    with torch.no_grad():
        data_scaled = data_scaler(data, inst_norm_dic[inst])
        bdata, bdata_scaled = data.unsqueeze(0), data_scaled.unsqueeze(0)
        actions, log_prob, entropy, step_mask = run_episode(bdata, bdata_scaled, start_time, dist_mat, 'greedy')
        reward = reward_fn(data, actions, device)

        return reward.item()


def validation(inp_val, run_episode, inst_norm_dic, device):
    reward_val =  torch.tensor(0.0).to(device)
    rew_dict = {}
    for k, (inst_name, data) in enumerate(inp_val):
        inst_data, start_time, dist_mat = data
        rew = test_model(inst_data, start_time, dist_mat, inst_name, inst_norm_dic, run_episode, device)
        reward_val += rew
        key_str = inst_name + '_' + str(k)
        rew_dict[key_str] = rew

    return rew_dict, reward_val.item()/len(inp_val)

```

## `src/utils.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/utils.py`
- **Size**: 9104 bytes
- **Last modified**: 2025-11-03 21:32:31

```python
import pandas as pd
import numpy as np
import logging
import torch

import src.config as cf
import src.problem_config as pcf


def read_instance_data(instance_name, path):

    """reads instance data"""
    PATH_TO_BENCHMARK_INSTANCES = path

    benchmark_file = '{path_to_benchmark_instances}/{instance}.txt' \
                     .format(path_to_benchmark_instances=PATH_TO_BENCHMARK_INSTANCES,
                             instance=instance_name)

    dfile = open(benchmark_file)
    data = [[float(x) for x in line.split()] for line in dfile]
    dfile.close()
    return data


def eliminate_extra_cordeau_columns(instance_data):
    """Cordeau instances have extra columns in some rows. This function eliminates the extra columns.
    This will also correct position of total time in row 0 for all instances"""
    DATA_INIT_ROW = 2
    N_RELEVANT_FIRST_COLUMNS = 8
    N_RELEVANT_LAST_COLUMNS = 2

    return [s[:N_RELEVANT_FIRST_COLUMNS]+s[-N_RELEVANT_LAST_COLUMNS:] \
            for s in instance_data[DATA_INIT_ROW :]]


def test_n_vert_1(instance_data, instance_type):
    N_VERT_ROW = 0

    if instance_type=='Gavalas':
        N_VERT_COL = 3
        DATA_INIT_ROW = 1
    else:
        N_VERT_COL = 2
        DATA_INIT_ROW = 2

    n_vert = instance_data[N_VERT_ROW][N_VERT_COL]
    count_vert = len(instance_data)-(DATA_INIT_ROW+1)

    assert count_vert==n_vert, 'number of vertices doesnt match number of data rows'


def test_n_vert_2(instance_data, instance_type):
    N_VERT_ROW = 0
    if instance_type=='Gavalas':
        N_VERT_COL = 3
    else:
        N_VERT_COL = 2
    COLUMN_NAMES = ['vertex number', 'x coordinate', 'y coordinate',
                    'service duration or visiting time', 'profit of the location',
                    'not relevant 1', 'not relevant 2', 'not relevant 3',
                    'opening of time window', 'closing of time window']
    COLUMN_NAMES = [s.replace(' ', '_') for s in COLUMN_NAMES]

    VERTEX_NUMBER_COL = [i for i,n in enumerate(COLUMN_NAMES) if n=='vertex_number'][0]
    n_vert = instance_data[N_VERT_ROW][N_VERT_COL]
    last_vert_number = instance_data[-1][VERTEX_NUMBER_COL]

    assert last_vert_number==n_vert, 'number of vertices doesnt match vertice count of last row'


def test_n_vert_3(instance_data, instance_type):
    if instance_type=='Gavalas':
        N_DAYS_INDEX = 1
        n_days = int(np.array(instance_data[0])[N_DAYS_INDEX])
        assert n_days==1, 'not a single tour/1 day instance'
    else:
        pass


def parse_instance_data_Gavalas(instance_data):
    """parse instance data into dataframe"""

    # get start date
    N_DAYS_INDEX = 1
    START_DAY_INDEX = 2
    M = instance_data[0][N_DAYS_INDEX]
    SD =  int(instance_data[0][START_DAY_INDEX])

    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 't',
                           'open_0', 'close_0', 'open_1', 'close_1',
                           'open_2', 'close_2', 'open_3', 'close_3',
                           'open_4', 'close_4', 'open_5', 'close_5',
                           'open_6', 'close_6', 'b']

    df = pd.DataFrame(instance_data[2:], columns=COLUMN_NAMES_ABBREV)

    df_ = df[['i', 'x', 'y', 'd', 'S', 't']+[c for c in df.columns if c[-1]==str(SD)]]
    columns = ['i', 'x', 'y', 'd', 'S', 't', OPENING_TIME_WINDOW_ABBREV_KEY, CLOSING_TIME_WINDOW_ABBREV_KEY]
    df_.columns=columns

    aux = pd.DataFrame([instance_data[1]], columns = ['i', 'x', 'y', 'd', 'S', 'O', 'C'])
    df = pd.concat([aux, df_], axis=0, sort=True).reset_index(drop=True)

    #add total time
    df[TOTAL_TIME_KEY] = 0
    df[TOTAL_TIME_KEY] = df.loc[0][CLOSING_TIME_WINDOW_ABBREV_KEY]

    return df


def parse_instance_data(instance_data):
    """parse instance data into dataframe"""

    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 'f', 'a', 'list', 'O', 'C']

    instance_data_clean = eliminate_extra_cordeau_columns(instance_data)
    df = pd.DataFrame(instance_data_clean, columns=COLUMN_NAMES_ABBREV)

    #add total time
    df[TOTAL_TIME_KEY] = 0
    df[TOTAL_TIME_KEY] = df.loc[0][CLOSING_TIME_WINDOW_ABBREV_KEY]

    return df


def get_instance_df(instance_name, path, instance_type):

    """combine read instance, tests and parse to dataframe"""

    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'


    COLUMN_NAMES = ['vertex number', 'x coordinate', 'y coordinate',
    'service duration or visiting time', 'profit of the location',
    'not relevant 1', 'not relevant 2', 'not relevant 3',
    'opening of time window', 'closing of time window']
    COLUMN_NAMES = [s.replace(' ', '_') for s in COLUMN_NAMES]
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 'f', 'a', 'list', 'O', 'C']
    VERTEX_NUMBER_COL = [i for i,n in enumerate(COLUMN_NAMES) if n=='vertex_number'][0]
    COLS_OF_INT = ['i', 'x', 'y', 'd', OPENING_TIME_WINDOW_ABBREV_KEY,
                   CLOSING_TIME_WINDOW_ABBREV_KEY, 'S', TOTAL_TIME_KEY]
    COLS_OF_INT_NEW_NAMES = ['i', 'x', 'y', 'duration', 'ti', 'tf', 'prof', TOTAL_TIME_KEY]

    standard2newnames_dict =  dict(((c, ca) for c, ca in zip(COLS_OF_INT, COLS_OF_INT_NEW_NAMES)))

    instance_data = read_instance_data(instance_name, path)

    # run tests
    test_n_vert_1(instance_data, instance_type)
    test_n_vert_2(instance_data, instance_type)
    # test if it's a single day (we are not considering TOPTW instances)
    test_n_vert_3(instance_data, instance_type)

    if instance_type=='Gavalas':
        df = parse_instance_data_Gavalas(instance_data)
    else:
        df = parse_instance_data(instance_data)

    #change column names
    COLS_OF_INT_NEW_NAMES = [standard2newnames_dict[s] for s in COLS_OF_INT]
    df_ = df[COLS_OF_INT].copy()
    df_.columns = COLS_OF_INT_NEW_NAMES
    df_['inst_name'] = instance_name
    df_['real_or_val'] = 'real'

    df_ = df_.append(df_.loc[0])
    return df_



def get_distance_matrix(instance_df, instance_type):
    """
    Distances between locations were rounded down to the first decimal
    for the Solomon instances and to the second decimal for the instances of Cordeau and Gavalas.
    """

    if instance_type in ['Solomon']:
        n_digits = 10.0

    elif instance_type in ['Cordeau', 'Gavalas']:
        n_digits = 100.0

    n = instance_df.shape[0]
    distm = np.zeros((n,n))
    x = instance_df.x.values
    y = instance_df.y.values

    for i in range(0, n-1):
        for j in range(i+1, n):
            distm[i,j] = np.floor(n_digits*(np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)))/n_digits
            distm[j,i] = distm[i,j]

    return distm



def get_instance_type(instance_name):
        if instance_name[:2]=='pr':
            return 'Cordeau'
        elif instance_name[0] in ['r', 'c']:
            return 'Solomon'
        elif instance_name[0] in ['t']:
            return 'Gavalas'
        raise Exception('weird instance')


def setup_logger(debug):
    logger = logging.getLogger()
    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger


def get_instance_data(instance, path, device):
    instance_type = get_instance_type(instance)
    df_inst = get_instance_df(instance, path, instance_type)
    distm = get_distance_matrix(df_inst, instance_type)
    raw_data = df_inst[['x', 'y', 'duration', 'ti', 'tf', 'prof', 'Total Time']].values
    raw_data = torch.FloatTensor(raw_data).to(device)
    raw_distm =  torch.FloatTensor(distm).to(device)

    return raw_data, raw_distm


def get_val_data(args, phase='train'):
    path_string = '{directory}/{file_name}'
    inp_val_path = path_string.format(directory=args.val_dir, file_name=args.val_set_pt_file)
    inp_val = torch.load(inp_val_path, map_location = args.map_location)

    new_inp_val = [(args.instance, inst_data) for inst_data in inp_val]

    if phase=='train':
        return new_inp_val
    else:

        return inp_val


def get_real_data(args, phase='train'):

    df = get_instance_df(args.instance, cf.BENCHMARK_INSTANCES_PATH,
                         args.instance_type)
    dist_mat = get_distance_matrix(df, args.instance_type)
    inp_real = df[['x', 'y', 'duration', 'ti', 'tf', 'prof', 'Total Time']].values

    if phase=='train':
        inp_real = [(torch.FloatTensor(inp_real).to(args.device),
                torch.tensor(inp_real[0, pcf.OPENING_TIME_WINDOW_IDX]).to(args.device),
                torch.FloatTensor(dist_mat).to(args.device))]

        new_inp_real = [(args.instance, inp_real[0])]
        return new_inp_real
    else:
        inp_real = [(torch.FloatTensor(inp_real).to(args.device),
                 torch.FloatTensor(dist_mat).to(args.device))]
        return inp_real
```

## `src/utils_transformer.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/src/utils_transformer.py`
- **Size**: 9211 bytes
- **Last modified**: 2025-11-19 20:03:30

```python
import pandas as pd
import numpy as np
import logging
import torch

import src.config as cf
import src.problem_config as pcf


def read_instance_data(instance_name, path):

    """reads instance data"""
    PATH_TO_BENCHMARK_INSTANCES = path

    benchmark_file = '{path_to_benchmark_instances}/{instance}.txt' \
                     .format(path_to_benchmark_instances=PATH_TO_BENCHMARK_INSTANCES,
                             instance=instance_name)

    dfile = open(benchmark_file)
    data = [[float(x) for x in line.split()] for line in dfile]
    dfile.close()
    return data


def eliminate_extra_cordeau_columns(instance_data):
    """Cordeau instances have extra columns in some rows. This function eliminates the extra columns.
    This will also correct position of total time in row 0 for all instances"""
    DATA_INIT_ROW = 2
    N_RELEVANT_FIRST_COLUMNS = 8
    N_RELEVANT_LAST_COLUMNS = 2

    return [s[:N_RELEVANT_FIRST_COLUMNS]+s[-N_RELEVANT_LAST_COLUMNS:] \
            for s in instance_data[DATA_INIT_ROW :]]


def test_n_vert_1(instance_data, instance_type):
    N_VERT_ROW = 0

    if instance_type=='Gavalas':
        N_VERT_COL = 3
        DATA_INIT_ROW = 1
    else:
        N_VERT_COL = 2
        DATA_INIT_ROW = 2

    n_vert = instance_data[N_VERT_ROW][N_VERT_COL]
    count_vert = len(instance_data)-(DATA_INIT_ROW+1)

    assert count_vert==n_vert, 'number of vertices doesnt match number of data rows'


def test_n_vert_2(instance_data, instance_type):
    N_VERT_ROW = 0
    if instance_type=='Gavalas':
        N_VERT_COL = 3
    else:
        N_VERT_COL = 2
    COLUMN_NAMES = ['vertex number', 'x coordinate', 'y coordinate',
                    'service duration or visiting time', 'profit of the location',
                    'not relevant 1', 'not relevant 2', 'not relevant 3',
                    'opening of time window', 'closing of time window']
    COLUMN_NAMES = [s.replace(' ', '_') for s in COLUMN_NAMES]

    VERTEX_NUMBER_COL = [i for i,n in enumerate(COLUMN_NAMES) if n=='vertex_number'][0]
    n_vert = instance_data[N_VERT_ROW][N_VERT_COL]
    last_vert_number = instance_data[-1][VERTEX_NUMBER_COL]

    assert last_vert_number==n_vert, 'number of vertices doesnt match vertice count of last row'


def test_n_vert_3(instance_data, instance_type):
    if instance_type=='Gavalas':
        N_DAYS_INDEX = 1
        n_days = int(np.array(instance_data[0])[N_DAYS_INDEX])
        assert n_days==1, 'not a single tour/1 day instance'
    else:
        pass


def parse_instance_data_Gavalas(instance_data):
    """parse instance data into dataframe"""

    # get start date
    N_DAYS_INDEX = 1
    START_DAY_INDEX = 2
    M = instance_data[0][N_DAYS_INDEX]
    SD =  int(instance_data[0][START_DAY_INDEX])

    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 't',
                           'open_0', 'close_0', 'open_1', 'close_1',
                           'open_2', 'close_2', 'open_3', 'close_3',
                           'open_4', 'close_4', 'open_5', 'close_5',
                           'open_6', 'close_6', 'b']

    df = pd.DataFrame(instance_data[2:], columns=COLUMN_NAMES_ABBREV)

    df_ = df[['i', 'x', 'y', 'd', 'S', 't']+[c for c in df.columns if c[-1]==str(SD)]]
    columns = ['i', 'x', 'y', 'd', 'S', 't', OPENING_TIME_WINDOW_ABBREV_KEY, CLOSING_TIME_WINDOW_ABBREV_KEY]
    df_.columns=columns

    aux = pd.DataFrame([instance_data[1]], columns = ['i', 'x', 'y', 'd', 'S', 'O', 'C'])
    df = pd.concat([aux, df_], axis=0, sort=True).reset_index(drop=True)

    #add total time
    df[TOTAL_TIME_KEY] = 0
    df[TOTAL_TIME_KEY] = df.loc[0][CLOSING_TIME_WINDOW_ABBREV_KEY]

    return df


def parse_instance_data(instance_data):
    """parse instance data into dataframe"""

    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 'f', 'a', 'list', 'O', 'C']

    instance_data_clean = eliminate_extra_cordeau_columns(instance_data)
    df = pd.DataFrame(instance_data_clean, columns=COLUMN_NAMES_ABBREV)

    #add total time
    df[TOTAL_TIME_KEY] = 0
    df[TOTAL_TIME_KEY] = df.loc[0][CLOSING_TIME_WINDOW_ABBREV_KEY]

    return df


def get_instance_df(instance_name, path, instance_type):

    """combine read instance, tests and parse to dataframe"""

    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'


    COLUMN_NAMES = ['vertex number', 'x coordinate', 'y coordinate',
    'service duration or visiting time', 'profit of the location',
    'not relevant 1', 'not relevant 2', 'not relevant 3',
    'opening of time window', 'closing of time window']
    COLUMN_NAMES = [s.replace(' ', '_') for s in COLUMN_NAMES]
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 'f', 'a', 'list', 'O', 'C']
    VERTEX_NUMBER_COL = [i for i,n in enumerate(COLUMN_NAMES) if n=='vertex_number'][0]
    COLS_OF_INT = ['i', 'x', 'y', 'd', OPENING_TIME_WINDOW_ABBREV_KEY,
                   CLOSING_TIME_WINDOW_ABBREV_KEY, 'S', TOTAL_TIME_KEY]
    COLS_OF_INT_NEW_NAMES = ['i', 'x', 'y', 'duration', 'ti', 'tf', 'prof', TOTAL_TIME_KEY]

    standard2newnames_dict =  dict(((c, ca) for c, ca in zip(COLS_OF_INT, COLS_OF_INT_NEW_NAMES)))

    instance_data = read_instance_data(instance_name, path)

    # run tests
    test_n_vert_1(instance_data, instance_type)
    test_n_vert_2(instance_data, instance_type)
    # test if it's a single day (we are not considering TOPTW instances)
    test_n_vert_3(instance_data, instance_type)

    if instance_type=='Gavalas':
        df = parse_instance_data_Gavalas(instance_data)
    else:
        df = parse_instance_data(instance_data)

    #change column names
    COLS_OF_INT_NEW_NAMES = [standard2newnames_dict[s] for s in COLS_OF_INT]
    df_ = df[COLS_OF_INT].copy()
    df_.columns = COLS_OF_INT_NEW_NAMES
    df_['inst_name'] = instance_name
    df_['real_or_val'] = 'real'

    # CHANGED: Use concat instead of append
    # df_ = df_.append(df_.loc[0])
    df_ = pd.concat([df_, df_.loc[[0]]], ignore_index=True)
    return df_



def get_distance_matrix(instance_df, instance_type):
    """
    Distances between locations were rounded down to the first decimal
    for the Solomon instances and to the second decimal for the instances of Cordeau and Gavalas.
    """

    if instance_type in ['Solomon']:
        n_digits = 10.0

    elif instance_type in ['Cordeau', 'Gavalas']:
        n_digits = 100.0

    n = instance_df.shape[0]
    distm = np.zeros((n,n))
    x = instance_df.x.values
    y = instance_df.y.values

    for i in range(0, n-1):
        for j in range(i+1, n):
            distm[i,j] = np.floor(n_digits*(np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)))/n_digits
            distm[j,i] = distm[i,j]

    return distm



def get_instance_type(instance_name):
        if instance_name[:2]=='pr':
            return 'Cordeau'
        elif instance_name[0] in ['r', 'c']:
            return 'Solomon'
        elif instance_name[0] in ['t']:
            return 'Gavalas'
        raise Exception('weird instance')


def setup_logger(debug):
    logger = logging.getLogger()
    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger


def get_instance_data(instance, path, device):
    instance_type = get_instance_type(instance)
    df_inst = get_instance_df(instance, path, instance_type)
    distm = get_distance_matrix(df_inst, instance_type)
    raw_data = df_inst[['x', 'y', 'duration', 'ti', 'tf', 'prof', 'Total Time']].values
    raw_data = torch.FloatTensor(raw_data).to(device)
    raw_distm =  torch.FloatTensor(distm).to(device)

    return raw_data, raw_distm


def get_val_data(args, phase='train'):
    path_string = '{directory}/{file_name}'
    inp_val_path = path_string.format(directory=args.val_dir, file_name=args.val_set_pt_file)
    inp_val = torch.load(inp_val_path, map_location = args.map_location)

    new_inp_val = [(args.instance, inst_data) for inst_data in inp_val]

    if phase=='train':
        return new_inp_val
    else:

        return inp_val


def get_real_data(args, phase='train'):

    df = get_instance_df(args.instance, cf.BENCHMARK_INSTANCES_PATH,
                         args.instance_type)
    dist_mat = get_distance_matrix(df, args.instance_type)
    inp_real = df[['x', 'y', 'duration', 'ti', 'tf', 'prof', 'Total Time']].values

    if phase=='train':
        inp_real = [(torch.FloatTensor(inp_real).to(args.device),
                torch.tensor(inp_real[0, pcf.OPENING_TIME_WINDOW_IDX]).to(args.device),
                torch.FloatTensor(dist_mat).to(args.device))]

        new_inp_real = [(args.instance, inp_real[0])]
        return new_inp_real
    else:
        inp_real = [(torch.FloatTensor(inp_real).to(args.device),
                 torch.FloatTensor(dist_mat).to(args.device))]
        return inp_real

```

## `train_missing_models.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/train_missing_models.py`
- **Size**: 1708 bytes
- **Last modified**: 2025-11-27 20:23:06

```python
import os
import subprocess
import sys
import time

# Configuration
PYTHON_EXEC = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES = ['c101', 'r101', 'rc101', 'pr01', 't101']

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(e)
        return False

def train_missing_model(instance, epochs=5000, batch_size=16, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Training GAT-LSTM (Benchmark) - Instance: {instance}")
    print(f"{'='*60}\n")

    script_name = "train_optw_gat_lstm.py"
    model_name = "gat_lstm_bench"
    
    cmd_train = (
        f"{PYTHON_EXEC} {script_name} "
        f"--instance {instance} "
        f"--model_name {model_name} "
        f"--nepocs {epochs} "
        f"--batch_size {batch_size} "
        f"--nprint 100 "
        f"--nsave 100 "
        f"--device {device} "
        f"--n_gat_layers 3"
    )
    
    return run_command(cmd_train)

def main():
    print(f"Starting training for missing GAT-LSTM models on {len(INSTANCES)} instances...")
    print(f"Instances: {INSTANCES}")
    
    start_time = time.time()
    results = {}

    for inst in INSTANCES:
        success = train_missing_model(inst)
        results[inst] = "SUCCESS" if success else "FAILED"

    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total time: {elapsed/60:.2f} minutes")
    for k, v in results.items():
        print(f"{k:<10}: {v}")
    print("="*60)

if __name__ == "__main__":
    main()

```

## `train_optw_baseline.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/train_optw_baseline.py`
- **Size**: 12061 bytes
- **Last modified**: 2025-11-19 20:07:32

```python
import os
import logging
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tnrange, tqdm
import torch
from torch import optim

import src.config as cf
# CHANGED: Use fixed utils
import src.utils_transformer as u
import src.train_utils as tu
import src.sampling_norm_utils as snu
from src.neural_net import RecPointerNetwork
# from src.hybrid_neural_net import HybridPointerNetwork
from src.solution_construction import RunEpisode

# for logging
N_DASHES = 40


def train_loop(inp_val, inp_real, raw_data, run_episode, args):

    raw_data, raw_dist_mat = inp_real[0][1][0], inp_real[0][1][2]
    reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0
    training_history = []
    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=args.lr)
    step_dict = {}

    for epoch in tqdm(range(1,args.nepocs+1)):

        avg_reward, min_reward, max_reward, loss = tu.train_model(raw_data,
                                                                  raw_dist_mat,
                                                                  norm_dic,
                                                                  run_episode,
                                                                  model_opt,
                                                                  args)

        reward_total += avg_reward
        min_reward_total += min_reward
        max_reward_total += max_reward
        loss_total += loss

        tu.exp_lr_scheduler(model_opt, epoch, init_lr=args.lr)


        if epoch==0 or epoch % args.nprint == 0:
            logger.info("Epoch %s" % str(epoch))
            rew_dict, avg_reward_val = tu.validation(inp_val, run_episode, norm_dic, args.device)
            _, avg_reward_real  = tu.validation(inp_real, run_episode, norm_dic, args.device)
            step_dict[epoch] = rew_dict

            if epoch == 0:
                avg_loss = loss_total
                avg_reward_total = reward_total
                avg_min_reward_total = min_reward_total
                avg_max_reward_total = max_reward_total

                training_history.append([epoch, reward_total, min_reward_total, max_reward_total,
                                         avg_reward_val, avg_reward_real, loss_total])

            else:
                avg_loss = loss_total / args.nprint
                avg_reward_total = reward_total / args.nprint
                avg_min_reward_total = min_reward_total / args.nprint
                avg_max_reward_total = max_reward_total / args.nprint


                training_history.append([epoch, avg_reward_total, avg_min_reward_total, avg_max_reward_total,
                                         avg_reward_val, avg_reward_real, avg_loss])


            logger.info(N_DASHES*'-')
            logger.info("Average total loss: %s" % avg_loss)
            logger.info("Average train mean reward: %s" % avg_reward_total)
            logger.info("Average train max reward: %s" % avg_max_reward_total)
            logger.info("Average train min reward: %s" % avg_min_reward_total)
            logger.info("Validation reward: %2.3f"  % (avg_reward_val))
            logger.info("Real instance reward: %2.3f"  % (avg_reward_real))

            reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0

        if epoch % args.nsave == 0 and not args.debug:
            print('saving model')
            torch.save(run_episode.neuralnet.state_dict(), args.save_w_dir+'/model_'+str(epoch)+'.pkl')

    return training_history



def setup_args_parser():

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--instance', help='which instance to train on')
    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--use_checkpoint', help='use checkpoint (see https://pytorch.org/docs/stable/checkpoint.html)', action='store_true')
    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: \n \
                                uniformly sampled (uni_samp), \
                                score proportional to each point of interest\'s duration of visit (corr_samp)' ,
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')
    parser.add_argument('--model_name', help='model name', default='default', type=str)
    parser.add_argument('--model_type', help='type of architecture to use', default='original', choices=['original', 'hybrid'])
    parser.add_argument('--n_gat_layers', help='number of GAT layers for the hybrid model', default=1, type=int)
    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--nsave', help='saves the model weights every <nsave> epochs', default=10000, type=int)
    parser.add_argument('--nprint', help='to log and save the training history \
                                          (total score in the benchmark and generated \
                                          instances of the validation set) every <nprint> epochs', default=2500, type=int)
    parser.add_argument('--nepocs', help='number of training epochs', default=100000, type=int)
    parser.add_argument('--batch_size', help='training batch size', default=32, type=int)
    parser.add_argument('--max_grad_norm', help='maximum norm value for gradient value clipping', default=1, type=int)
    parser.add_argument('--lr', help='initial learning rate', default=1e-4, type=float)
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    parser.add_argument('--beta', help='entropy term coefficient', default=0.01, type=float)
    parser.add_argument('--rnn_hidden', help='hidden size of RNN', default=128, type=int)
    parser.add_argument('--n_layers', help='number of attention layers in the encoder', default=2, type=int)
    parser.add_argument('--n_heads', help='number heads in attention layers', default=8, type=int)
    parser.add_argument('--ff_dim', help='hidden dimension of the encoder\'s feedforward sublayer', default=256, type=int)
    parser.add_argument('--nfeatures', help='number of non-dynamic features', default=7, type=int)
    parser.add_argument('--ndfeatures', help='number of dynamic features', default=8, type=int)

    return parser

def parse_args_further(args):

    SAVE_W_DIR_STRING = '{results_path}/{benchmark_instance}/model_w/model_{model_name}_{sample_type}'
    SAVE_H_DIR_STRING = '{results_path}/{benchmark_instance}/outputs/model_{model_name}_{sample_type}'
    GENERATED_DIR_STRING = '{generated_path}/{benchmark_instance}'

    args.save_h_file = 'training_history.csv'

    GENERATED_PT_FILE = 'inp_val_{sample_type}.pt'
    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    args.instance_type = u.get_instance_type(args.instance)
    args.map_location =  {'cpu': args.device_name}
    args.save_w_dir = SAVE_W_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)
    args.save_h_dir = SAVE_H_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)

    args.val_dir = GENERATED_DIR_STRING.format(generated_path=cf.GENERATED_INSTANCES_PATH,
                                               benchmark_instance = args.instance)
    args.val_set_pt_file = GENERATED_PT_FILE.format(sample_type=args.sample_type)


    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    return args


def log_args(args):
    logger.info(N_DASHES*'-')
    logger.info('Running train_model_main.py')
    logger.info(N_DASHES*'-')
    logger.info('model name:  %s' % args.model_name)
    logger.info(N_DASHES*'-')
    logger.info('device: %s' % args.device_name)
    logger.info('instance: %s' % args.instance)
    logger.info('instance type: %s' % args.instance_type)
    logger.info('use_checkpoint: %s' % args.use_checkpoint)
    logger.info('sample type: %s' % args.sample_type)
    logger.info('debug mode: %s' % args.debug)
    logger.info('nsave: %s' % args.nsave)
    logger.info('nprint: %s' % args.nprint)
    logger.info('nepocs: %s' % args.nepocs)
    logger.info('seed: %s' % args.seed)
    logger.info(N_DASHES*'-')
    logger.info('batch_size: %s' % args.batch_size)
    logger.info('max_grad_norm: %s' % args.max_grad_norm)
    logger.info('learning rate (lr): %s' % args.lr)
    logger.info('entropy term coefficient (beta): %s' % args.beta)
    logger.info('hidden size of RNN (hidden): %s' % args.rnn_hidden)
    logger.info('number of attention layers in the Encoder: %s' % args.n_layers)
    logger.info('number of features: %s' % args.nfeatures)
    logger.info('number of dynamic features: %s' % args.ndfeatures)
    logger.info(N_DASHES*'-')
    logger.info(args.instance)


def save_training_history(training_history, file_path, args):
    TRAIN_HIST_COLUMNS = ['epoch', 'avg_reward_train',
        'min_reward_train', 'max_reward_train', 'avg_reward_val_'+args.sample_type,
        'avg_reward_real', 'tloss_train']

    df = pd.DataFrame(training_history, columns = TRAIN_HIST_COLUMNS)
    df.to_csv(file_path, index=False)
    return


def save_args(args):
    keys_list = ['instance', 'model_type', 'n_layers', 'n_gat_layers', 'n_heads', 'ff_dim', 'nfeatures',
                 'ndfeatures', 'rnn_hidden', 'sample_type', 'lr', 'batch_size',
                 'seed', 'beta', 'max_grad_norm', 'nsave', 'nepocs']

    dict_to_save = { key: args.__dict__[key] for key in keys_list }
    FILE_NAME = '/model_'+args.model_name+'_training_args.txt'

    with open(args.save_w_dir+FILE_NAME, 'w') as f:
        json.dump(dict_to_save, f, indent=2)

    return


if __name__ == "__main__":

    # parse arguments and setup logger
    parser = setup_args_parser()
    args_temp = parser.parse_args()
    args = parse_args_further(args_temp)

    logger = u.setup_logger(args.debug)
    log_args(args)

    # for reproducibility"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(args.seed)

    # create directories for saving
    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    # get val and real instance scores
    inp_real = u.get_real_data(args, phase='train')
    inp_val = u.get_val_data(args, phase='train')

    # get Tmax and Smax
    raw_data = inp_real[0][1][0]
    Tmax, Smax = snu.instance_dependent_norm_const(raw_data)
    norm_dic = {args.instance: {'Tmax': Tmax, 'Smax': Smax}}

    # save args to file
    if not args.debug:
        save_args(args)

    # train
    # if args.model_type == 'hybrid':
    #     logger.info(f"Using HYBRID model with {args.n_gat_layers} GAT layer(s).")
    #     model = HybridPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device)
    # else: # original
    logger.info("Using ORIGINAL Transformer model.")
    model = RecPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device)

    run_episode = RunEpisode(model, args)

    training_history = train_loop(inp_val, inp_real, norm_dic, run_episode, args)

    # save
    if not args.debug:
        file_path = '%s/%s' % (args.save_h_dir, args.save_h_file)
        save_training_history(training_history, file_path, args)

```

## `train_optw_gat_lstm.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/train_optw_gat_lstm.py`
- **Size**: 11518 bytes
- **Last modified**: 2025-11-26 21:12:22

```python
import os
import logging
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tnrange, tqdm
import torch
from torch import optim

import src.config as cf
import src.utils_transformer as u
import src.train_utils as tu
import src.sampling_norm_utils as snu
from src.neural_net_gat_lstm import GATLSTMPointerNetwork
from src.solution_construction import RunEpisode

# for logging
N_DASHES = 40


def train_loop(inp_val, inp_real, raw_data, run_episode, args):

    raw_data, raw_dist_mat = inp_real[0][1][0], inp_real[0][1][2]
    reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0
    training_history = []
    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=args.lr)
    step_dict = {}

    for epoch in tqdm(range(1,args.nepocs+1)):

        avg_reward, min_reward, max_reward, loss = tu.train_model(raw_data,
                                                                  raw_dist_mat,
                                                                  norm_dic,
                                                                  run_episode,
                                                                  model_opt,
                                                                  args)

        reward_total += avg_reward
        min_reward_total += min_reward
        max_reward_total += max_reward
        loss_total += loss

        tu.exp_lr_scheduler(model_opt, epoch, init_lr=args.lr)


        if epoch==0 or epoch % args.nprint == 0:
            logger.info("Epoch %s" % str(epoch))
            rew_dict, avg_reward_val = tu.validation(inp_val, run_episode, norm_dic, args.device)
            _, avg_reward_real  = tu.validation(inp_real, run_episode, norm_dic, args.device)
            step_dict[epoch] = rew_dict

            if epoch == 0:
                avg_loss = loss_total
                avg_reward_total = reward_total
                avg_min_reward_total = min_reward_total
                avg_max_reward_total = max_reward_total

                training_history.append([epoch, reward_total, min_reward_total, max_reward_total,
                                         avg_reward_val, avg_reward_real, loss_total])

            else:
                avg_loss = loss_total / args.nprint
                avg_reward_total = reward_total / args.nprint
                avg_min_reward_total = min_reward_total / args.nprint
                avg_max_reward_total = max_reward_total / args.nprint


                training_history.append([epoch, avg_reward_total, avg_min_reward_total, avg_max_reward_total,
                                         avg_reward_val, avg_reward_real, avg_loss])


            logger.info(N_DASHES*'-')
            logger.info("Average total loss: %s" % avg_loss)
            logger.info("Average train mean reward: %s" % avg_reward_total)
            logger.info("Average train max reward: %s" % avg_max_reward_total)
            logger.info("Average train min reward: %s" % avg_min_reward_total)
            logger.info("Validation reward: %2.3f"  % (avg_reward_val))
            logger.info("Real instance reward: %2.3f"  % (avg_reward_real))

            reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0

        if epoch % args.nsave == 0 and not args.debug:
            print('saving model')
            torch.save(run_episode.neuralnet.state_dict(), args.save_w_dir+'/model_'+str(epoch)+'.pkl')

    return training_history



def setup_args_parser():

    parser = argparse.ArgumentParser(description='train GAT-LSTM model')
    parser.add_argument('--instance', help='which instance to train on')
    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--use_checkpoint', help='use checkpoint (see https://pytorch.org/docs/stable/checkpoint.html)', action='store_true')
    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: uniformly sampled (uni_samp), score proportional to each point of interest duration (corr_samp)',
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')
    parser.add_argument('--model_name', help='model name', default='gat_lstm', type=str)
    parser.add_argument('--n_gat_layers', help='number of GAT layers', default=1, type=int)
    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--nsave', help='saves the model weights every nsave epochs', default=100, type=int)
    parser.add_argument('--nprint', help='to log and save the training history every nprint epochs', default=100, type=int)
    parser.add_argument('--nepocs', help='number of training epochs', default=10000, type=int)
    parser.add_argument('--batch_size', help='training batch size', default=32, type=int)
    parser.add_argument('--max_grad_norm', help='maximum norm value for gradient value clipping', default=1, type=int)
    parser.add_argument('--lr', help='initial learning rate', default=1e-4, type=float)
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    parser.add_argument('--beta', help='entropy term coefficient', default=0.01, type=float)
    parser.add_argument('--rnn_hidden', help='hidden size of RNN', default=128, type=int)
    parser.add_argument('--n_layers', help='number of attention layers in the encoder', default=2, type=int)
    parser.add_argument('--n_heads', help='number heads in attention layers', default=8, type=int)
    parser.add_argument('--ff_dim', help='hidden dimension of the encoder feedforward sublayer', default=256, type=int)
    parser.add_argument('--nfeatures', help='number of non-dynamic features', default=7, type=int)
    parser.add_argument('--ndfeatures', help='number of dynamic features', default=8, type=int)

    return parser

def parse_args_further(args):

    SAVE_W_DIR_STRING = '{results_path}/{benchmark_instance}/model_w/model_{model_name}_{sample_type}'
    SAVE_H_DIR_STRING = '{results_path}/{benchmark_instance}/outputs/model_{model_name}_{sample_type}'
    GENERATED_DIR_STRING = '{generated_path}/{benchmark_instance}'

    args.save_h_file = 'training_history.csv'

    GENERATED_PT_FILE = 'inp_val_{sample_type}.pt'
    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    args.instance_type = u.get_instance_type(args.instance)
    args.map_location =  {'cpu': args.device_name}
    args.save_w_dir = SAVE_W_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)
    args.save_h_dir = SAVE_H_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)

    args.val_dir = GENERATED_DIR_STRING.format(generated_path=cf.GENERATED_INSTANCES_PATH,
                                               benchmark_instance = args.instance)
    args.val_set_pt_file = GENERATED_PT_FILE.format(sample_type=args.sample_type)


    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    return args


def log_args(args):
    logger.info(N_DASHES*'-')
    logger.info('Running train_optw_gat_lstm.py')
    logger.info(N_DASHES*'-')
    logger.info('model name:  %s' % args.model_name)
    logger.info(N_DASHES*'-')
    logger.info('device: %s' % args.device_name)
    logger.info('instance: %s' % args.instance)
    logger.info('instance type: %s' % args.instance_type)
    logger.info('use_checkpoint: %s' % args.use_checkpoint)
    logger.info('sample type: %s' % args.sample_type)
    logger.info('n_gat_layers: %s' % args.n_gat_layers)
    logger.info('debug mode: %s' % args.debug)
    logger.info('nsave: %s' % args.nsave)
    logger.info('nprint: %s' % args.nprint)
    logger.info('nepocs: %s' % args.nepocs)
    logger.info('seed: %s' % args.seed)
    logger.info(N_DASHES*'-')
    logger.info('batch_size: %s' % args.batch_size)
    logger.info('max_grad_norm: %s' % args.max_grad_norm)
    logger.info('learning rate (lr): %s' % args.lr)
    logger.info('entropy term coefficient (beta): %s' % args.beta)
    logger.info('hidden size of RNN (hidden): %s' % args.rnn_hidden)
    logger.info('number of attention layers in the Encoder: %s' % args.n_layers)
    logger.info('number of features: %s' % args.nfeatures)
    logger.info('number of dynamic features: %s' % args.ndfeatures)
    logger.info(N_DASHES*'-')
    logger.info(args.instance)


def save_training_history(training_history, file_path, args):
    TRAIN_HIST_COLUMNS = ['epoch', 'avg_reward_train',
        'min_reward_train', 'max_reward_train', 'avg_reward_val_'+args.sample_type,
        'avg_reward_real', 'tloss_train']

    df = pd.DataFrame(training_history, columns = TRAIN_HIST_COLUMNS)
    df.to_csv(file_path, index=False)
    return


def save_args(args):
    keys_list = ['instance', 'model_type', 'n_layers', 'n_gat_layers', 'n_heads', 'ff_dim', 'nfeatures',
                 'ndfeatures', 'rnn_hidden', 'sample_type', 'lr', 'batch_size',
                 'seed', 'beta', 'max_grad_norm', 'nsave', 'nepocs']

    dict_to_save = { key: args.__dict__.get(key, None) for key in keys_list }
    dict_to_save['model_type'] = 'gat_lstm'  # Hard-code model type
    FILE_NAME = '/model_'+args.model_name+'_training_args.txt'

    with open(args.save_w_dir+FILE_NAME, 'w') as f:
        json.dump(dict_to_save, f, indent=2)

    return


if __name__ == "__main__":

    # parse arguments and setup logger
    parser = setup_args_parser()
    args_temp = parser.parse_args()
    args = parse_args_further(args_temp)

    logger = u.setup_logger(args.debug)
    log_args(args)

    # for reproducibility"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(args.seed)

    # create directories for saving
    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    # get val and real instance scores
    inp_real = u.get_real_data(args, phase='train')
    inp_val = u.get_val_data(args, phase='train')

    # get Tmax and Smax
    raw_data = inp_real[0][1][0]
    Tmax, Smax = snu.instance_dependent_norm_const(raw_data)
    norm_dic = {args.instance: {'Tmax': Tmax, 'Smax': Smax}}

    # save args to file
    if not args.debug:
        save_args(args)

    # train GAT-LSTM model
    logger.info(f"Using GAT-LSTM model with {args.n_gat_layers} GAT layer(s).")
    model = GATLSTMPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device)

    run_episode = RunEpisode(model, args)

    training_history = train_loop(inp_val, inp_real, norm_dic, run_episode, args)

    # save
    if not args.debug:
        file_path = '%s/%s' % (args.save_h_dir, args.save_h_file)
        save_training_history(training_history, file_path, args)

```

## `train_optw_gat_transformer.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/train_optw_gat_transformer.py`
- **Size**: 11461 bytes
- **Last modified**: 2025-11-26 21:26:23

```python
import os
import logging
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tnrange, tqdm
import torch
from torch import optim

import src.config as cf
import src.utils_transformer as u
import src.train_utils as tu
import src.sampling_norm_utils as snu
from src.neural_net_gat_transformer import GATTransformerPointerNetwork
from src.solution_construction import RunEpisode

# for logging
N_DASHES = 40


def train_loop(inp_val, inp_real, raw_data, run_episode, args):

    raw_data, raw_dist_mat = inp_real[0][1][0], inp_real[0][1][2]
    reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0
    training_history = []
    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=args.lr)
    step_dict = {}

    for epoch in tqdm(range(1,args.nepocs+1)):

        avg_reward, min_reward, max_reward, loss = tu.train_model(raw_data,
                                                                  raw_dist_mat,
                                                                  norm_dic,
                                                                  run_episode,
                                                                  model_opt,
                                                                  args)

        reward_total += avg_reward
        min_reward_total += min_reward
        max_reward_total += max_reward
        loss_total += loss

        tu.exp_lr_scheduler(model_opt, epoch, init_lr=args.lr)


        if epoch==0 or epoch % args.nprint == 0:
            logger.info("Epoch %s" % str(epoch))
            rew_dict, avg_reward_val = tu.validation(inp_val, run_episode, norm_dic, args.device)
            _, avg_reward_real  = tu.validation(inp_real, run_episode, norm_dic, args.device)
            step_dict[epoch] = rew_dict

            if epoch == 0:
                avg_loss = loss_total
                avg_reward_total = reward_total
                avg_min_reward_total = min_reward_total
                avg_max_reward_total = max_reward_total

                training_history.append([epoch, reward_total, min_reward_total, max_reward_total,
                                         avg_reward_val, avg_reward_real, loss_total])

            else:
                avg_loss = loss_total / args.nprint
                avg_reward_total = reward_total / args.nprint
                avg_min_reward_total = min_reward_total / args.nprint
                avg_max_reward_total = max_reward_total / args.nprint


                training_history.append([epoch, avg_reward_total, avg_min_reward_total, avg_max_reward_total,
                                         avg_reward_val, avg_reward_real, avg_loss])


            logger.info(N_DASHES*'-')
            logger.info("Average total loss: %s" % avg_loss)
            logger.info("Average train mean reward: %s" % avg_reward_total)
            logger.info("Average train max reward: %s" % avg_max_reward_total)
            logger.info("Average train min reward: %s" % avg_min_reward_total)
            logger.info("Validation reward: %2.3f"  % (avg_reward_val))
            logger.info("Real instance reward: %2.3f"  % (avg_reward_real))

            reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0

        if epoch % args.nsave == 0 and not args.debug:
            print('saving model')
            torch.save(run_episode.neuralnet.state_dict(), args.save_w_dir+'/model_'+str(epoch)+'.pkl')

    return training_history



def setup_args_parser():

    parser = argparse.ArgumentParser(description='train GAT-Transformer model')
    parser.add_argument('--instance', help='which instance to train on')
    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--use_checkpoint', help='use checkpoint', action='store_true')
    parser.add_argument('--sample_type', help='how to sample scores: uniformly sampled (uni_samp) or score proportional (corr_samp)',
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')
    parser.add_argument('--model_name', help='model name', default='gat_transformer', type=str)
    parser.add_argument('--n_gat_layers', help='number of GAT layers', default=1, type=int)
    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--nsave', help='saves the model weights every nsave epochs', default=100, type=int)
    parser.add_argument('--nprint', help='to log and save the training history every nprint epochs', default=100, type=int)
    parser.add_argument('--nepocs', help='number of training epochs', default=10000, type=int)
    parser.add_argument('--batch_size', help='training batch size', default=32, type=int)
    parser.add_argument('--max_grad_norm', help='maximum norm value for gradient value clipping', default=1, type=int)
    parser.add_argument('--lr', help='initial learning rate', default=1e-4, type=float)
    parser.add_argument('--seed', help='seed random generators (for reproducibility)', default=2925, type=int)
    parser.add_argument('--beta', help='entropy term coefficient', default=0.01, type=float)
    parser.add_argument('--rnn_hidden', help='hidden size of RNN', default=128, type=int)
    parser.add_argument('--n_layers', help='number of attention layers in the encoder', default=2, type=int)
    parser.add_argument('--n_heads', help='number heads in attention layers', default=8, type=int)
    parser.add_argument('--ff_dim', help='hidden dimension of the encoder feedforward sublayer', default=256, type=int)
    parser.add_argument('--nfeatures', help='number of non-dynamic features', default=7, type=int)
    parser.add_argument('--ndfeatures', help='number of dynamic features', default=8, type=int)

    return parser

def parse_args_further(args):

    SAVE_W_DIR_STRING = '{results_path}/{benchmark_instance}/model_w/model_{model_name}_{sample_type}'
    SAVE_H_DIR_STRING = '{results_path}/{benchmark_instance}/outputs/model_{model_name}_{sample_type}'
    GENERATED_DIR_STRING = '{generated_path}/{benchmark_instance}'

    args.save_h_file = 'training_history.csv'

    GENERATED_PT_FILE = 'inp_val_{sample_type}.pt'
    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    args.instance_type = u.get_instance_type(args.instance)
    args.map_location =  {'cpu': args.device_name}
    args.save_w_dir = SAVE_W_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)
    args.save_h_dir = SAVE_H_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)

    args.val_dir = GENERATED_DIR_STRING.format(generated_path=cf.GENERATED_INSTANCES_PATH,
                                               benchmark_instance = args.instance)
    args.val_set_pt_file = GENERATED_PT_FILE.format(sample_type=args.sample_type)


    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    return args


def log_args(args):
    logger.info(N_DASHES*'-')
    logger.info('Running train_optw_gat_transformer.py')
    logger.info(N_DASHES*'-')
    logger.info('model name:  %s' % args.model_name)
    logger.info(N_DASHES*'-')
    logger.info('device: %s' % args.device_name)
    logger.info('instance: %s' % args.instance)
    logger.info('instance type: %s' % args.instance_type)
    logger.info('use_checkpoint: %s' % args.use_checkpoint)
    logger.info('sample type: %s' % args.sample_type)
    logger.info('n_gat_layers: %s' % args.n_gat_layers)
    logger.info('debug mode: %s' % args.debug)
    logger.info('nsave: %s' % args.nsave)
    logger.info('nprint: %s' % args.nprint)
    logger.info('nepocs: %s' % args.nepocs)
    logger.info('seed: %s' % args.seed)
    logger.info(N_DASHES*'-')
    logger.info('batch_size: %s' % args.batch_size)
    logger.info('max_grad_norm: %s' % args.max_grad_norm)
    logger.info('learning rate (lr): %s' % args.lr)
    logger.info('entropy term coefficient (beta): %s' % args.beta)
    logger.info('hidden size of RNN (hidden): %s' % args.rnn_hidden)
    logger.info('number of attention layers in the Encoder: %s' % args.n_layers)
    logger.info('number of features: %s' % args.nfeatures)
    logger.info('number of dynamic features: %s' % args.ndfeatures)
    logger.info(N_DASHES*'-')
    logger.info(args.instance)


def save_training_history(training_history, file_path, args):
    TRAIN_HIST_COLUMNS = ['epoch', 'avg_reward_train',
        'min_reward_train', 'max_reward_train', 'avg_reward_val_'+args.sample_type,
        'avg_reward_real', 'tloss_train']

    df = pd.DataFrame(training_history, columns = TRAIN_HIST_COLUMNS)
    df.to_csv(file_path, index=False)
    return


def save_args(args):
    keys_list = ['instance', 'model_type', 'n_layers', 'n_gat_layers', 'n_heads', 'ff_dim', 'nfeatures',
                 'ndfeatures', 'rnn_hidden', 'sample_type', 'lr', 'batch_size',
                 'seed', 'beta', 'max_grad_norm', 'nsave', 'nepocs']

    dict_to_save = { key: args.__dict__.get(key, None) for key in keys_list }
    dict_to_save['model_type'] = 'gat_transformer'  # Hard-code model type
    FILE_NAME = '/model_'+args.model_name+'_training_args.txt'

    with open(args.save_w_dir+FILE_NAME, 'w') as f:
        json.dump(dict_to_save, f, indent=2)

    return


if __name__ == "__main__":

    # parse arguments and setup logger
    parser = setup_args_parser()
    args_temp = parser.parse_args()
    args = parse_args_further(args_temp)

    logger = u.setup_logger(args.debug)
    log_args(args)

    # for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(args.seed)

    # create directories for saving
    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    # get val and real instance scores
    inp_real = u.get_real_data(args, phase='train')
    inp_val = u.get_val_data(args, phase='train')

    # get Tmax and Smax
    raw_data = inp_real[0][1][0]
    Tmax, Smax = snu.instance_dependent_norm_const(raw_data)
    norm_dic = {args.instance: {'Tmax': Tmax, 'Smax': Smax}}

    # save args to file
    if not args.debug:
        save_args(args)

    # train GAT-Transformer model
    logger.info(f"Using GAT-Transformer model with {args.n_gat_layers} GAT layer(s).")
    model = GATTransformerPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device)

    run_episode = RunEpisode(model, args)

    training_history = train_loop(inp_val, inp_real, norm_dic, run_episode, args)

    # save
    if not args.debug:
        file_path = '%s/%s' % (args.save_h_dir, args.save_h_file)
        save_training_history(training_history, file_path, args)

```

## `train_optw_rl.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/train_optw_rl.py`
- **Size**: 11993 bytes
- **Last modified**: 2025-11-15 09:25:46

```python
import os
import logging
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tnrange, tqdm
import torch
from torch import optim

import src.config as cf
import src.utils as u
import src.train_utils as tu
import src.sampling_norm_utils as snu
from src.neural_net import RecPointerNetwork
from src.hybrid_neural_net import HybridPointerNetwork
from src.solution_construction import RunEpisode

# for logging
N_DASHES = 40


def train_loop(inp_val, inp_real, raw_data, run_episode, args):

    raw_data, raw_dist_mat = inp_real[0][1][0], inp_real[0][1][2]
    reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0
    training_history = []
    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=args.lr)
    step_dict = {}

    for epoch in tqdm(range(1,args.nepocs+1)):

        avg_reward, min_reward, max_reward, loss = tu.train_model(raw_data,
                                                                  raw_dist_mat,
                                                                  norm_dic,
                                                                  run_episode,
                                                                  model_opt,
                                                                  args)

        reward_total += avg_reward
        min_reward_total += min_reward
        max_reward_total += max_reward
        loss_total += loss

        tu.exp_lr_scheduler(model_opt, epoch, init_lr=args.lr)


        if epoch==0 or epoch % args.nprint == 0:
            logger.info("Epoch %s" % str(epoch))
            rew_dict, avg_reward_val = tu.validation(inp_val, run_episode, norm_dic, args.device)
            _, avg_reward_real  = tu.validation(inp_real, run_episode, norm_dic, args.device)
            step_dict[epoch] = rew_dict

            if epoch == 0:
                avg_loss = loss_total
                avg_reward_total = reward_total
                avg_min_reward_total = min_reward_total
                avg_max_reward_total = max_reward_total

                training_history.append([epoch, reward_total, min_reward_total, max_reward_total,
                                         avg_reward_val, avg_reward_real, loss_total])

            else:
                avg_loss = loss_total / args.nprint
                avg_reward_total = reward_total / args.nprint
                avg_min_reward_total = min_reward_total / args.nprint
                avg_max_reward_total = max_reward_total / args.nprint


                training_history.append([epoch, avg_reward_total, avg_min_reward_total, avg_max_reward_total,
                                         avg_reward_val, avg_reward_real, avg_loss])


            logger.info(N_DASHES*'-')
            logger.info("Average total loss: %s" % avg_loss)
            logger.info("Average train mean reward: %s" % avg_reward_total)
            logger.info("Average train max reward: %s" % avg_max_reward_total)
            logger.info("Average train min reward: %s" % avg_min_reward_total)
            logger.info("Validation reward: %2.3f"  % (avg_reward_val))
            logger.info("Real instance reward: %2.3f"  % (avg_reward_real))

            reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0

        if epoch % args.nsave == 0 and not args.debug:
            print('saving model')
            torch.save(run_episode.neuralnet.state_dict(), args.save_w_dir+'/model_'+str(epoch)+'.pkl')

    return training_history



def setup_args_parser():

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--instance', help='which instance to train on')
    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--use_checkpoint', help='use checkpoint (see https://pytorch.org/docs/stable/checkpoint.html)', action='store_true')
    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: \n \
                                uniformly sampled (uni_samp), \
                                score proportional to each point of interest\'s duration of visit (corr_samp)' ,
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')
    parser.add_argument('--model_name', help='model name', default='default', type=str)
    parser.add_argument('--model_type', help='type of architecture to use', default='original', choices=['original', 'hybrid'])
    parser.add_argument('--n_gat_layers', help='number of GAT layers for the hybrid model', default=1, type=int)
    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--nsave', help='saves the model weights every <nsave> epochs', default=10000, type=int)
    parser.add_argument('--nprint', help='to log and save the training history \
                                          (total score in the benchmark and generated \
                                          instances of the validation set) every <nprint> epochs', default=2500, type=int)
    parser.add_argument('--nepocs', help='number of training epochs', default=100000, type=int)
    parser.add_argument('--batch_size', help='training batch size', default=32, type=int)
    parser.add_argument('--max_grad_norm', help='maximum norm value for gradient value clipping', default=1, type=int)
    parser.add_argument('--lr', help='initial learning rate', default=1e-4, type=float)
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    parser.add_argument('--beta', help='entropy term coefficient', default=0.01, type=float)
    parser.add_argument('--rnn_hidden', help='hidden size of RNN', default=128, type=int)
    parser.add_argument('--n_layers', help='number of attention layers in the encoder', default=2, type=int)
    parser.add_argument('--n_heads', help='number heads in attention layers', default=8, type=int)
    parser.add_argument('--ff_dim', help='hidden dimension of the encoder\'s feedforward sublayer', default=256, type=int)
    parser.add_argument('--nfeatures', help='number of non-dynamic features', default=7, type=int)
    parser.add_argument('--ndfeatures', help='number of dynamic features', default=8, type=int)

    return parser

def parse_args_further(args):

    SAVE_W_DIR_STRING = '{results_path}/{benchmark_instance}/model_w/model_{model_name}_{sample_type}'
    SAVE_H_DIR_STRING = '{results_path}/{benchmark_instance}/outputs/model_{model_name}_{sample_type}'
    GENERATED_DIR_STRING = '{generated_path}/{benchmark_instance}'

    args.save_h_file = 'training_history.csv'

    GENERATED_PT_FILE = 'inp_val_{sample_type}.pt'
    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    args.instance_type = u.get_instance_type(args.instance)
    args.map_location =  {'cpu': args.device_name}
    args.save_w_dir = SAVE_W_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)
    args.save_h_dir = SAVE_H_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)

    args.val_dir = GENERATED_DIR_STRING.format(generated_path=cf.GENERATED_INSTANCES_PATH,
                                               benchmark_instance = args.instance)
    args.val_set_pt_file = GENERATED_PT_FILE.format(sample_type=args.sample_type)


    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    return args


def log_args(args):
    logger.info(N_DASHES*'-')
    logger.info('Running train_model_main.py')
    logger.info(N_DASHES*'-')
    logger.info('model name:  %s' % args.model_name)
    logger.info(N_DASHES*'-')
    logger.info('device: %s' % args.device_name)
    logger.info('instance: %s' % args.instance)
    logger.info('instance type: %s' % args.instance_type)
    logger.info('use_checkpoint: %s' % args.use_checkpoint)
    logger.info('sample type: %s' % args.sample_type)
    logger.info('debug mode: %s' % args.debug)
    logger.info('nsave: %s' % args.nsave)
    logger.info('nprint: %s' % args.nprint)
    logger.info('nepocs: %s' % args.nepocs)
    logger.info('seed: %s' % args.seed)
    logger.info(N_DASHES*'-')
    logger.info('batch_size: %s' % args.batch_size)
    logger.info('max_grad_norm: %s' % args.max_grad_norm)
    logger.info('learning rate (lr): %s' % args.lr)
    logger.info('entropy term coefficient (beta): %s' % args.beta)
    logger.info('hidden size of RNN (hidden): %s' % args.rnn_hidden)
    logger.info('number of attention layers in the Encoder: %s' % args.n_layers)
    logger.info('number of features: %s' % args.nfeatures)
    logger.info('number of dynamic features: %s' % args.ndfeatures)
    logger.info(N_DASHES*'-')
    logger.info(args.instance)


def save_training_history(training_history, file_path, args):
    TRAIN_HIST_COLUMNS = ['epoch', 'avg_reward_train',
        'min_reward_train', 'max_reward_train', 'avg_reward_val_'+args.sample_type,
        'avg_reward_real', 'tloss_train']

    df = pd.DataFrame(training_history, columns = TRAIN_HIST_COLUMNS)
    df.to_csv(file_path, index=False)
    return


def save_args(args):
    keys_list = ['instance', 'model_type', 'n_layers', 'n_gat_layers', 'n_heads', 'ff_dim', 'nfeatures',
                 'ndfeatures', 'rnn_hidden', 'sample_type', 'lr', 'batch_size',
                 'seed', 'beta', 'max_grad_norm', 'nsave', 'nepocs']

    dict_to_save = { key: args.__dict__[key] for key in keys_list }
    FILE_NAME = '/model_'+args.model_name+'_training_args.txt'

    with open(args.save_w_dir+FILE_NAME, 'w') as f:
        json.dump(dict_to_save, f, indent=2)

    return


if __name__ == "__main__":

    # parse arguments and setup logger
    parser = setup_args_parser()
    args_temp = parser.parse_args()
    args = parse_args_further(args_temp)

    logger = u.setup_logger(args.debug)
    log_args(args)

    # for reproducibility"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(args.seed)

    # create directories for saving
    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    # get val and real instance scores
    inp_real = u.get_real_data(args, phase='train')
    inp_val = u.get_val_data(args, phase='train')

    # get Tmax and Smax
    raw_data = inp_real[0][1][0]
    Tmax, Smax = snu.instance_dependent_norm_const(raw_data)
    norm_dic = {args.instance: {'Tmax': Tmax, 'Smax': Smax}}

    # save args to file
    save_args(args)

    # train
    if args.model_type == 'hybrid':
        logger.info(f"Using HYBRID model with {args.n_gat_layers} GAT layer(s).")
        model = HybridPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device)
    else: # original
        logger.info("Using ORIGINAL Transformer model.")
        model = RecPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device)

    run_episode = RunEpisode(model, args)

    training_history = train_loop(inp_val, inp_real, norm_dic, run_episode, args)

    # save
    if not args.debug:
        file_path = '%s/%s' % (args.save_h_dir, args.save_h_file)
        save_training_history(training_history, file_path, args)

```

## `train_optw_transformer.py`

- **Path**: `/home/huyngo/Project/ML/optw_rl/train_optw_transformer.py`
- **Size**: 11904 bytes
- **Last modified**: 2025-11-24 23:05:56

```python
import os
import logging
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tnrange, tqdm
import torch
from torch import optim

import src.config as cf
import src.utils_transformer as u
import src.train_utils as tu
import src.sampling_norm_utils as snu
# CHANGED: Import from new file
from src.neural_net_transformer import TransformerPointerNetwork
from src.solution_construction import RunEpisode

# for logging
N_DASHES = 40


def train_loop(inp_val, inp_real, raw_data, run_episode, args):

    raw_data, raw_dist_mat = inp_real[0][1][0], inp_real[0][1][2]
    reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0
    training_history = []
    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=args.lr)
    step_dict = {}

    for epoch in tqdm(range(1,args.nepocs+1)):

        avg_reward, min_reward, max_reward, loss = tu.train_model(raw_data,
                                                                  raw_dist_mat,
                                                                  norm_dic,
                                                                  run_episode,
                                                                  model_opt,
                                                                  args)

        reward_total += avg_reward
        min_reward_total += min_reward
        max_reward_total += max_reward
        loss_total += loss

        tu.exp_lr_scheduler(model_opt, epoch, init_lr=args.lr)


        if epoch==0 or epoch % args.nprint == 0:
            logger.info("Epoch %s" % str(epoch))
            rew_dict, avg_reward_val = tu.validation(inp_val, run_episode, norm_dic, args.device)
            _, avg_reward_real  = tu.validation(inp_real, run_episode, norm_dic, args.device)
            step_dict[epoch] = rew_dict

            if epoch == 0:
                avg_loss = loss_total
                avg_reward_total = reward_total
                avg_min_reward_total = min_reward_total
                avg_max_reward_total = max_reward_total

                training_history.append([epoch, reward_total, min_reward_total, max_reward_total,
                                         avg_reward_val, avg_reward_real, loss_total])

            else:
                avg_loss = loss_total / args.nprint
                avg_reward_total = reward_total / args.nprint
                avg_min_reward_total = min_reward_total / args.nprint
                avg_max_reward_total = max_reward_total / args.nprint


                training_history.append([epoch, avg_reward_total, avg_min_reward_total, avg_max_reward_total,
                                         avg_reward_val, avg_reward_real, avg_loss])


            logger.info(N_DASHES*'-')
            logger.info("Average total loss: %s" % avg_loss)
            logger.info("Average train mean reward: %s" % avg_reward_total)
            logger.info("Average train max reward: %s" % avg_max_reward_total)
            logger.info("Average train min reward: %s" % avg_min_reward_total)
            logger.info("Validation reward: %2.3f"  % (avg_reward_val))
            logger.info("Real instance reward: %2.3f"  % (avg_reward_real))

            reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0

        if epoch % args.nsave == 0 and not args.debug:
            print('saving model')
            torch.save(run_episode.neuralnet.state_dict(), args.save_w_dir+'/model_'+str(epoch)+'.pkl')

    return training_history



def setup_args_parser():

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--instance', help='which instance to train on')
    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--use_checkpoint', help='use checkpoint (see https://pytorch.org/docs/stable/checkpoint.html)', action='store_true')
    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: \n \
                                uniformly sampled (uni_samp), \
                                score proportional to each point of interest\'s duration of visit (corr_samp)' ,
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')
    parser.add_argument('--model_name', help='model name', default='default', type=str)
    # Removed 'model_type' choice since we force transformer here, or we can keep it for logging
    parser.add_argument('--model_type', help='type of architecture to use', default='transformer', choices=['original', 'hybrid', 'transformer'])
    parser.add_argument('--n_gat_layers', help='number of GAT layers for the hybrid model', default=1, type=int)
    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--nsave', help='saves the model weights every <nsave> epochs', default=100, type=int)
    parser.add_argument('--nprint', help='to log and save the training history \
                                          (total score in the benchmark and generated \
                                          instances of the validation set) every <nprint> epochs', default=2500, type=int)
    parser.add_argument('--nepocs', help='number of training epochs', default=50000, type=int)
    parser.add_argument('--batch_size', help='training batch size', default=32, type=int)
    parser.add_argument('--max_grad_norm', help='maximum norm value for gradient value clipping', default=1, type=int)
    parser.add_argument('--lr', help='initial learning rate', default=1e-4, type=float)
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    parser.add_argument('--beta', help='entropy term coefficient', default=0.01, type=float)
    parser.add_argument('--rnn_hidden', help='hidden size of RNN', default=128, type=int)
    parser.add_argument('--n_layers', help='number of attention layers in the encoder', default=2, type=int)
    parser.add_argument('--n_heads', help='number heads in attention layers', default=8, type=int)
    parser.add_argument('--ff_dim', help='hidden dimension of the encoder\'s feedforward sublayer', default=256, type=int)
    parser.add_argument('--nfeatures', help='number of non-dynamic features', default=7, type=int)
    parser.add_argument('--ndfeatures', help='number of dynamic features', default=8, type=int)

    return parser

def parse_args_further(args):

    SAVE_W_DIR_STRING = '{results_path}/{benchmark_instance}/model_w/model_{model_name}_{sample_type}'
    SAVE_H_DIR_STRING = '{results_path}/{benchmark_instance}/outputs/model_{model_name}_{sample_type}'
    GENERATED_DIR_STRING = '{generated_path}/{benchmark_instance}'

    args.save_h_file = 'training_history.csv'

    GENERATED_PT_FILE = 'inp_val_{sample_type}.pt'
    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    args.instance_type = u.get_instance_type(args.instance)
    args.map_location =  {'cpu': args.device_name}
    args.save_w_dir = SAVE_W_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)
    args.save_h_dir = SAVE_H_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)

    args.val_dir = GENERATED_DIR_STRING.format(generated_path=cf.GENERATED_INSTANCES_PATH,
                                               benchmark_instance = args.instance)
    args.val_set_pt_file = GENERATED_PT_FILE.format(sample_type=args.sample_type)


    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    return args


def log_args(args):
    logger.info(N_DASHES*'-')
    logger.info('Running train_optw_transformer.py')
    logger.info(N_DASHES*'-')
    logger.info('model name:  %s' % args.model_name)
    logger.info(N_DASHES*'-')
    logger.info('device: %s' % args.device_name)
    logger.info('instance: %s' % args.instance)
    logger.info('instance type: %s' % args.instance_type)
    logger.info('use_checkpoint: %s' % args.use_checkpoint)
    logger.info('sample type: %s' % args.sample_type)
    logger.info('debug mode: %s' % args.debug)
    logger.info('nsave: %s' % args.nsave)
    logger.info('nprint: %s' % args.nprint)
    logger.info('nepocs: %s' % args.nepocs)
    logger.info('seed: %s' % args.seed)
    logger.info(N_DASHES*'-')
    logger.info('batch_size: %s' % args.batch_size)
    logger.info('max_grad_norm: %s' % args.max_grad_norm)
    logger.info('learning rate (lr): %s' % args.lr)
    logger.info('entropy term coefficient (beta): %s' % args.beta)
    logger.info('hidden size of RNN (hidden): %s' % args.rnn_hidden)
    logger.info('number of attention layers in the Encoder: %s' % args.n_layers)
    logger.info('number of features: %s' % args.nfeatures)
    logger.info('number of dynamic features: %s' % args.ndfeatures)
    logger.info(N_DASHES*'-')
    logger.info(args.instance)


def save_training_history(training_history, file_path, args):
    TRAIN_HIST_COLUMNS = ['epoch', 'avg_reward_train',
        'min_reward_train', 'max_reward_train', 'avg_reward_val_'+args.sample_type,
        'avg_reward_real', 'tloss_train']

    df = pd.DataFrame(training_history, columns = TRAIN_HIST_COLUMNS)
    df.to_csv(file_path, index=False)
    return


def save_args(args):
    keys_list = ['instance', 'model_type', 'n_layers', 'n_gat_layers', 'n_heads', 'ff_dim', 'nfeatures',
                 'ndfeatures', 'rnn_hidden', 'sample_type', 'lr', 'batch_size',
                 'seed', 'beta', 'max_grad_norm', 'nsave', 'nepocs']

    dict_to_save = { key: args.__dict__[key] for key in keys_list }
    FILE_NAME = '/model_'+args.model_name+'_training_args.txt'

    with open(args.save_w_dir+FILE_NAME, 'w') as f:
        json.dump(dict_to_save, f, indent=2)

    return


if __name__ == "__main__":

    # parse arguments and setup logger
    parser = setup_args_parser()
    args_temp = parser.parse_args()
    args = parse_args_further(args_temp)

    logger = u.setup_logger(args.debug)
    log_args(args)

    # for reproducibility"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(args.seed)

    # create directories for saving
    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    # get val and real instance scores
    inp_real = u.get_real_data(args, phase='train')
    inp_val = u.get_val_data(args, phase='train')

    # get Tmax and Smax
    raw_data = inp_real[0][1][0]
    Tmax, Smax = snu.instance_dependent_norm_const(raw_data)
    norm_dic = {args.instance: {'Tmax': Tmax, 'Smax': Smax}}

    # save args to file
    if not args.debug:
        save_args(args)

    # train
    logger.info("Using NEW Transformer Pointer Network.")
    model = TransformerPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device)

    run_episode = RunEpisode(model, args)

    training_history = train_loop(inp_val, inp_real, norm_dic, run_episode, args)

    # save
    if not args.debug:
        file_path = '%s/%s' % (args.save_h_dir, args.save_h_file)
        save_training_history(training_history, file_path, args)

```

