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
