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
