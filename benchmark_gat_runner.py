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
        models_to_run = ['gat_lstm', 'gat_transformer'overleaf_session2=s%3ANJdjoCLIzH4PIogp0ruE7aeJ6wdE68DZ.G6K6h6rjCdzwvd7ZzHetI9CRGyKfHF6CgOI2W62o5f8; GCLB=CPiBoLOP6PLXswEQAw]
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