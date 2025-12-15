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
