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
