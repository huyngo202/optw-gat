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
