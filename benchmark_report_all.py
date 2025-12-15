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