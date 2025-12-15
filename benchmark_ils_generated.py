
import torch
import pandas as pd
import numpy as np
import time
import src.config as cf
from src.ils import OPTW_ILS
import os

def load_generated_data(instance):
    # Path based on src/config.py and standard naming
    gen_path = cf.GENERATED_INSTANCES_PATH
    val_dir = os.path.join(gen_path, instance)
    pt_file = "inp_val_uni_samp.pt" 
    path = os.path.join(val_dir, pt_file)
    if not os.path.exists(path):
        return []
    return torch.load(path)

def tensor_to_df(raw_data, instance):
    # raw_data: (N, 7) tensor
    # df cols needed: i, x, y, duration, ti, tf, prof, Total Time
    data = raw_data.numpy()
    N = data.shape[0]
    ids = np.arange(N).reshape(-1, 1)
    
    # data columns mapping based on utils.get_instance_data:
    # raw_data = df_inst[['x', 'y', 'duration', 'ti', 'tf', 'prof', 'Total Time']].values
    
    df = pd.DataFrame(data, columns=['x', 'y', 'duration', 'ti', 'tf', 'prof', 'Total Time'])
    df.insert(0, 'i', ids)
    df['inst_name'] = instance
    df['real_or_val'] = 'val'
    return df

def run_ils_generated():
    INSTANCES = ['c101', 'r101', 'rc101', 'pr01', 't101']
    results = []
    
    print(f"{'Instance':<10} {'Avg Profit':<15} {'Avg Time':<15}")
    print("-" * 45)
    
    for inst in INSTANCES:
        try:
            inp_val = load_generated_data(inst)
            if not inp_val:
                print(f"No data for {inst}")
                continue
                
            total_profit = 0
            total_time = 0
            count = 0
            
            # Run on subset (first 5) for speed
            SUBSET_SIZE = 5 
            
            for item in inp_val:
                if len(item) == 2:
                    raw_data, raw_distm = item
                elif len(item) == 3:
                     # likely (data, start_time, dist)
                     raw_data, _, raw_distm = item
                else:
                    print(f"Unexpected item length: {len(item)}")
                    continue

                df = tensor_to_df(raw_data, inst)
                dist_m = raw_distm.numpy()
                
                # ILS Configuration
                # Limit time to 30s per instance as requested
                ils = OPTW_ILS(inst, max_iter=20000, time_limit=30, df=df, dist_mat=dist_m)
                
                start = time.time()
                profit, route = ils.solve()
                elapsed = time.time() - start
                
                total_profit += profit
                total_time += elapsed
                count += 1
                
                if count >= SUBSET_SIZE:
                    break
            
            if count > 0:
                avg_profit = total_profit / count
                avg_time = total_time / count
                print(f"{inst:<10} {avg_profit:<15.2f} {avg_time:<15.2f}")
                results.append({
                    'Instance': inst, 
                    'ILS_Gen_Reward': avg_profit,
                    'ILS_Gen_Time': avg_time
                })
            
        except Exception as e:
            print(f"Error {inst}: {e}")
            import traceback
            traceback.print_exc()

    pd.DataFrame(results).to_csv('results/ils_generated_results.csv', index=False)

if __name__ == '__main__':
    run_ils_generated()
