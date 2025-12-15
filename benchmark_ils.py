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
