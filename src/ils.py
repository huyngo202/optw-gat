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
    def __init__(self, instance_name, max_iter=1000, time_limit=None, df=None, dist_mat=None):
        self.instance_name = instance_name
        self.max_iter = max_iter
        self.time_limit = time_limit
        
        # Load Data
        if df is not None and dist_mat is not None:
             self.df = df
             self.dist_mat = dist_mat
             self.instance_type = get_instance_type(instance_name) # Still needed for some logic? or assume standard
        else:
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
