import matplotlib.pyplot as plt
import os

# Configuration
DATA_DIR = 'data/benchmark'
OUTPUT_DIR = 'images/benchmark_plots'
INSTANCES = {
    'c101': 'c101.txt',
    'r101': 'r101.txt',
    'rc101': 'rc101.txt',
    'pr01': 'pr01.txt'
}

def parse_instance(filepath):
    """
    Parses an OPTW instance file to extract coordinates.
    Assumes the format: ID X Y ...
    Returns:
        depot: (x, y) tuple
        customers: list of (x, y) tuples
    """
    customers = []
    depot = None
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                try:
                    # Check if the first column is an ID
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    
                    if node_id == 0:
                        depot = (x, y)
                    else:
                        customers.append((x, y))
                except ValueError:
                    # Skip header lines that don't match the format
                    continue
                    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

    return depot, customers

def plot_instance(name, depot, customers):
    """
    Plots the instance nodes.
    """
    if depot is None:
        print(f"No depot found for {name}")
        return

    plt.figure(figsize=(6, 6))
    
    # Plot customers
    cust_x = [c[0] for c in customers]
    cust_y = [c[1] for c in customers]
    plt.scatter(cust_x, cust_y, c='blue', marker='o', s=30, label='Participants', alpha=0.6)
    
    # Plot depot
    plt.scatter(depot[0], depot[1], c='red', marker='s', s=100, label='Depot')
    
    plt.title(f'Instance: {name}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_path = os.path.join(OUTPUT_DIR, f'{name}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for name, filename in INSTANCES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print(f"Processing {name}...")
        depot, customers = parse_instance(filepath)
        
        if depot:
            plot_instance(name, depot, customers)
        else:
            print(f"Failed to parse {name}")

if __name__ == "__main__":
    main()
