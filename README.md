# Deep Reinforcement Learning for OPTW with GAT-Transformer

This repository implements a **Graph Attention Network (GAT)** based approach for solving the **Orienteering Problem with Time Windows (OPTW)**, a complex combinatorial optimization problem relevant to Tourist Trip Design Problems (TTDP). The implementation relies on Deep Reinforcement Learning (DRL) using the REINFORCE algorithm with a baseline.

## Key Features

- **Hybrid Architecture:** Combines Graph Attention Networks (GAT) for local feature extraction with a Recursive Transformer Encoder for global sequence modeling.
- **Dynamic Feature Engineering:** Robust handling of time-dependent constraints via dynamic node embedding updates.
- **Advanced Inference:** Supports Greedy Search and Beam Search (k=10/128) decoding strategies.
- **Comparative Benchmarking:** Includes scripts to benchmark against Iterated Local Search (ILS) and classical Pointer Networks.
- **Robust Data Generation:** Tools to generate realistic tourist instances (Simulated User Profiles) for validation.

## Architecture Overview

The model framework consists of:
1.  **GAT Encoder:** Extracts topological features from the graph of available points.
2.  **Recursive Transformer:** Captures long-range dependencies in the route sequence.
3.  **Decoder with Pointing Mechanism:** Sequentially selects the next point to visit, satisfying all time window constraints via masking.

## Setup Instructions

### Prerequisites
- Python 3.7+
- Anaconda or Miniconda

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/huyngo202/optw-gat.git
    cd optw-gat
    ```

2.  Create and activate the environment:
    ```bash
    conda env create --file environment.yml
    conda activate optw_env
    ```

## Usage

### 1. Data Generation
Generate validation datasets (Simulated User Profiles) to robustly evaluate the model:
```bash
python generate_instances.py --instance c101 --sample_type uni_samp
```

### 2. Training
Train the GAT-Transformer model on specific instances (e.g., `c101`):
```bash
# Train GAT-Transformer with REINFORCE
python train_optw_gat_transformer.py --instance c101 --nepocs 5000 --model_name gat_trans_demo
```

### 3. Inference & Benchmarking
Run inference using the trained model with Beam Search:

```bash
# Run Beam Search inference on Generated Instances
python benchmark_beam_generated.py
```

Benchmark against Iterated Local Search (ILS):
```bash
# Run ILS benchmark (Metaheuristic baseline)
python benchmark_ils_generated.py
```

## Results Summary

Experiments on Solomon (`c101`, `r101`, `rc101`) and Cordeau (`pr01`) benchmarks demonstrate that:
-   **GAT-Transformer (Beam Search)** achieves valid solutions in sub-second time (<1s), making it ~140x faster than ILS (30s).
-   **Performance:**
    -   Competitive with Metaheuristics on Random/Mixed instances (`r101`).
    -   Outperforms Metaheuristics on large-scale instances (`pr01`) in terms of efficiency.
    -   Slightly lower reward on strictly Clustered instances (`c101`) compared to intensive search methods, but significantly faster.

## Directory Structure
-   `src/`: Core implementation (Models, Logic, Utils).
-   `data/`: Benchmark and Generated datasets.
-   `results/`: Training logs and model checkpoints.
-   `report/`: LaTeX source for the thesis report.

## References
This work builds upon the Pointer Network architecture and the OPTW formulation by [Gama & Fernandes (2021)](https://www.sciencedirect.com/science/article/pii/S0305054821001349).
