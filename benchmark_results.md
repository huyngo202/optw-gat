# Benchmark Results: Transformer vs Baseline

## Overview
We compared the performance of the new **Transformer-based** architecture against the **LSTM Baseline** on a representative subset of 5 instances.
- **Training Duration**: 1000 Epochs per model per instance.
- **Device**: CPU (Intel i5 1135g7).
- **Total Time**: ~78 minutes.

## Results Summary

| Instance | Baseline Max Reward | Transformer Max Reward | Gap % | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **c101** | 252.77 | **253.41** | +0.25% | Transformer |
| **pr01** | 175.84 | **178.33** | +1.41% | Transformer |
| **r101** | **109.77** | 109.14 | -0.57% | Baseline |
| **rc101** | 146.61 | **146.91** | +0.20% | Transformer |
| **t101** | **747.91** | 746.77 | -0.15% | Baseline |

## Analysis
- **Performance**: The Transformer model is performing **comparably** to the Baseline, with slight improvements in 3 out of 5 instances even with limited training (1000 epochs).
- **Convergence**: Both models showed signs of learning, but 1000 epochs is likely too short for full convergence. The Transformer might require more epochs to fully outperform the LSTM due to its complexity.
- **Stability**: The training was stable on CPU.

## Conclusion
The Transformer implementation is functional and competitive. To see significant improvements, we recommend:
1.  Training for longer (e.g., 10,000+ epochs).
2.  Tuning hyperparameters (learning rate, attention heads) specifically for the Transformer.
