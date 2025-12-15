# Beam Search Inference Report

## Overview
This report analyzes the performance of different inference strategies on the OPTW benchmark instances. We compare three approaches:
- **Greedy Decoding**: Fast, deterministic selection of highest-probability actions
- **Beam Search (BS)**: Explores multiple solution candidates (beam size = 128)
- **ILS Heuristic**: Baseline comparison from previous analysis

Models tested: **Baseline (LSTM)** and **Transformer**

## Results Summary

### Greedy Decoding Performance

| Instance | Baseline | Transformer | Time (ms) - Base | Time (ms) - Trans |
| :--- | :--- | :--- | :--- | :--- |
| **c101** | 310 | **320** | 22 | 37 |
| **r101** | 192 | **197** | 23 | 27 |
| **rc101** | 200 | **207** | 20 | 39 |
| **pr01** | 241 | **250** | 28 | 35 |
| **t101** | 287 | **304** | 53 | 61 |

### Beam Search Performance (Beam Size = 128)

| Instance | Baseline | Transformer | Time (ms) - Base | Time (ms) - Trans |
| :--- | :--- | :--- | :--- | :--- |
| **c101** | **320** | **320** | 1463 | 1950 |
| **r101** | **198** | **198** | 1462 | 1736 |
| **rc101** | **219** | 216 | 1447 | 2139 |
| **pr01** | **299** | 278 | 1108 | 1750 |
| **t101** | 318 | **332** | 5447 | 7923 |

### Comparison: Greedy vs Beam Search

| Instance | Model | Greedy | Beam Search | Improvement | Speedup (Greedy/BS) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **c101** | Baseline | 310 | 320 | +10 (+3.2%) | 66x |
| | Transformer | 320 | 320 | +0 (0%) | 53x |
| **r101** | Baseline | 192 | 198 | +6 (+3.1%) | 64x |
| | Transformer | 197 | 198 | +1 (+0.5%) | 64x |
| **rc101** | Baseline | 200 | 219 | +19 (+9.5%) | 72x |
| | Transformer | 207 | 216 | +9 (+4.3%) | 55x |
| **pr01** | Baseline | 241 | 299 | +58 (+24.1%) | 40x |
| | Transformer | 250 | 278 | +28 (+11.2%) | 50x |
| **t101** | Baseline | 287 | 318 | +31 (+10.8%) | 103x |
| | Transformer | 304 | 332 | +28 (+9.2%) | 130x |

### Cross-Method Comparison

| Instance | Greedy (Best) | Beam Search (Best) | ILS | Best Method |
| :--- | :--- | :--- | :--- | :--- |
| **c101** | 320 (T) | 320 (B/T) | **320** | All Tied |
| **r101** | 197 (T) | 198 (B/T) | **182** | ILS |
| **rc101** | 207 (T) | 219 (B) | **219** | BS/ILS Tied |
| **pr01** | 250 (T) | 299 (B) | **273** | Beam Search |
| **t101** | 304 (T) | 332 (T) | 344 | ILS |

## Key Observations

### Beam Search Effectiveness
*   **Significant Gains on pr01 & t101**: Beam Search provides 10-24% improvement over Greedy on these instances, demonstrating strong exploration benefits.
*   **Moderate Gains on rc101**: ~9% improvement for Baseline, ~4% for Transformer.
*   **Minimal Gains on c101 & r101**: Beam Search offers marginal or no improvement (<= 3%), suggesting Greedy already finds near-optimal paths for these topologies.

### Baseline vs Transformer
*   **Greedy**: Transformer consistently outperforms Baseline by 3-6% across all instances.
*   **Beam Search**: Results are mixed. Transformer wins on c101, r101, and t101, while Baseline dominates on pr01 and rc101. This suggests the Transformer's advantage diminishes with extensive search.

### Computational Cost
*   Beam Search is **40-130x slower** than Greedy, with inference time ranging from 1-8 seconds vs 20-60 milliseconds.
*   The Transformer is ~30-50% slower than the Baseline for both Greedy and Beam Search.

### ILS vs RL Methods
*   **c101**: All methods achieve the same score (320), showing convergence.
*   **r101 & rc101**: ILS finds the best solutions, outperforming even Beam Search.
*   **pr01**: Beam Search (Baseline) achieves 299, outperforming ILS (273) by 9.5%.
*   **t101**: ILS (344) significantly outperforms all RL methods.

## Recommendations

1. **For Real-Time Applications**: Use **Greedy Decoding with Transformer** for fastest inference (~30-60ms) with competitive quality.
2. **For Quality-Critical Scenarios**: Use **Beam Search (beam size 128)** on instances like pr01, rc101, and t101 where it provides 10-24% gains.
3. **Hybrid Approach**: For instances like c101 and r101, Greedy is sufficient; reserve Beam Search for harder instances (pr01, t101).
4. **Future Work**: 
   - Investigate why Beam Search has minimal effect on c101/r101.
   - Optimize Beam Search implementation to reduce 40-130x overhead.
   - Explore adaptive beam sizing based on instance characteristics.

## Conclusion

Beam Search significantly improves solution quality on certain instances (pr01, t101, rc101) at the cost of 40-130x longer inference time. The Transformer model shows promise in Greedy mode but does not consistently leverage Beam Search better than the Baseline. For production deployment, a hybrid strategy using Greedy for most instances and Beam Search for specific problem types offers the best quality-speed trade-off.
