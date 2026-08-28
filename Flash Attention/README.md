# FlashAttention-1 (FA1) — TL;DR

FA1 demonstrates how tiling and online softmax let attention be computed without materializing the full N×N attention matrix. That reduces peak memory from O(N^2) to O(N) and avoids repeated HBM read/write cycles.

## What this folder contains
- `benchmark_fa1.py`: small scripts comparing naive attention vs FA1 simulations.
- `flash_attention_benchmark.ipynb`: interactive notebook illustrating tiling and online softmax.
- `analysis_results.md`: mathematical notes and memory-scaling proofs.

## How to use
1. Open `flash_attention_benchmark.ipynb` in Colab/Kaggle or locally with a GPU runtime.
2. Run cells sequentially to see memory/latency tables and visualizations.

## Where this is useful
- Educational walkthroughs to understand the core idea of online softmax and tiling.
- Baseline comparisons when implementing or testing new attention kernels.
