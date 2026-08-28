# Empirical Analysis of Attention Benchmark Results (Kaggle/T4 Run)

This document provides a detailed mathematical and architectural analysis of the benchmark results obtained from running the FlashAttention-1 suite on a Kaggle Nvidia T4 GPU (FP16 precision).

---

## 1. Memory Scaling Analysis: Quadratic vs. Linear

### Standard Attention ($O(N^2)$ Memory)
Let's analyze the peak memory allocated by Standard Attention at different sequence lengths:
* **$N = 128$**: 12.15 MB
* **$N = 512$**: 48.15 MB
* **$N = 1024$**: 152.15 MB
* **$N = 2048$**: 552.15 MB

### Native FlashAttention ($O(N)$ Memory)
Let's look at the memory allocated by Native FlashAttention:
* **$N = 128$**: 10.15 MB
* **$N = 512$**: 16.15 MB
* **$N = 1024$**: 24.15 MB
* **$N = 2048$**: 40.15 MB

### Mathematical Scaling Proof
To isolate the memory overhead caused strictly by the intermediate attention matrix, we subtract the linear footprint of FlashAttention (which contains only inputs, outputs, and running vectors) from Standard Attention:
* **Overhead at $N = 128$**: $12.15\text{ MB} - 10.15\text{ MB} = 2.00\text{ MB}$
* **Overhead at $N = 512$**: $48.15\text{ MB} - 16.15\text{ MB} = 32.00\text{ MB}$
* **Overhead at $N = 1024$**: $152.15\text{ MB} - 24.15\text{ MB} = 128.00\text{ MB}$
* **Overhead at $N = 2048$**: $552.15\text{ MB} - 40.15\text{ MB} = 512.00\text{ MB}$

Let's check the scaling factors:
* When sequence length increases by $4\times$ ($128 \rightarrow 512$), the memory overhead increases by:
  \[
  \frac{32.00\text{ MB}}{2.00\text{ MB}} = 16\times \quad (\text{Since } 4^2 = 16)
  \]
* When sequence length increases by $2\times$ ($512 \rightarrow 1024$), the memory overhead increases by:
  \[
  \frac{128.00\text{ MB}}{32.00\text{ MB}} = 4\times \quad (\text{Since } 2^2 = 4)
  \]
* When sequence length increases by $2\times$ ($1024 \rightarrow 2048$), the memory overhead increases by:
  \[
  \frac{512.00\text{ MB}}{128.00\text{ MB}} = 4\times \quad (\text{Since } 2^2 = 4)
  \]

This empirical data is a **perfect mathematical proof** of the $O(N^2)$ quadratic scaling of Standard Attention's memory versus the $O(N)$ linear scaling of FlashAttention. At $N=2048$, FlashAttention saves you **512 MB of GPU VRAM** for a tiny batch size (4) and head count (8).

---

## 2. Runtime Speedup Analysis

Let's evaluate the speedup ratio ($\text{Time}_{\text{Standard}} / \text{Time}_{\text{Native}}$) as $N$ scales:
* **$N = 128$**: $\frac{0.123\text{ ms}}{0.096\text{ ms}} = 1.28\times$ speedup
* **$N = 512$**: $\frac{0.694\text{ ms}}{0.443\text{ ms}} = 1.57\times$ speedup
* **$N = 1024$**: $\frac{2.419\text{ ms}}{1.341\text{ ms}} = 1.80\times$ speedup
* **$N = 2048$**: $\frac{9.457\text{ ms}}{3.341\text{ ms}} = 2.83\times$ speedup

### Architectural Inference:
1. **The Fused Kernel Advantage**:
   As the sequence length grows, Standard Attention's runtime increases quadratically ($O(N^2)$ arithmetic and memory operations). FlashAttention's runtime scales much slower.
2. **Bandwidth Bound Crossover**:
   At small sequence lengths ($N=128$), memory latency is dominated by kernel launch overhead. As $N$ grows ($N=2048$), HBM read/write saturation bottlenecks standard attention. Native FlashAttention fusions keep data inside the fast SRAM cache, widening the speedup gap to **2.83x** at $N=2048$.

---

## 3. Python Simulation: Correctness vs. Interpreter Bottleneck

### The Memory Success:
The `FA1 Simulation` peak memory is virtually identical to `Native FlashAttn`:
* **At $N=1024$**: Simulation uses **24.43 MB** vs. Native's **24.15 MB**.
This verifies that the sequential block-loading logic successfully prevents allocating the $N \times N$ matrix.

### The Compute Slowdown:
Despite the memory success, the Python simulation is extremely slow:
* **At $N=1024$**: Standard Attention takes **2.419 ms**, while the Python simulation takes **2,817.969 ms** (~1,165x slower).

### Architectural Inference:
This discrepancy demonstrates why we cannot deploy Python-based tiling in production. 
In Python, each block multiplication, maximum check, exponential, and sum is launched as an independent PyTorch operation. This results in **thousands of sequential GPU kernel launches and CPU-GPU synchronization steps** (overhead). 
Production FlashAttention must be written in **CUDA/Triton** as a single fused kernel, compiling the entire block loop into a single GPU program that runs in parallel.
