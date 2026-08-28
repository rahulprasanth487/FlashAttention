# FlashAttention-2 (FA2) Explained: Restructuring Loops, Reducing Non-Matmul FLOPs, and Warp Layouts

While FlashAttention-1 dramatically reduced memory reads/writes, it still operated at only 20-40% of the GPU's theoretical maximum compute capacity. FlashAttention-2 (released in July 2023 by Tri Dao) restructured the algorithm to address GPU register pressure, thread synchronization, and instruction overhead.

---

## 1. Why Was FlashAttention-1 Still Sub-Optimal?

Standard GPUs have two types of compute units:
1. **Tensor Cores**: High-speed matrix-multiplication units (e.g., handles $Q @ K^T$ and $A @ V$).
2. **CUDA Cores (ALUs)**: Standard processing units that handle all non-matmul operations (exponential, divisions, scalar additions, and scaling).

In FA1, although HBM access was minimized, the CUDA cores were bottlenecked by:
* **Excessive Rescaling**: At every step in the inner loop, the running output block $O_i$ was scaled and divided by the running row sum $l_i$ to maintain mathematical alignment.
* **Shared Memory Overhead**: In the warp-level work allocation, warps inside a thread block had to write intermediate results to shared memory and synchronize via barriers (`__syncthreads()`) constantly.
* **Low Occupancy**: Under low batch sizes or small head counts, many SMs (Streaming Multiprocessors) sat idle.

---

## 2. Key Enhancements in FlashAttention-2

FA2 introduces four primary upgrades to push performance closer to theoretical hardware ceilings (often achieving 50-73% of peak GPU throughput):

```
+-----------------------------------------------------------------------------------------+
|                               FLASHATTENTION-2 OPTIMIZATIONS                            |
+-----------------------------------------------------------------------------------------+
|  1. FLIPPED LOOP ORDER        |  Outer loop: Rows of Q. Inner loop: Columns of K/V.      |
|                               |  Write output block to HBM only ONCE at the very end.   |
+-------------------------------+---------------------------------------------------------+
|  2. FEWER NON-MATMUL FLOPs    |  Store un-normalized accumulator (O * l) in registers.  |
|                               |  Perform single division at the end of the inner loop.  |
+-------------------------------+---------------------------------------------------------+
|  3. WARP-LEVEL LAYOUT         |  Split Q rows across warps; share K and V.             |
|                               |  Eliminates warp synchronization and shared memory writes. |
+-------------------------------+---------------------------------------------------------+
|  4. SPLIT-KV PARALLELISM      |  Parallelize across sequence length (Split-KV) when     |
|                               |  batch/head count is low, maximizing GPU utilization.   |
+-----------------------------------------------------------------------------------------+
```

### A. Flipped Loop Order (Forward Pass)
In FlashAttention-1, the outer loop iterated over columns of $K, V$ ($T_c$) and the inner loop iterated over rows of $Q$ ($T_r$). This forced the algorithm to read/write the running output $O_i$, running max $m_i$, and running sum $l_i$ to global memory repeatedly.

FA2 flips this:
* **Outer Loop**: Iterates over blocks of $Q$ ($T_r$).
* **Inner Loop**: Iterates over blocks of $K, V$ ($T_c$).
* **Advantage**: A block of $Q$ is loaded into SRAM registers, compared against all blocks of $K, V$ in the inner loop, and the final rescaled output block is written to HBM **only once at the very end** of the inner loop. This minimizes HBM write traffic.

### B. Fewer Non-Matmul FLOPs (Delayed Normalization)
In FA1, the update step for the output block was:
\[
O_i^{\text{new}} = \frac{l_i \cdot e^{m_i - m_i^{\text{new}}} \cdot O_i + e^{\tilde{m} - m_i^{\text{new}}} \cdot \tilde{P}_{ij} V_j}{l_i^{\text{new}}}
\]
Notice the division by $l_i^{\text{new}}$ happens inside the inner loop at *every single iteration*.

FA2 simplifies this by storing the **un-normalized output accumulator** $S_i = O_i \cdot l_i$ in registers. 
The update step becomes a simple multiply-add:
\[
S_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \cdot S_i + e^{\tilde{m} - m_i^{\text{new}}} \cdot \tilde{P}_{ij} V_j
\]
\[
l_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \cdot l_i + e^{\tilde{m} - m_i^{\text{new}}} \cdot \tilde{l}_{ij}
\]
After the inner loop terminates (all blocks of $K, V$ have been processed), we perform a **single division** to normalize:
\[
O_i = \frac{S_i}{l_i}
\]
This completely removes division operations from the hot path loop.

### C. Warp-Level Work Partitioning
A GPU Streaming Multiprocessor (SM) executes instructions in warps (groups of 32 threads).
* **FA1 Warp Layout**: The thread block loads $Q$ to shared memory. The warps partition $K$ and $V$. Different warps compute partial matrix multiplications and must write to shared memory and wait for each other to aggregate their results.
* **FA2 Warp Layout**: Each thread block splits $Q$ among the warps (each warp gets a unique slice of rows), while $K$ and $V$ are shared by all warps. Each warp can perform its GEMM independently and write its slice of the output directly to HBM without communicating with other warps, eliminating shared memory read/write barriers.

### D. Split-KV Parallelism (Long-Context Inference)
In LLM generation (decoding), the batch size is small, and sequence length is long. Traditional attention parallelizes across batch size $\times$ number of heads. If this product is small (e.g. batch size 1, 16 heads), only 16 SMs are used, leaving 90% of a large GPU idle.

FA2 introduces **Split-KV**:
1. Split the keys/values sequence length into multiple segments.
2. Assign different segments to different GPU SMs.
3. Each SM computes partial attention outputs and scaling statistics ($m, l$).
4. A final reduction kernel combines these partial outputs to form the final result.
This guarantees high GPU occupancy even during single-sequence generation.
