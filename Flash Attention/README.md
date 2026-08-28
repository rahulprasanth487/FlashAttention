# FlashAttention-1 (FA1) Explained: Tiling, Online Softmax, and Memory-Efficiency

Standard attention is highly inefficient because it is bottlenecked by GPU memory read/write speeds (**I/O bottleneck**) rather than mathematical calculation speeds. This document explains the core principles, mathematical mechanics, and an intuitive analogy to help you understand FlashAttention-1.

---

## 1. The Core Bottleneck: Memory Hierarchy (HBM vs. SRAM)

To understand FlashAttention, we must first look at the physical architecture of a GPU:

```
+-------------------------------------------------------------+
|                     High Bandwidth Memory (HBM)              |
|                     Size: Large (16GB - 80GB)               |
|                     Speed: Slow (~1.5 - 3.0 TB/s)           |
+-------------------------------------------------------------+
                              ^
                              |  (Slow read/write transfer)
                              v
+-------------------------------------------------------------+
|                     On-Chip SRAM Cache                      |
|                     Size: Tiny (~20MB)                      |
|                     Speed: Extremely Fast (~19 TB/s)        |
+-------------------------------------------------------------+
                              ^
                              |  (Direct access)
                              v
+-------------------------------------------------------------+
|                     Compute Cores (ALUs)                    |
|                     Speed: Hundreds of TFLOPs               |
+-------------------------------------------------------------+
```

* **HBM (High Bandwidth Memory)**: Large memory (e.g., 40GB/80GB on an A100 GPU) but has relatively slow transfer bandwidth (~1.5-2.0 TB/s).
* **SRAM (Static Random Access Memory)**: Fast on-chip cache (typically ~20MB) with extremely high bandwidth (~19 TB/s). The compute engines (Tensor Cores) can only operate directly on data loaded into SRAM.

### Standard Attention's Memory Problem
In standard self-attention:
1. Read \(Q\) and \(K\) from **HBM** $\rightarrow$ load to **SRAM** $\rightarrow$ compute $S = QK^T / \sqrt{d_k}$.
2. Write the giant $N \times N$ matrix $S$ from **SRAM** back to **HBM** (Memory Write).
3. Read $S$ from **HBM** back to **SRAM** $\rightarrow$ compute attention weights $A = \text{softmax}(S)$.
4. Write the giant $N \times N$ matrix $A$ from **SRAM** back to **HBM** (Memory Write).
5. Read $A$ and $V$ from **HBM** to **SRAM** $\rightarrow$ compute $O = AV$.
6. Write final output $O$ back to **HBM**.

Because $N$ (sequence length) can be large (e.g., 2048, 8192), reading and writing the intermediate $N \times N$ attention matrix to HBM multiple times takes **much longer** than the actual matrix multiplications. Standard attention is **memory-bound, not compute-bound**.

---

## 2. The Intuitive Analogy: The Library vs. The Desk

Let's conceptualize this using a student writing a research summary:

| Concept | Hardware Component | Analogy Element | Description |
| :--- | :--- | :--- | :--- |
| **HBM** | High Bandwidth Memory | **Main Library Archive** | Contains thousands of books (large storage), but it is a 10-minute walk away. |
| **SRAM** | On-Chip Cache | **Your Study Desk** | Right in front of you. You can read/write instantly, but it can only fit a few pages at a time. |
| **ALU** | Compute Cores | **Your Brain** | Performs the actual reading, comparing, and summarizing. |

### The Standard Attention Student
The student wants to write a summary of how 1,000 different pages (sequence length $N = 1000$) relate to each other:
1. The student walks to the **Library Archive**, reads page 1 and page 2, walks back to their **Desk**, compares them, and writes a temporary draft on a giant sheet of paper.
2. The desk is too small to hold the giant sheet, so they walk all the way back to the **Library Archive** to store this draft.
3. They repeat this walking back-and-forth for *every single pair* of pages (doing this $1000 \times 1000 = 1,000,000$ times).
4. By the end, the student has spent **99% of their time walking back and forth** (HBM bandwidth latency) and only 1% actually thinking (computation).

### The FlashAttention Student (Tiling)
The student realizes they can work smarter:
1. They bring a small stack of pages (say, 64 pages) from the archive to their **Desk**.
2. They keep a small running summary sheet on the corner of the desk.
3. As they read each new page, they immediately compute its contribution, **update the running summary sheet on the desk using a mathematical scaling formula**, and throw away the temporary intermediate comparison drafts.
4. The giant intermediate comparison sheets **never** leave the desk; they are discarded immediately.
5. The student only walks back to the library once at the very end to store the final summary sheet.

---

## 3. The Core Mechanism of FlashAttention-1: Tiling & Online Softmax

To compute attention block-by-block without storing the intermediate $N \times N$ matrix, we must solve a mathematical hurdle: **Softmax requires global knowledge of the entire row** (specifically the maximum value and the sum of exponentials).

FlashAttention solves this by using **Online Softmax**, updating running scaling statistics dynamically as new blocks of keys/values arrive.

### The Mathematics of Online Softmax
Suppose we have a row vector of scores divided into two blocks: $x = [x^{(1)}, x^{(2)}]$.

1. For the first block $x^{(1)}$, we compute:
   * Running maximum: $m^{(1)} = \max(x^{(1)})$
   * Running sum of exponents: $l^{(1)} = \sum \exp(x^{(1)} - m^{(1)})$
   * Running output numerator: $O^{(1)} = \sum \exp(x^{(1)} - m^{(1)}) \cdot V^{(1)}$

2. When the second block $x^{(2)}$ arrives, we compute its local statistics:
   * Local maximum: $\tilde{m} = \max(x^{(2)})$
   * Local sum of exponents: $\tilde{l} = \sum \exp(x^{(2)} - \tilde{m})$
   * Local numerator: $\tilde{O} = \sum \exp(x^{(2)} - \tilde{m}) \cdot V^{(2)}$

3. To merge the two blocks and get the new running statistics without recalculating block 1 from scratch, we compute the global maximum:
   \[
   m^{\text{new}} = \max(m^{(1)}, \tilde{m})
   \]

4. We then **rescale** the old running sum $l^{(1)}$ and the new local sum $\tilde{l}$ to align with the new maximum $m^{\text{new}}$, and sum them:
   \[
   l^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} + e^{\tilde{m} - m^{\text{new}}} \cdot \tilde{l}
   \]

5. Finally, we update the output vector using the same scaling factors:
   \[
   O^{\text{new}} = \frac{l^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}} \cdot O^{(1)} + e^{\tilde{m} - m^{\text{new}}} \cdot \tilde{O}}{l^{\text{new}}}
   \]

Using this update step, **FlashAttention gets the exact same output as standard attention** but only requires storing $O(N)$ memory states (the scaling factors $m$ and $l$ of size $N$) instead of the $O(N^2)$ attention matrix.
