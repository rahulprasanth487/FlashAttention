import math
import time
import torch
import torch.nn as nn

class NaiveSelfAttentionBenchmark(nn.Module):
    """
    A simple, self-contained PyTorch module that implements standard (naive)
    self-attention. It records the intermediate attention weights matrix to
    demonstrate O(N^2) memory scaling.
    """
    def __init__(self, vocab_size, d_model=64, d_k=32):
        super().__init__()
        self.d_k = d_k
        
        # Trainable embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
        # Projection matrices for Query, Key, and Value
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)
        
        # Container to explicitly store intermediate weights for memory demonstration
        self.attn_weights = None

    def forward(self, x):
        # x shape: (seq_len)
        
        # 1. Token Embeddings: shape (seq_len, d_model)
        embeddings = self.embedding(x)
        
        # 2. Linear projections to Q, K, V: shape (seq_len, d_k)
        Q = self.W_Q(embeddings)
        K = self.W_K(embeddings)
        V = self.W_V(embeddings)
        
        # 3. Attention scores: (Q @ K^T) / sqrt(d_k)
        # shape: (seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Softmax activation to get attention weights
        # shape: (seq_len, seq_len)
        self.attn_weights = torch.softmax(scores, dim=-1)
        
        # 5. Output calculation: weights @ V
        # shape: (seq_len, d_k)
        output = torch.matmul(self.attn_weights, V)
        
        return output

def benchmark_device(model, token_tensor, device, warmups=5, runs=10):
    """
    Benchmarks the model on a given device (CPU/GPU) with warmups and timed runs.
    """
    # Move model and input tensor to the target device
    model = model.to(device)
    input_device = token_tensor.to(device)
    
    # Warm-up runs (to compile/initialize device kernels)
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(input_device)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # Reset peak memory tracker before measuring
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            start = time.perf_counter()
            _ = model(input_device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000.0) # Convert to ms
            
    # Calculate average time and standard deviation
    avg_time = sum(times) / len(times)
    variance = sum((t - avg_time) ** 2 for t in times) / len(times)
    std_time = math.sqrt(variance)
    
    # Measure peak GPU memory if on CUDA
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024) # Convert bytes to MB
    else:
        peak_mem = None
        
    # Get sample output slice (first 5 rows, first 5 cols) and attention shape
    with torch.no_grad():
        output = model(input_device)
        attn_shape = model.attn_weights.shape
        output_slice = output[:5, :5].cpu().numpy()
        
    return avg_time, std_time, peak_mem, attn_shape, output_slice

def main():
    # 1. Input Sample Sentence
    sentence = "The quick brown fox jumps over the lazy dog today"
    words = sentence.split()
    seq_len = len(words)
    print(f"Input Sentence: '{sentence}'")
    print(f"Sequence Length (N): {seq_len} words")
    
    # 2. Tokenization and Vocabulary Building
    unique_words = sorted(list(set(w.lower() for w in words)))
    vocab = {word: idx for idx, word in enumerate(unique_words)}
    token_indices = [vocab[w.lower()] for w in words]
    token_tensor = torch.tensor(token_indices, dtype=torch.long)
    
    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Tokens: {words}")
    print(f"Token Indices Tensor: {token_indices}\n")
    
    # 3. Model Initialization
    d_model = 64
    d_k = 32
    vocab_size = len(vocab)
    
    torch.manual_seed(42)
    model = NaiveSelfAttentionBenchmark(vocab_size=vocab_size, d_model=d_model, d_k=d_k)
    
    # 4. CPU & GPU Verification (using the small sentence)
    print("--- Verification Run (N=10) ---")
    cpu_device = torch.device("cpu")
    cpu_avg, cpu_std, _, cpu_attn_shape, cpu_slice = benchmark_device(
        model, token_tensor, cpu_device, warmups=5, runs=10
    )
    
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_device = torch.device("cuda")
        gpu_avg, gpu_std, gpu_mem, gpu_attn_shape, gpu_slice = benchmark_device(
            model, token_tensor, gpu_device, warmups=5, runs=10
        )
    else:
        gpu_avg, gpu_std, gpu_mem, gpu_attn_shape, gpu_slice = None, None, None, None, None
        
    print(f"  - Attention Weights Shape: {cpu_attn_shape}")
    print("CPU Output Slice (First 5x5):")
    print(cpu_slice)
    if gpu_available:
        print("GPU Output Slice (First 5x5):")
        print(gpu_slice)
    print("\n" + "="*80 + "\n")
    
    # 5. Scalability Benchmark
    # We will test sequence lengths: 10, 128, 512, 1024, 2048
    seq_lengths = [10, 128, 512, 1024, 2048]
    results = []
    
    print("Running Scaling Benchmark (Increasing Complexity)...")
    for N in seq_lengths:
        print(f"Benchmarking N = {N} ...")
        # For scaled sequences, we generate random token indices within the vocab range
        scaled_indices = torch.randint(0, vocab_size, (N,), dtype=torch.long)
        
        # Calculate FLOPs for this sequence length
        qk_flops = 2 * (N ** 2) * d_k
        weights_v_flops = 2 * (N ** 2) * d_k
        softmax_flops = 3 * (N ** 2)
        total_flops = qk_flops + weights_v_flops + softmax_flops
        
        # Benchmark CPU
        c_avg, c_std, _, _, _ = benchmark_device(model, scaled_indices, cpu_device, warmups=5, runs=10)
        
        # Benchmark GPU
        if gpu_available:
            g_avg, g_std, g_mem, _, _ = benchmark_device(model, scaled_indices, gpu_device, warmups=5, runs=10)
        else:
            g_avg, g_std, g_mem = None, None, None
            
        results.append({
            'N': N,
            'flops': total_flops,
            'cpu_time': c_avg,
            'cpu_std': c_std,
            'gpu_time': g_avg,
            'gpu_std': g_std,
            'gpu_mem': g_mem
        })
        
    # 6. Consolidated Results Table
    print("\n" + "="*90)
    print("SCALED BENCHMARK RESULTS TABLE (CPU vs GPU)")
    print("="*90)
    header = f"{'Seq Len (N)':<12} | {'Total FLOPs':<14} | {'CPU Time (ms)':<18} | {'GPU Time (ms)':<18} | {'GPU Peak Mem (MB)':<18}"
    print(header)
    print("-" * 90)
    
    for r in results:
        flops_str = f"{r['flops']:,}"
        cpu_time_str = f"{r['cpu_time']:.4f} ± {r['cpu_std']:.4f}"
        if gpu_available:
            gpu_time_str = f"{r['gpu_time']:.4f} ± {r['gpu_std']:.4f}"
            gpu_mem_str = f"{r['gpu_mem']:.4f} MB"
        else:
            gpu_time_str = "N/A"
            gpu_mem_str = "N/A"
            
        print(f"{r['N']:<12} | {flops_str:<14} | {cpu_time_str:<18} | {gpu_time_str:<18} | {gpu_mem_str:<18}")
    print("="*90)
    
    print("\nExplanation of GPU vs CPU Timing:")
    print("1. For N = 10, the CPU runs faster because GPU kernel launch overhead (typically 10-50 microseconds)")
    print("   dominates the execution. The workload is too small to saturate the GPU's parallel cores.")
    print("2. As N increases, the quadratic O(N^2) complexity of attention scales up the compute requirements.")
    print("   The GPU's parallel architecture excels here, and it will outperform the CPU by a large margin")
    print("   as the sequence length gets larger.")

if __name__ == "__main__":
    main()
