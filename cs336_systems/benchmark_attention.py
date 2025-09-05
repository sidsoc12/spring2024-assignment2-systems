import torch
import torch.nn.functional as F
import timeit
import pandas as pd
import itertools

def benchmark_attention(seq_len, d_model, batch_size, device="cuda"):
    """
    Benchmarks the forward and backward pass of a naïve attention implementation.
    """
    print(f"--- Benchmarking: seq_len={seq_len}, d_model={d_model} ---")
    
    # Use float32 for accurate memory accounting
    dtype = torch.float32
    
    # Create random inputs on the specified device
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    
    # We need gradients for the backward pass
    q.requires_grad = True
    k.requires_grad = True
    
    warmup_steps = 10
    measure_steps = 100

    # --- Warmup ---
    for _ in range(warmup_steps):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model**0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        # Create a dummy gradient for the backward pass warmup
        grad_output = torch.randn_like(output)
        output.backward(grad_output, retain_graph=True)

    torch.cuda.synchronize()

    # --- Time Forward Pass ---
    forward_timings = []
    for _ in range(measure_steps):
        t0 = timeit.default_timer()
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model**0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        forward_timings.append(t1 - t0)
    
    avg_forward_ms = sum(forward_timings) / len(forward_timings) * 1000

    # --- Measure Memory and Time Backward Pass ---
    # We need to run one more forward pass to have something to backpropagate from
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model**0.5)
    attn_probs = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)
    grad_output = torch.randn_like(output)
    
    # The key memory measurement: how much is allocated before the backward pass?
    # This includes the large (N, N) attn_probs matrix needed for the gradient calculation.
    torch.cuda.synchronize()
    mem_before_backward_gb = torch.cuda.memory_allocated(device) / (1024**3)

    backward_timings = []
    for _ in range(measure_steps):
        t0 = timeit.default_timer()
        # The backward pass re-uses the graph from the last forward pass
        output.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        backward_timings.append(t1 - t0)
        
    avg_backward_ms = sum(backward_timings) / len(backward_timings) * 1000
    
    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "forward_ms": avg_forward_ms,
        "backward_ms": avg_backward_ms,
        "mem_gb": mem_before_backward_gb
    }

if __name__ == "__main__":
    
    BATCH_SIZE = 8
    D_MODELS = [16, 32, 64, 128]
    SEQ_LENS = [256, 1024, 4096, 8192, 16384]
    
    all_results = []
    
    # Iterate through the Cartesian product of all configurations
    for d_model, seq_len in itertools.product(D_MODELS, SEQ_LENS):
        try:
            result = benchmark_attention(seq_len, d_model, BATCH_SIZE)
            all_results.append(result)
        except torch.cuda.OutOfMemoryError:
            print(f"--- OOM at: seq_len={seq_len}, d_model={d_model} ---")
            # Record the OOM error
            all_results.append({
                "seq_len": seq_len,
                "d_model": d_model,
                "forward_ms": "OOM",
                "backward_ms": "OOM",
                "mem_gb": "OOM"
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("naive_attention_benchmark.csv", index=False)
    
    print("\n--- Benchmark Complete ---")
    print("Results saved to naive_attention_benchmark.csv")
    print("\nResults Summary:")
    print(results_df.to_markdown(index=False))