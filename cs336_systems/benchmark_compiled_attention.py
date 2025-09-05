import torch
import torch.nn.functional as F
import timeit
import pandas as pd
import itertools

# A simple function for our naive attention implementation
def naive_attention(q, k, v):
    d_model = q.shape[-1]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model**0.5)
    attn_probs = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)
    return output

def benchmark(seq_len, d_model, batch_size, compiled_fn, device="cuda"):
    """Generic benchmark function for either the original or compiled version."""
    dtype = torch.float32
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    
    warmup_steps = 10
    measure_steps = 100

    # Warmup
    for _ in range(warmup_steps):
        output = compiled_fn(q, k, v)
        grad_output = torch.randn_like(output)
        output.backward(grad_output, retain_graph=True)
    torch.cuda.synchronize()

    # Time Forward Pass
    forward_timings = []
    for _ in range(measure_steps):
        t0 = timeit.default_timer()
        output = compiled_fn(q, k, v)
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        forward_timings.append(t1 - t0)
    avg_forward_ms = sum(forward_timings) / len(forward_timings) * 1000

    # Time Backward Pass
    backward_timings = []
    for _ in range(measure_steps):
        # We need a fresh graph for each backward pass in the compiled case
        output = compiled_fn(q, k, v) 
        grad_output = torch.randn_like(output)
        torch.cuda.synchronize() # Sync before timing backward
        t0 = timeit.default_timer()
        output.backward(grad_output, retain_graph=False) # No retain_graph needed now
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        backward_timings.append(t1 - t0)
    avg_backward_ms = sum(backward_timings) / len(backward_timings) * 1000
    
    return { "forward_ms": avg_forward_ms, "backward_ms": avg_backward_ms }

if __name__ == "__main__":
    BATCH_SIZE = 8
    D_MODELS = [16, 32, 64, 128]
    SEQ_LENS = [256, 1024, 4096, 8192, 16384]
    
    all_results = []
    
    # --- Create the compiled version of our function ---
    compiled_attention = torch.compile(naive_attention)

    for d_model, seq_len in itertools.product(D_MODELS, SEQ_LENS):
        print(f"--- Benchmarking: seq_len={seq_len}, d_model={d_model} ---")
        
        # Benchmark eager version
        try:
            eager_results = benchmark(seq_len, d_model, BATCH_SIZE, naive_attention)
        except torch.cuda.OutOfMemoryError:
            eager_results = { "forward_ms": "OOM", "backward_ms": "OOM" }
        
        # Benchmark compiled version
        try:
            compiled_results = benchmark(seq_len, d_model, BATCH_SIZE, compiled_attention)
        except torch.cuda.OutOfMemoryError:
            compiled_results = { "forward_ms": "OOM", "backward_ms": "OOM" }
        
        all_results.append({
            "seq_len": seq_len, "d_model": d_model,
            "eager_forward_ms": eager_results["forward_ms"],
            "eager_backward_ms": eager_results["backward_ms"],
            "compiled_forward_ms": compiled_results["forward_ms"],
            "compiled_backward_ms": compiled_results["backward_ms"]
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("compiled_attention_benchmark.csv", index=False)
    
    print("\n--- Benchmark Complete ---")
    print("Results saved to compiled_attention_benchmark.csv")
    print("\nResults Summary:")
    print(results_df.to_markdown(index=False))
