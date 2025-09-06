import torch
import torch.nn.functional as F
import pandas as pd
import itertools
import triton
from cs336_systems.flash_attention_triton import FlashAttentionTriton

def vanilla_attention(q, k, v, is_causal):
    """Standard, 'naïve' PyTorch implementation of attention."""
    d_model = q.shape[-1]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model**0.5)
    if is_causal:
        causal_mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool), diagonal=1)
        attn_scores.masked_fill_(causal_mask, float('-inf'))
    attn_probs = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)
    return output

def run_benchmark(seq_len, d_model, dtype, is_flash):
    """Runs the benchmark for a single configuration and returns a dictionary of latencies."""
    BATCH_SIZE, NUM_HEADS = 1, 1 # Fixed as per prompt
    IS_CAUSAL = True
    
    # Use a try-except block to gracefully handle Out-of-Memory errors
    try:
        q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, d_model, device="cuda", dtype=dtype)
        k = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, d_model, device="cuda", dtype=dtype)
        v = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, d_model, device="cuda", dtype=dtype)
        dO = torch.randn_like(q)

        # Select the correct attention function
        attention_fn = FlashAttentionTriton.apply if is_flash else vanilla_attention
        
        # Define lambdas for the three measurements that do_bench will call
        fwd_fn = lambda: attention_fn(q, k, v, IS_CAUSAL)
        bwd_fn = lambda: fwd_fn().backward(dO, retain_graph=True)
        fwd_bwd_fn = lambda: fwd_fn().backward(dO)

        # Run the benchmarks using triton.testing.do_bench
        # It returns (min, max, mean, median, num_iters)
        # We will use the median as it's robust to outliers.
        fwd_latency = triton.testing.do_bench(fwd_fn, rep=50)[3]
        bwd_latency = triton.testing.do_bench(bwd_fn, rep=50)[3]
        fwd_bwd_latency = triton.testing.do_bench(fwd_bwd_fn, rep=50)[3]
        
        return {
            "forward_ms": fwd_latency,
            "backward_ms": bwd_latency,
            "fwd_bwd_ms": fwd_bwd_latency,
        }
    except torch.cuda.OutOfMemoryError:
        return { "forward_ms": "OOM", "backward_ms": "OOM", "fwd_bwd_ms": "OOM" }

if __name__ == "__main__":
    SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    D_MODELS = [16, 32, 64, 128]
    DTYPES = [torch.float32, torch.bfloat16]
    
    all_results = []
    
    # Iterate through the Cartesian product of all configurations
    for seq_len, d_model, dtype in itertools.product(SEQ_LENS, D_MODELS, DTYPES):
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        print(f"--- Benchmarking: seq_len={seq_len}, d_model={d_model}, dtype={dtype_str} ---")

        print("  Running Vanilla PyTorch...")
        vanilla_results = run_benchmark(seq_len, d_model, dtype, is_flash=False)
        
        print("  Running FlashAttention (Triton)...")
        flash_results = run_benchmark(seq_len, d_model, dtype, is_flash=True)
        
        all_results.append({
            "seq_len": seq_len,
            "d_model": d_model,
            "precision": dtype_str,
            "vanilla_fwd_ms": vanilla_results["forward_ms"],
            "flash_fwd_ms": flash_results["forward_ms"],
            "vanilla_bwd_ms": vanilla_results["backward_ms"],
            "flash_bwd_ms": flash_results["backward_ms"],
            "vanilla_fwd_bwd_ms": vanilla_results["fwd_bwd_ms"],
            "flash_fwd_bwd_ms": flash_results["fwd_bwd_ms"],
        })
        
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("flash_attention_benchmark.csv", index=False)
    
    print("\n--- Benchmark Complete ---")
    print("Results saved to flash_attention_benchmark.csv")
    print("\nResults Summary:")
    print(results_df.to_markdown(index=False))
