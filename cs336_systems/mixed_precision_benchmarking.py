import argparse
import torch
import numpy as np
import pandas as pd
import timeit
from contextlib import nullcontext #<-- IMPORT NULLCONTEXT

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy

# --- Model Configurations (from before) ---
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def benchmark_model(model_size: str, precision: str, include_backward: bool, device: str):
    """Benchmarks a model with a specified precision."""
    
    # --- 1. Setup ---
    config = MODEL_CONFIGS[model_size]
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float32
    torch.set_default_device(device)
    
    # Check if the GPU supports bfloat16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print(f"WARNING: Device {device} does not support bfloat16. Skipping.")
        return None

    model = BasicsTransformerLM(vocab_size=10000, context_length=512, rope_theta=10000, **config)
    model.to(device)
    model.train()
    
    random_dataset = np.random.randint(0, 10000, size=(10000,))
    warmup_steps = 5
    measure_steps = 10
    
    # --- 2. CHOOSE THE CONTEXT MANAGER ---
    # This is the key change. If bf16 is selected, we use autocast.
    # Otherwise, we use nullcontext, which does nothing.
    autocast_manager = torch.autocast(device_type="cuda", dtype=dtype) if precision == 'bf16' else nullcontext()

    # --- 3. Benchmarking Loop ---
    timings = []
    # Warmup
    for _ in range(warmup_steps):
        inputs, _ = get_batch(random_dataset, 4, 512, device)
        with autocast_manager:
            logits = model(inputs)
            if include_backward:
                # Use random targets for simplicity
                targets = torch.randint(0, 10000, inputs.shape, device=device)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if include_backward:
            loss.backward() # loss.backward() is NOT in the autocast context

    # Measurement
    for _ in range(measure_steps):
        inputs, targets = get_batch(random_dataset, 4, 512, device)
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        
        # --- APPLY THE CONTEXT MANAGER ---
        with autocast_manager:
            logits = model(inputs)
            if include_backward:
                loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        if include_backward:
            loss.backward()

        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        timings.append(t1 - t0)
        
    # --- 4. Return Results ---
    avg_time_ms = np.mean(timings) * 1000
    return {
        "model_size": model_size,
        "precision": precision,
        "pass_type": "forward+backward" if include_backward else "forward_only",
        "avg_time_ms": avg_time_ms,
    }

if __name__ == "__main__":
    all_results = []
    models_to_test = list(MODEL_CONFIGS.keys())
    precisions_to_test = ['fp32', 'bf16']

    print("--- Starting Mixed Precision Benchmark Sweep ---")
    for model_name in models_to_test:
        for precision_name in precisions_to_test:
            print(f"\n--- Benchmarking Model: {model_name}, Precision: {precision_name} ---")
            
            # Run forward + backward pass
            result = benchmark_model(
                model_size=model_name,
                precision=precision_name,
                include_backward=True,
                device="cuda:0",
            )
            if result:
                all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("mixed_precision_results.csv", index=False)
    
    print("\n--- Benchmark Sweep Complete ---")
    print("Results saved to mixed_precision_results.csv")
    print("\nResults Summary:")
    print(results_df.to_markdown(index=False))
