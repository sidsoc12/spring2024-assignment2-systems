import torch
import numpy as np
import pandas as pd
import timeit
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# --- Model Configurations ---
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    # Not including the largest models to make the benchmark run faster
}

def benchmark_model(model, pass_type, device="cuda"):
    """Generic benchmark function for a given model instance."""
    warmup_steps = 5
    measure_steps = 10
    random_dataset = np.random.randint(0, 10000, size=(10000,))
    optimizer = AdamW(model.parameters())

    # --- Warmup ---
    # The first call to a compiled model triggers the compilation, which is slow.
    # We must do this during the warmup phase.
    print("    Warming up...")
    for _ in range(warmup_steps):
        inputs, targets = get_batch(random_dataset, 4, 512, device)
        logits = model(inputs)
        if pass_type == "full_step":
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.cuda.synchronize()

    # --- Measurement ---
    print("    Measuring...")
    timings = []
    for _ in range(measure_steps):
        inputs, targets = get_batch(random_dataset, 4, 512, device)
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        
        logits = model(inputs)
        
        if pass_type == "full_step":
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        timings.append(t1 - t0)
    
    return np.mean(timings) * 1000

if __name__ == "__main__":
    all_results = []
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n--- Benchmarking Model: {model_name} ---")
        
        # --- Eager Mode ---
        print("Benchmarking Eager Mode...")
        eager_model = BasicsTransformerLM(vocab_size=10000, context_length=512, rope_theta=10000, **config).to("cuda")
        eager_forward_ms = benchmark_model(eager_model, "forward_only")
        eager_full_step_ms = benchmark_model(eager_model, "full_step")
        del eager_model # Free up GPU memory
        
        # --- Compiled Mode ---
        print("Benchmarking Compiled Mode...")
        # Important: create a new model instance before compiling
        compiled_model = torch.compile(BasicsTransformerLM(vocab_size=10000, context_length=512, rope_theta=10000, **config).to("cuda"))
        compiled_forward_ms = benchmark_model(compiled_model, "forward_only")
        compiled_full_step_ms = benchmark_model(compiled_model, "full_step")
        del compiled_model # Free up GPU memory

        all_results.append({
            "model_size": model_name,
            "eager_forward_ms": f"{eager_forward_ms:.2f}",
            "compiled_forward_ms": f"{compiled_forward_ms:.2f}",
            "eager_full_step_ms": f"{eager_full_step_ms:.2f}",
            "compiled_full_step_ms": f"{compiled_full_step_ms:.2f}",
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("compiled_model_benchmark.csv", index=False)

    print("\n--- Benchmark Complete ---")
    print("Results saved to compiled_model_benchmark.csv")
    print("\nResults Summary:")
    print(results_df.to_markdown(index=False))

