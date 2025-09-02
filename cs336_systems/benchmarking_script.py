import argparse
import timeit
import torch
import numpy as np 
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch 
from cs336_basics.nn_utils import cross_entropy 

MODEL_CONFIGS = { 
    "small" : {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def benchmark_model(model_size: str, context_length: int, batch_size: int, warmup_steps: int, measure_steps: int, include_backward: bool, device: str):
    print("--- Starting Benchmark ---")
    print(f"Model Size: {model_size}, Context Length: {context_length}, Batch Size: {batch_size}")
    print(f"Device: {device}")


    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    
    torch.set_default_device(device)

    config = MODEL_CONFIGS[model_size]

    vocab_size = 10000
    rope_theta = 10000

    model = BasicsTransformerLM(
        vocab_size = vocab_size,
        context_length = context_length,
        rope_theta = rope_theta,
        **config,
    )

    model.to(device)

    model.train()

    print(f"\nModel initialized with {model.get_num_params()/1e6:.2f}M parameters.")

    # generate dataset 

    random_dataset = np.random.randint(0, vocab_size, size = (10000,))

    inputs, targets = get_batch(random_dataset, batch_size, context_length, device)

    print(f"Generated a random data batch of shape: {inputs.shape}")

    timings = []

    for i in range( warmup_steps + measure_steps):
        inputs, targets = get_batch(random_dataset, batch_size, context_length, device)


        if "cuda" in device:
            # make sure gpu preparation stuff is done and not happening asynchronously
            torch.cuda.synchronize()
        
        t0 = timeit.default_timer()

        # forward pass 
        logits = model(inputs)

        if include_backward:
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
        
        if "cuda" in device:
            torch.cuda.synchronize()

        t1 = timeit.default_timer()

        if i >= warmup_steps:
            timings.append(t1 -t0)
        
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{warmup_steps + measure_steps} complete.")

    timings_s = np.array(timings)

    timings_ms = timings_s * 1000

    avg_time_ms = np.mean(timings_ms)
    std_dev_ms = np.std(timings_ms)

    print("\n--- Benchmark Results ---")
    print(f"Average time per step: {avg_time_ms:.2f} ms")
    print(f"Standard deviation: {std_dev_ms:.2f} ms")
    print("-------------------------\n")

    return {
        "model_size": model_size,
        "pass_type": "forward+backward" if include_backward else "forward_only",
        "avg_time_ms": avg_time_ms,
        "std_dev_ms": std_dev_ms,
        "context_length": context_length,
        "batch_size": batch_size
    }
            
        



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Benchmark a Transformer model.")

    # # parser.add_argument("--model_size", type=str, required=True, choices=MODEL_CONFIGS.keys(),
    # #                     help="Size of the model to benchmark.")
    
    # # parser.add_argument("--warmup_steps", type=int, default=5,
    # #                     help="Number of warmup steps before timing.")
    
    # # # --measure_steps: How many steps to time and average.
    # # parser.add_argument("--measure_steps", type=int, default=10,
    # #                     help="Number of steps to measure and average.")
    
    # # parser.add_argument("--no_backward", action="store_true",
    # #                     help="If set, only benchmark the forward pass.")
    

    # # parser.add_argument("--device", type=str, default="cuda:0",
    # #                     help="Device to run on (e.g., 'cuda:0', 'cpu').")
    
    # # parser.add_argument("--context_length", type=int, default=1024,
    # #                     help="Context length for the model.")
    
    # # parser.add_argument("--batch_size", type=int, default=4,
    # #                     help="Batch size for training.")
   
    # # parser.add_argument("--output_csv", type=str, default="benchmark_results.csv",
    # #                     help="Path to save the benchmark results CSV file.")


    # args = parser.parse_args()


    parser = argparse.ArgumentParser(description="Run the full benchmark suite.")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv",
                        help="Path to save the benchmark results CSV file.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on (e.g., 'cuda:0', 'cpu').")
    args = parser.parse_args()

    # This list will hold all the result dictionaries from our benchmark runs.
    all_results = []

    # Define the model sizes we want to test, right from the config dictionary.
    models_to_test = list(MODEL_CONFIGS.keys())

    print("--- Starting Full Benchmark Sweep ---")
    # Loop through each model size.
    for model_name in models_to_test:
        print(f"\n--- Benchmarking Model: {model_name} ---")

       
        print(f"Running forward-only pass...")
        result_fwd = benchmark_model(
            model_size=model_name,
            context_length=512,
            batch_size=4,
            warmup_steps=5,
            measure_steps=10,
            include_backward=False, 
            device=args.device,
        )
        all_results.append(result_fwd)

      
        print(f"Running forward+backward pass...")
        result_fwd_bwd = benchmark_model(
            model_size=model_name,
            context_length=512,
            batch_size=4,
            warmup_steps=5,
            measure_steps=10,
            include_backward=True, # This is the key difference
            device=args.device,
        )
        all_results.append(result_fwd_bwd)

    
    import pandas as pd
    results_df = pd.DataFrame(all_results)

    # Save the DataFrame to a CSV file.
    results_df.to_csv(args.output_csv, index=False)

    print("\n--- Benchmark Sweep Complete ---")
    print(f"Results saved to {args.output_csv}")


    print("\nResults Summary:")
    print(results_df.to_markdown(index=False))

    