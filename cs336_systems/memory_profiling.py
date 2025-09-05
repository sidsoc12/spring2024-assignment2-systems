import argparse
import torch
import numpy as np
from contextlib import nullcontext

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# Configuration for the 2.7B model
MODEL_CONFIG = {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}

def profile_memory(context_length: int, precision: str, full_train_step: bool, device: str):
    """Profiles a single run of the 2.7B model and saves a memory snapshot."""
    
    print(f"--- Starting Memory Profile ---")
    print(f"Context: {context_length}, Precision: {precision}, Full Step: {full_train_step}")

    # --- 1. Setup ---
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float32
    torch.set_default_device(device)
    
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"Device {device} does not support bfloat16.")

    model = BasicsTransformerLM(vocab_size=10000, context_length=context_length, rope_theta=10000, **MODEL_CONFIG)
    model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters())
    random_dataset = np.random.randint(0, 10000, size=(10000,))
    inputs, targets = get_batch(random_dataset, 4, context_length, device)
    
    autocast_manager = torch.autocast(device_type="cuda", dtype=dtype) if precision == 'bf16' else nullcontext()

    # --- 2. Run and Profile ---
    # Warm-up one step to avoid capturing initialization overhead
    with autocast_manager:
        logits = model(inputs)
    
    # Start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # The single, profiled step
    with autocast_manager:
        logits = model(inputs)
        if full_train_step:
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    if full_train_step:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # --- 3. Save Snapshot ---
    output_filename = f"mem_snapshot_ctx{context_length}_{precision}_{'train' if full_train_step else 'infer'}.pickle"
    torch.cuda.memory._dump_snapshot(output_filename)
    torch.cuda.memory._record_memory_history(enabled=None) # Stop recording
    
    print(f"--- Profile Complete. Snapshot saved to {output_filename} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile model memory usage.")
    parser.add_argument("--context_length", type=int, required=True, choices=[128, 256, 512, 1024])
    parser.add_argument("--precision", type=str, default='fp32', choices=['fp32', 'bf16'])
    parser.add_argument("--full_train_step", action='store_true', help="Run a full train step instead of just inference.")
    args = parser.parse_args()

    profile_memory(
        context_length=args.context_length,
        precision=args.precision,
        full_train_step=args.full_train_step,
        device="cuda:0",
    )