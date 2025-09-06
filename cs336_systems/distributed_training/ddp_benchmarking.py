import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import numpy as np


from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


# As per the prompt, we are benchmarking the "XL" model size.
MODEL_CONFIG = {
    "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25
}


def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # 'nccl' backend for NVIDIA GPUs.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Pin the process to a specific GPU
    torch.cuda.set_device(rank)

def cleanup():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


def ddp_worker(rank, world_size):
    """
    This function is executed by each GPU process.
    """
    setup(rank, world_size)
    device = f'cuda:{rank}'

    # 1. Model Initialization
    # We only initialize with random weights on the main process (rank 0).
    # All other processes will receive a copy.
    model = BasicsTransformerLM(
        vocab_size=10000, context_length=512, rope_theta=10000, **MODEL_CONFIG
    ).to(device)
    
    # 2. Synchronize Initial Model Weights
    # This ensures all models start with the exact same parameters.
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Create a dummy dataset for benchmarking
    # Each process creates its own data to avoid conflicts.
    if rank == 0:
        print("Starting DDP benchmark for XL model on 2 GPUs...")
    random_dataset = np.random.randint(0, 10000, size=(10000,))

    # Benchmarking parameters
    warmup_steps = 5
    measure_steps = 10
    total_step_times = []
    communication_times = []

    # --- Benchmarking Loop ---
    for i in range(warmup_steps + measure_steps):
        # Start timer for the full step
        torch.cuda.synchronize(device)
        step_start_time = timeit.default_timer()
        
        # Each process gets a full batch of data. DDP handles sharding internally
        # when you use a DistributedSampler, but for this simple case, we just
        # let each process compute on a full batch.
        inputs, targets = get_batch(random_dataset, batch_size=4, context_length=512, device=device)

        # Forward and Backward pass
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

        # --- Measure Communication Time ---
        comm_start_time = timeit.default_timer()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        # Synchronize to wait for the All-Reduce to complete
        torch.cuda.synchronize(device)
        comm_end_time = timeit.default_timer()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Stop timer for the full step
        torch.cuda.synchronize(device)
        step_end_time = timeit.default_timer()

        # Record timings only after warmup
        if i >= warmup_steps:
            total_step_times.append(step_end_time - step_start_time)
            communication_times.append(comm_end_time - comm_start_time)

  
    # Only the main process (rank 0) should print the final results.
    if rank == 0:
        avg_step_time_ms = np.mean(total_step_times) * 1000
        avg_comm_time_ms = np.mean(communication_times) * 1000
        comm_proportion = (avg_comm_time_ms / avg_step_time_ms) * 100
        
        print("\n--- DDP Benchmark Results (XL Model, 2 GPUs) ---")
        print(f"Average time per training step: {avg_step_time_ms:.2f} ms")
        print(f"Average time spent on gradient communication: {avg_comm_time_ms:.2f} ms")
        print(f"Proportion of time spent on communication: {comm_proportion:.2f}%")
        print("--------------------------------------------------")

    cleanup()


if __name__ == "__main__":
    # We are running on a single node with 2 GPUs
    world_size = 2
    mp.spawn(ddp_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)
