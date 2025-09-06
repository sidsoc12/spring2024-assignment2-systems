import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import numpy as np
import socket
import torch.cuda.nvtx as nvtx # Import NVTX

# Import our custom components
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from ddp_overlap import DDPIndividualParameters

# --- Model Configuration for "XL" size ---
MODEL_CONFIG = {
    "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25
}

# --- DDP Setup and Cleanup Functions ---
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# --- The Main DDP Worker Function ---
def ddp_worker(rank, world_size):
    setup(rank, world_size)
    device = f'cuda:{rank}'

    model = BasicsTransformerLM(
        vocab_size=10000, context_length=512, rope_theta=10000, **MODEL_CONFIG
    ).to(device)
    
    ddp_model = DDPIndividualParameters(model)
    
    optimizer = AdamW(ddp_model.parameters(), lr=1e-3)
    
    if rank == 0:
        print("Starting DDP benchmark with OVERLAPPED communication (XL model, 2 GPUs)...")
    random_dataset = np.random.randint(0, 10000, size=(10000,))
    warmup_steps, measure_steps = 5, 10
    total_step_times = []

    # --- Benchmarking Loop ---
    for i in range(warmup_steps + measure_steps):
      
        torch.cuda.synchronize(device)
        step_start_time = timeit.default_timer()
       
        
        with nvtx.range(f"step_{i}"):
            inputs, targets = get_batch(random_dataset, batch_size=4, context_length=512, device=device)

            with nvtx.range("forward_pass"):
                logits = ddp_model(inputs)
            
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            with nvtx.range("backward_pass"):
                loss.backward()
            
            with nvtx.range("gradient_sync_wait"):
                ddp_model.finish_gradient_synchronization()
            
            with nvtx.range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()
        
      
        torch.cuda.synchronize(device)
        step_end_time = timeit.default_timer()

        if i >= warmup_steps:
            total_step_times.append(step_end_time - step_start_time)
       
    if rank == 0:

        avg_step_time_ms = np.mean(total_step_times) * 1000
        print(f"\n--- DDP Benchmark Results (Overlapped) ---")
        print(f"Average time per training step: {avg_step_time_ms:.2f} ms")
        print("----------------------------------------------------")
       

    cleanup()

if __name__ == "__main__":
    world_size = 2
    sock = socket.socket(); sock.bind(('', 0)); port = sock.getsockname()[1]; sock.close()
    os.environ["MASTER_PORT"] = str(port)
    
    mp.spawn(ddp_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)


