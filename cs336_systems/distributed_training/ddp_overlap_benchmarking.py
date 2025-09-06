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
from cs336_systems.ddp_overlap import DDPIndividualParameters # Import your custom DDP class

# --- Model Configuration for "XL" size ---
MODEL_CONFIG = {
    "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25
}

# --- DDP Setup and Cleanup Functions ---
def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    # Port is set in the main block
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

# --- The Main DDP Worker Function ---
def ddp_worker(rank, world_size):
    setup(rank, world_size)
    device = f'cuda:{rank}'

    # Instantiate the base model
    model = BasicsTransformerLM(
        vocab_size=10000, context_length=512, rope_theta=10000, **MODEL_CONFIG
    ).to(device)
    

    # wrap the model with the DDPIndividualParameters class
    ddp_model = DDPIndividualParameters(model)
    
    optimizer = AdamW(ddp_model.parameters(), lr=1e-3)
    
    if rank == 0:
        print("Starting DDP benchmark with OVERLAPPED communication (XL model, 2 GPUs)...")
    random_dataset = np.random.randint(0, 10000, size=(10000,))
    warmup_steps, measure_steps = 5, 10
    total_step_times = []

    # --- Benchmarking Loop ---
    for i in range(warmup_steps + measure_steps):
        # Add NVTX range for the entire step for profiling
        with nvtx.range(f"step_{i}"):
            inputs, targets = get_batch(random_dataset, batch_size=4, context_length=512, device=device)

            # Use the DDP-wrapped model for the forward pass
            with nvtx.range("forward_pass"):
                logits = ddp_model(inputs)
            
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # The backward pass will automatically trigger the asynchronous hooks
            with nvtx.range("backward_pass"):
                loss.backward()
            
            # Wait for all background gradient communications to finish before the optimizer step
            with nvtx.range("gradient_sync_wait"):
                ddp_model.finish_gradient_synchronization()
            
            # Perform the optimizer step
            with nvtx.range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

            

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Find a free port to avoid conflicts
    sock = socket.socket(); sock.bind(('', 0)); port = sock.getsockname()[1]; sock.close()
    os.environ["MASTER_PORT"] = str(port)
    
    mp.spawn(ddp_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)

