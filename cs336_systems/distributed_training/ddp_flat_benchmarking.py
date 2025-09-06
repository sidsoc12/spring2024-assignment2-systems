import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import numpy as np
import socket
import torch.cuda.nvtx as nvtx # Import NVTX

# Import PyTorch's internal utilities for flattening tensors
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# Import our custom components
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# --- Model Configuration ---
MODEL_CONFIG = {
    "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25
}

# --- DDP Setup and Cleanup Functions ---
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    # Port is set in the main block
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
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    if rank == 0:
        print("Starting DDP benchmark with FLATTENED gradients (XL model, 2 GPUs)...")
    random_dataset = np.random.randint(0, 10000, size=(10000,))
    warmup_steps, measure_steps = 5, 10

    for i in range(warmup_steps + measure_steps):
        with nvtx.range(f"step_{i}"):
            inputs, targets = get_batch(random_dataset, batch_size=4, context_length=512, device=device)

            with nvtx.range("forward_pass"):
                logits = model(inputs)
            
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            with nvtx.range("backward_pass"):
                loss.backward()
            
            with nvtx.range("communication"):
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                flat_gradients = _flatten_dense_tensors(gradients)
                dist.all_reduce(flat_gradients, op=dist.ReduceOp.SUM)
                flat_gradients /= world_size
                for grad, flat_grad in zip(gradients, _unflatten_dense_tensors(flat_gradients, gradients)):
                    grad.copy_(flat_grad)
            
            with nvtx.range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

if __name__ == "__main__":
    world_size = 2
    sock = socket.socket(); sock.bind(('', 0)); port = sock.getsockname()[1]; sock.close()
    os.environ["MASTER_PORT"] = str(port)
    mp.spawn(ddp_worker, args=(world_size,), nprocs=world_size, join=True)

