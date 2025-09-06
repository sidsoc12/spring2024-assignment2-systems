import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

# Toy dataset (unchanged)
def get_data(n=64, in_dim=10, out_dim=2):
    x = torch.randn(n, in_dim)
    y = torch.randint(0, out_dim, (n,))
    return x, y

# Tiny model (unchanged)
class ToyModel(nn.Module):
    def __init__(self, in_dim=10, hidden=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

# Setup/Cleanup (unchanged)
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# --- MODIFIED DDP WORKER ---
# We now accept an 'init_state_dict' to ensure a common starting point
def ddp_worker(rank, world_size, x, y, baseline_params, init_state_dict):
    setup(rank, world_size)
    
    # Create the model...
    model = ToyModel()
    # ...and explicitly load the shared initial state
    model.load_state_dict(init_state_dict)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    # !!! THIS BROADCAST IS NO LONGER NEEDED !!!
    # Since all processes are explicitly loading the *same state dict*,
    # they are already synchronized. We can remove this loop.
    # for param in model.parameters():
    #     dist.broadcast(param.data, src=0)

    # shard data
    shard_size = x.size(0) // world_size
    start, end = rank * shard_size, (rank + 1) * shard_size
    x_local, y_local = x[start:end], y[start:end]

    # forward + backward
    out = model(x_local)
    loss = criterion(out, y_local) 
    loss.backward()

    # all-reduce each parameter's gradient
    for param in model.parameters():
        dist.all_reduce(param.grad)
        param.grad /= world_size

    optimizer.step()

    # Only rank 0 prints comparison
    if rank == 0:
        print("\n=== Parameter comparison (baseline vs DDP) ===")
        for i, (bp, dp) in enumerate(zip(baseline_params, model.parameters())):
            # Increase tolerance slightly for floating point math differences
            match = torch.allclose(bp, dp.data, atol=1e-5) 
            print(f"Param {i}: Match={match}")
            if not match: # Print more info if they don't match
                print(f"  Baseline: {bp.flatten()[:5]} ...")
                print(f"  DDP:      {dp.data.flatten()[:5]} ...\n")

    cleanup()

# --- MODIFIED SINGLE PROCESS FUNCTION ---
# It also accepts the 'init_state_dict' to ensure it starts from the same point
def single_process(x, y, init_state_dict):
    # Create the model...
    model = ToyModel()
    # ...and explicitly load the shared initial state
    model.load_state_dict(init_state_dict)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return [p.data.clone() for p in model.parameters()]

if __name__ == "__main__":
    world_size = 2
    x, y = get_data()

    # --- NEW: Create ONE initial model state ---
    # We create the model once here in the main process to capture its initial state.
    model_init = ToyModel()
    init_state_dict = model_init.state_dict()
    # We clone the state dict. This isn't strictly necessary for state_dict, 
    # but it's good practice to avoid processes modifying shared memory unexpectedly.
    init_state_dict = {k: v.clone() for k, v in init_state_dict.items()}


    # single-process baseline
    # We pass the initial state to the baseline function
    baseline_params = single_process(x, y, init_state_dict)

    # run naive DDP
    # We also pass the SAME initial state to all DDP workers
    mp.spawn(ddp_worker,
             args=(world_size, x, y, baseline_params, init_state_dict),
             nprocs=world_size,
             join=True)