import os
import torch
import time
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # initialize communication for each process
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        # map each process to a GPU
        torch.cuda.set_device(rank)

# ran for each process
def benchmark_allreduce(rank, world_size, tensor_size_mb, num_iters=20, warmup=5):

    # Initialize each process
    setup(rank, world_size, backend)


    # pick device
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    
    num_elements = tensor_size_mb * 1024 * 1024 // 4  # convert MB to number of elements

    # each process creates data on its own device
    data = torch.randn(num_elements, dtype = torch.float32, device = device)

    # warmup
    for _ in range(warmup):
        dist.all_reduce(data)
        if backend == "nccl":
            torch.cuda.synchronize()
    
    start = time.time()

    for _ in range(num_iters):
        dist.all_reduce(data)
        if backend == "nccl":
            torch.cuda.synchronize()
    
    end = time.time()

    avg_time = (end - start) / num_iters

    # collect timings from all ranks

    avg_time_tensor = torch.tensor([avg_time], device=device)
    times = [torch.zeros(1, device=device) for _ in range(world_size)]

    dist.all_gather(times, avg_time_tensor)

    if rank == 0:   
        print(f"Backend={backend}, world_size={world_size}, size={tensor_size_mb}MB, times={[t.item() for t in times]}")
    
    dist.destroy_process_group()









if __name__ == "__main__":
    backends = ["gloo", "nccl"]
    tensor_sizes = [1,10,100,1024]
    world_sizes = [2,4,6]

    for backend in backends:
        for ws in world_sizes:
            for ts in tensor_sizes:
                # spawn distributed processes
                mp.spawn(benchmark_allreduce, args=(ws, backend, ts), nprocs = ws, join=True)

