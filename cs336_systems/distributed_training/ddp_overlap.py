import torch
import torch.nn as nn
import torch.distributed as dist 
from typing import List

class DDPIndividualParameters(nn.Module):


    """
    A container for a PyTorch nn.Module that handles distributed data parallel
    training by overlapping gradient communication with the backward pass.
    Gradients for each parameter are communicated individually and asynchronously.
    """

    """
    Wraps a model, it then loads model weights from rank 0 to all other ranks, 
    then it register a hook on each parameters' backward pass that does an asynchronous all-reduce call of just that parameter right after the gradient has been calculated.
   (The cpu would just issue the call and then move on withou waiting for it to be dispatched to the gpu
    , then it has a function that allows for the waiting of all unfinished asynchronous ommunication/all-reduce  calls.)
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # get world size 
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1
        
        # communication handles for async calls to All-Reduce
        self.handles: List[dist.Work] = []

        # broadcast parameters from rank 0 to all other ranks 
        if self.world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
        
        # register backward hook for each parameter in module 
        for param in self.module.parameters():
            if param.requires_grad:
                # hook is a function that is called after the gradient is computed 
                param.register_post_accumulate_grad_hook(self._hook)
        
    def _hook(self, param: torch.Tensor) -> None:
        """
        Hook function that is called after the gradient is computed for a parameter.
        """

        if param.grad is not None:
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self) -> None:
        """
        Wait for all pending gradient synchronization operations to complete.
        """

        for handle in self.handles:
            handle.wait()
        
        self.handles.clear() # clear handles after waiting for them to complete 

        if self.world_size > 1:
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad.data /= self.world_size

        
        

            
     
          
