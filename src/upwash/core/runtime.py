import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

@dataclass
class Runtime:
    rank: int = 0
    world_size: int = 1
    device: str = "cpu"
    initialized_here: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0 # rank 0 as the main rank
    
def setup_runtime(device: str) -> Runtime:
    world_size =  int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return Runtime(device=device)
    
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    use_cuda = device.startswith("cuda")

    initialized_here = False
    if not dist.is_initialized():
        if use_cuda:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")
        initialized_here = True

    if use_cuda: 
        device = f"cuda:{local_rank}" 
    else:
        device = "cpu"

    return Runtime(
        rank = rank,
        world_size=world_size,
        device =device,
        initialized_here=initialized_here
    )

def teardown_runtime(runtime: Runtime) -> None:
    if runtime.initialized_here:
        dist.destroy_process_group()

    