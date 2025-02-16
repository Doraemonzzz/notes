import os
import sys
import torch
import torch.nn as nn

from torch.distributed._tensor import Shard

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

# ---- GPU check ------------
_min_gpu_count = 2

from log_utils import rank_log, get_logger, verify_min_gpu_count
# ---------------------------


from torch.distributed._tensor.device_mesh import init_device_mesh



"""
This is the script to test Sequence Parallel(SP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

We use the example of two `nn.Linear` layers with an element-wise `nn.RELU`
in between to show an example of sequence parallel, which was proposed in paper:

https://arxiv.org/pdf/2205.05198.pdf.

Like tensor parallel, we parallelize the first linear layer by column
and also parallelize the second linear layer by row. But the input in each rank
now is different so that we need one all-gather for input and one reduce-scatter
in the end of the second linear layer.
"""


class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


"""
Main body of the demo of a basic version of sequence parallel by using
PyTorch native APIs.
"""
logger = get_logger()

# create a device mesh based on the given world_size.
device_mesh = init_device_mesh(
    device_type="cuda", mesh_shape=(int(os.environ["WORLD_SIZE"]),)
)

_rank = device_mesh.get_rank()

print(f"Starting PyTorch Sequence Parallel example on rank {_rank}.")

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

# create model and move it to GPU.  Init_device_mesh has already assigned gpu ids...
model = ToyModel().to("cuda")
print(model)

# Custom parallelization plan for the model
sp_model = parallelize_module(
    module=model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
    },
)

print(sp_model)
print(sp_model.in_proj.weight.shape)