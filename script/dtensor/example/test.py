import torch.nn as nn
import torch
from torch.distributed.tensor import Shard, distribute_tensor, distribute_module, init_device_mesh
from einops import rearrange

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.fc1(input) + self.fc2(input))

mesh = init_device_mesh("cuda", (4,))

def shard_params(mod_name, mod, mesh):
    col_linear_placement = [Shard(0)]
    # shard fc1 and fc2
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, col_linear_placement)
            )
            mod.register_parameter(name, dist_param)

sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_params)

big_tensor = torch.randn(4, 8)
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])

o = sharded_module(my_dtensor)
print(o.shape)

print(o.to_local().shape)
print(o.device_mesh)
print(o.placements)

o1 = o.view(-1, 1)
o2 = rearrange(o, "b d -> (b d)")
print("o1", o1.to_local().shape, o1.device_mesh, o1.placements)
print("o2", o2.to_local().shape, o2.device_mesh, o2.placements)