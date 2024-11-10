# https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475/3
import torch
from torch.cuda.amp import autocast

net = torch.nn.Conv2d(3,3,3,3).to('cuda')
input = torch.rand([3,3,5,5],device='cuda')

with autocast(enabled=True):
    with torch.no_grad():
        y = net(input)

    z = net(input)
    print('enable=True, z {}'.format(z.requires_grad))
    
with autocast(enabled=False):
    with torch.no_grad():
        y = net(input)

    z = net(input)
    print('enable=False, z {}'.format(z.requires_grad))