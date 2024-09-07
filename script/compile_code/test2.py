import torch

fn = torch.compile(torch.square)
x = torch.tensor([2], device=torch.cuda.current_device())
y = fn(x)
# torch._logging.set_logs(output_code=True)