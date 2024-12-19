import torch
from transformers import AutoModelForCausalLM

def generate(model, x):
    model.eval()
    b, n = x.shape
    y = []
    past_key_values = None
    with torch.inference_mode():
        for i in range(n):
            output = model(
                input_ids=x[:, i : i + 1].contiguous(),
                past_key_values=past_key_values,
            )
            past_key_values = output["past_key_values"]
            y.append(output["logits"].contiguous())

    y = torch.cat(y, dim=1)
    return y

name = "facebook/opt-125m"
device = torch.cuda.current_device()
m = 50272
b = 2
n = 128
dtype = torch.bfloat16
dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(name).to(device).to(dtype)

for n in [32, 64, 128, 256]:
    input = torch.randint(0, m, (b, n)).to(device)
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        o1 = model(input)["logits"]
        o2 = generate(model, input)
    print(f"n: {n}, diff: {torch.norm(o1 - o2)}")