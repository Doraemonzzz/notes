# https://github.com/huggingface/transformers/pull/35207
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

num_batch = 32
gradient_accumulation_steps = 2  # or 1
per_device_train_batch_size = 3  # or 6
seq_len = 5

eff_batch_size = per_device_train_batch_size * gradient_accumulation_steps
dataset_len = num_batch * eff_batch_size

data = torch.arange(0, dataset_len * seq_len)
data = data.reshape(dataset_len, seq_len)
data = data.tolist()

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")
dataset = Dataset.from_dict({"input_ids": data, "labels": data})

args = TrainingArguments(
    output_dir=f"out_bs_{eff_batch_size}_grad_{gradient_accumulation_steps}_before",
    per_device_train_batch_size= per_device_train_batch_size,
    gradient_accumulation_steps= gradient_accumulation_steps,
    logging_steps=2,
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)

trainer.train()