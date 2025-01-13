from datasets import load_dataset

dataset = load_dataset(
    path="/mnt/iem-nas/home/qinzhen/data/hf/fineweb-edu",
    name="sample-10BT",
    split='train',
    streaming=True,
    trust_remote_code=True
)

print(dataset)