# import wandb
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

local_dir = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment/nanogpt/wandb"
files = []
for dir in os.listdir(local_dir):
    dir_path = os.path.join(local_dir, dir)
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            if file.endswith(".wandb"):
                print(file)
                files.append(os.path.join(dir_path, file))


for file in files:
    data = pd.read_json(file)
    print(data.keys())
    assert False

sns.set_theme()

# 4.3
keyword = None
names = ["llama_train_xmixers_llama_small_50b_init1_token_mixer_init2", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1_token_mixer_init2", "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2", "llama_train_xmixers_llama_small_50b_init1", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1", "gpt_train_xmixers_gpt_small_50b_init1", "train_xmixers_llama_small_50b_token_mixer_init2"]
y_name = 'val/loss'
y_name_label = "Validation Loss"
# y_name = 'train/loss'
# y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_xmixers_llama_small_50b_token_mixer_init2": "LLaMA-method4.1",
    "llama_train_xmixers_llama_small_50b_init1_token_mixer_init2": "LLaMA-method4.2",
    # "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1_token_mixer_init2": "GPT-method4.2",
    "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2": "GPT-sincos-method4.2",
    
    "llama_train_xmixers_llama_small_50b_init1": "LLaMA-method3",
    # "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1": "GPT-method3",
    "gpt_train_xmixers_gpt_small_50b_init1": "GPT-sincos-method3",
}
hue_order = ["GPT-method3", "GPT-method4.2", "GPT-sincos-method3", "GPT-sincos-method4.2", "LLaMA-method3", "LLaMA-method4.1", "LLaMA-method4.2"]
title = "Language Model(124M) loss curve"
output_name = f"method4.2_vs_3_{y_name}".replace("/", "_")
folder = "v1"

os.makedirs(folder, exist_ok=True)

for run in runs: 
    if not y_name in run.summary:
        continue
    name = run.name
    if keyword is not None:
        if keyword in name:
            hist = run.history(keys=['iter', y_name])
            hist['name'] = name
            hist_list.append(hist)
    else:
        if name in names:
            hist = run.history(keys=['iter', y_name])
            hist['name'] = name
            hist_list.append(hist)
        
        
df = pd.concat(hist_list, ignore_index=True)
df = df.query(f"`{y_name}` != 'NaN'")
df["name"] = df["name"].map(name_map)
df["iter"] = (df["iter"] / 1000).astype("int32")
min = df[y_name].min()

plt.ylim(min - 0.05, 3.5)
ax = sns.lineplot(x="iter", y=y_name, hue="name", hue_order=hue_order, data=df, linewidth=2)
ax.get_legend().set_title("")

# set size
plt.xticks(fontsize=10)
plt.xlabel(x_name_label, fontsize=12)
plt.yticks(fontsize=10)
plt.ylabel(y_name_label, fontsize=12)
plt.legend(fontsize=12)
plt.title(label=title, fontsize=15)

plt.savefig(f"{folder}/{output_name}.pdf", bbox_inches='tight')
plt.savefig(f"{folder}/{output_name}.jpg", bbox_inches='tight')