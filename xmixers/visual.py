import wandb
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set_theme()

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("doraemonzzz/nanogpt")
hist_list = []

keyword = "v1_method1"
y_name = 'val/loss'
y_name_label = "Validation Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_gpt2_small_50b_v1_method1": "GPT-baseline",
    "train_xmixers_gpt_small_lpe_50b_v1_method1": "GPT",
    "train_xmixers_gpt_small_50b_v1_method1": "GPT-sincos",
    "train_xmixers_llama_small_50b_v1_method1": "LLaMA",
}
hue_order = ["GPT-baseline", "GPT", "GPT-sincos", "LLaMA"]
title = "Language Model(124M) loss curve"
output_name = f"{keyword}_{y_name}.pdf".replace("/", "_")

for run in runs: 
    if not y_name in run.summary:
        continue
    # print(run.name, run.id)
    # print(run.config)
    # name = run.config['model']['_target_'].split('.')[-1]
    name = run.name
    if keyword in name:
        # print(run.history())
        hist = run.history(keys=['iter', y_name])
        hist['name'] = name
        # print(hist)
        hist_list.append(hist)
        
    # hist = run.history(keys=['iter', y_name])
    # hist['name'] = name
    # # print(hist)
    # hist_list.append(hist)
    # break
        
        
df = pd.concat(hist_list, ignore_index=True)
df = df.query(f"`{y_name}` != 'NaN'")
df["name"] = df["name"].map(name_map)
df["iter"] = (df["iter"] / 1000).astype("int32")
min = df[y_name].min()

plt.ylim(min - 0.05, 3.5)
ax = sns.lineplot(x="iter", y=y_name, hue="name", data=df, linewidth=2)
ax.get_legend().set_title("")

# set size
plt.xticks(fontsize=10)
plt.xlabel(x_name_label, fontsize=12)
plt.yticks(fontsize=10)
plt.ylabel(y_name_label, fontsize=12)
plt.legend(fontsize=12)
plt.title(label=title, fontsize=15)

plt.savefig(f"{output_name}", bbox_inches='tight')