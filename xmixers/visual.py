import wandb
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_CONFIG_DIR"] = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment/nanogpt/wandb"

sns.set_theme()

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("doraemonzzz/nanogpt")
hist_list = []

##### v1
# method1
keyword = "v1_method1"
# y_name = 'val/loss'
# y_name_label = "Validation Loss"
y_name = 'train/loss'
y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_gpt2_small_50b_v1_method1_layerinit": "GPT-baseline",
    "train_xmixers_gpt_small_lpe_50b_v1_method1": "GPT",
    "train_xmixers_gpt_small_50b_v1_method1": "GPT-sincos",
    "train_xmixers_llama_small_50b_v1_method1": "LLaMA",
}
hue_order = ["GPT-baseline", "GPT", "GPT-sincos", "LLaMA"]
title = "Language Model(124M) loss curve"
output_name = f"{keyword}_{y_name}".replace("/", "_")
folder = "v1"

# method2
keyword = "layerinit"
# y_name = 'val/loss'
# y_name_label = "Validation Loss"
# y_name = 'train/loss'
# y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_gpt2_small_50b_v1_method1_layerinit": "GPT-baseline",
    "train_xmixers_gpt_small_lpe_50b_add_layerinit": "GPT",
    "train_xmixers_gpt_small_50b_add_layerinit": "GPT-sincos",
    "train_xmixers_llama_small_50b_add_layerinit": "LLaMA",
}
hue_order = ["GPT-baseline", "GPT", "GPT-sincos", "LLaMA"]
title = "Language Model(124M) loss curve"
output_name = f"{keyword}_{y_name}".replace("/", "_")
folder = "v1"

# method 3
keyword = None
names = ["train_gpt2_small_50b_v1_method1_layerinit", "llama_train_xmixers_llama_small_50b_init1", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1", "gpt_train_xmixers_gpt_small_50b_init1"]
y_name = 'val/loss'
y_name_label = "Validation Loss"
y_name = 'train/loss'
y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_gpt2_small_50b_v1_method1_layerinit": "GPT-baseline",
    "llama_train_xmixers_llama_small_50b_init1": "LLaMA",
    "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1": "GPT",
    "gpt_train_xmixers_gpt_small_50b_init1": "GPT-sincos",
}
hue_order = ["GPT-baseline", "GPT", "GPT-sincos", "LLaMA"]
title = "Language Model(124M) loss curve"
output_name = f"method3_{y_name}".replace("/", "_")
folder = "v1"

# method 2 Vs 3
keyword = None
names = ["llama_train_xmixers_llama_small_50b_init1", "train_xmixers_llama_small_50b_add_layerinit", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1", "train_xmixers_gpt_small_lpe_50b_add_layerinit"]
y_name = 'val/loss'
y_name_label = "Validation Loss"
# y_name = 'train/loss'
# y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "llama_train_xmixers_llama_small_50b_init1": "LLaMA-method3",
    "train_xmixers_llama_small_50b_add_layerinit": "LLaMA-method2",
    "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1": "GPT-method3",
    "train_xmixers_gpt_small_lpe_50b_add_layerinit": "GPT-method2",
}
hue_order = ["GPT-method2", "GPT-method3", "LLaMA-method2", "LLaMA-method3"]
title = "Language Model(124M) loss curve"
output_name = f"method3_vs_method2_{y_name}".replace("/", "_")
folder = "v1"

# method 4
keyword = None
names = ["train_xmixers_llama_small_50b_add_layerinit", "train_xmixers_llama_small_50b_token_mixer_init1", "train_xmixers_llama_small_50b_token_mixer_init2"]
y_name = 'val/loss'
y_name_label = "Validation Loss"
y_name = 'train/loss'
y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_xmixers_llama_small_50b_token_mixer_init1": "LLaMA-fla",
    "train_xmixers_llama_small_50b_token_mixer_init2": "LLaMA-fairseq",
    "train_xmixers_llama_small_50b_add_layerinit": "LLaMA",
}
hue_order = ["LLaMA", "LLaMA-fla", "LLaMA-fairseq"]
title = "Language Model(124M) loss curve"
output_name = f"method4_{y_name}".replace("/", "_")
folder = "v1"

# 4.2
keyword = None
names = ["llama_train_xmixers_llama_small_50b_init1_token_mixer_init2", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1_token_mixer_init2", "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2", "train_gpt2_small_50b_v1_method1_layerinit"]
y_name = 'val/loss'
y_name_label = "Validation Loss"
# y_name = 'train/loss'
# y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_gpt2_small_50b_v1_method1_layerinit": "GPT-baseline",
    "llama_train_xmixers_llama_small_50b_init1_token_mixer_init2": "LLaMA",
    "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1_token_mixer_init2": "GPT",
    "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2": "GPT-sincos",
}
hue_order = ["GPT-baseline", "GPT", "GPT-sincos", "LLaMA"]
title = "Language Model(124M) loss curve"
output_name = f"method4.2_{y_name}".replace("/", "_")
folder = "v1"

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