# import wandb
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set_theme()

# method2
keyword = "layerinit"
y_name = 'val/loss'
y_name_label = "Validation Loss"
# y_name = 'train/loss'
# y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_gpt2_small_50b_v1_method1_layerinit": "GPT-baseline",
    "train_xmixers_gpt_small_lpe_50b_add_layerinit": "GPT",
    "train_xmixers_gpt_small_50b_add_layerinit": "GPT-sincos",
    "train_xmixers_llama_small_50b_add_layerinit": "LLaMA",
}
names = list(name_map.keys())
hue_order = ["GPT-baseline", "GPT", "GPT-sincos", "LLaMA"]
title = "Language Model(124M) loss curve"
output_name = f"{keyword}_{y_name}".replace("/", "_")
folder = "v1"

# # method 4
# keyword = None
# names = ["train_xmixers_llama_small_50b_add_layerinit", "train_xmixers_llama_small_50b_token_mixer_init1", "train_xmixers_llama_small_50b_token_mixer_init2"]
# y_name = 'val/loss'
# y_name_label = "Validation Loss"
# y_name = 'train/loss'
# y_name_label = "Training Loss"
# x_name_label = "Number of iterations(k)"
# name_map = {
#     "train_xmixers_llama_small_50b_token_mixer_init1": "LLaMA-fla",
#     "train_xmixers_llama_small_50b_token_mixer_init2": "LLaMA-fairseq",
#     "train_xmixers_llama_small_50b_add_layerinit": "LLaMA",
# }
# hue_order = ["LLaMA", "LLaMA-fla", "LLaMA-fairseq"]
# title = "Language Model(124M) loss curve"
# output_name = f"method4_{y_name}".replace("/", "_")
# folder = "v1"


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
    "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1": "GPT-method3",
    # "gpt_train_xmixers_gpt_small_50b_init1": "GPT-sincos-method3",
}
# hue_order = ["GPT-method3", "GPT-method4.2", "GPT-sincos-method3", "GPT-sincos-method4.2", "LLaMA-method3", "LLaMA-method4.1", "LLaMA-method4.2"]
hue_order = ["GPT-method3", "GPT-sincos-method4.2", "LLaMA-method3", "LLaMA-method4.1", "LLaMA-method4.2"]

title = "Language Model(124M) loss curve"
output_name = f"method4.2_vs_3_{y_name}".replace("/", "_")
folder = "v1"

# # 4.3 medium
# names = ["train_xmixers_llama_medium_50b", "llama_train_xmixers_llama_medium_50b_init1_token_mixer_init2", "train_xmixers_gpt_medium_lpe_50b", "gpt_lpe_train_xmixers_gpt_medium_lpe_50b_init1", "gpt_train_xmixers_gpt_medium_50b_init1_token_mixer_init2"]
# y_name = 'val/loss'
# y_name_label = "Validation Loss"
# # y_name = 'train/loss'
# # y_name_label = "Training Loss"
# x_name_label = "Number of iterations(k)"
# name_map = {
#     "train_xmixers_llama_medium_50b": "LLaMA-method2",
#     "llama_train_xmixers_llama_medium_50b_init1_token_mixer_init2": "LLaMA-method4.2",
#     "train_xmixers_gpt_medium_lpe_50b": "GPT-method2",
#     "gpt_lpe_train_xmixers_gpt_medium_lpe_50b_init1": "GPT-method3",
#     "gpt_train_xmixers_gpt_medium_50b_init1_token_mixer_init2": "GPT-sincos-method4.2"
# }
# hue_order = ["GPT-method2", "GPT-method3", "GPT-sincos-method4.2", "LLaMA-method2", "LLaMA-method4.2",]

# title = "Language Model(350M) loss curve"
# output_name = f"method4_medium_{y_name}".replace("/", "_")
# folder = "v1"

# 4.3 large
names = ["train_xmixers_llama_large_50b", "train_xmixers_gpt_large_lpe_50b", ]
y_name = 'val/loss'
y_name_label = "Validation Loss"
# y_name = 'train/loss'
# y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_xmixers_llama_large_50b": "LLaMA-method2",
    "train_xmixers_gpt_large_lpe_50b": "GPT-method2",
}
hue_order = ["GPT-method2", "LLaMA-method2",]

title = "Language Model(774M) loss curve"
output_name = f"method4_large_{y_name}".replace("/", "_")
folder = "v1"

# # method 5
# keyword = None
# names = ["llama_train_xmixers_llama_small_50b_init1_token_mixer_init2", "llama_train_xmixers_llama_small_50b_init1", "llama_train_xmixers_llama_small_50b_init2_token_mixer_init2", 
#          "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2", "gpt_train_xmixers_gpt_small_50b_init2_token_mixer_init2",
#          "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init2"]
# y_name = 'val/loss'
# y_name_label = "Validation Loss"
# # y_name = 'train/loss'
# # y_name_label = "Training Loss"
# x_name_label = "Number of iterations(k)"
# name_map = {
#     "llama_train_xmixers_llama_small_50b_init1_token_mixer_init2": "LLaMA-method4.2",
#     "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2": "GPT-sincos-method4.2",
    
#     "llama_train_xmixers_llama_small_50b_init1": "LLaMA-method3",
#     "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1": "GPT-method3",

#     "llama_train_xmixers_llama_small_50b_init2_token_mixer_init2": "LLaMA-method5",
#     "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init2": "GPT-method5",
#     "gpt_train_xmixers_gpt_small_50b_init2_token_mixer_init2": "GPT-sincos-method5",
    
# }
# # hue_order = ["GPT-method3", "GPT-method4.2", "GPT-sincos-method3", "GPT-sincos-method4.2", "LLaMA-method3", "LLaMA-method4.1", "LLaMA-method4.2"]
# hue_order = ["GPT-method3", "GPT-method5", "GPT-sincos-method4.2", "GPT-sincos-method5", "LLaMA-method3", "LLaMA-method4.2", "LLaMA-method5"]

# title = "Language Model(124M) loss curve"
# output_name = f"method5_{y_name}".replace("/", "_")
# folder = "v1"

# method 6
keyword = None
names = ["llama_train_xmixers_llama_small_50b_init1_token_mixer_init2_scale", "llama_train_xmixers_llama_small_50b_init1_token_mixer_init2", 
         "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1_scale", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1",
         "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init2_scale", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init2",
         "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2_scale", "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2"]
# names = [
#          "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1_scale", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1",
#          "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init2_scale", "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init2",
# ]
y_name = 'val/loss'
y_name_label = "Validation Loss"
# y_name = 'train/loss'
# y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "llama_train_xmixers_llama_small_50b_init1_token_mixer_init2_scale": "LLaMA-method4.2-scale",
    "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1_scale": "GPT-method3-scale",
    "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2_scale": "GPT-sincos-method4.2-scale",
    "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init2_scale": "GPT-method5-scale",

    "llama_train_xmixers_llama_small_50b_init1_token_mixer_init2": "LLaMA-method4.2",
    "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init1": "GPT-method3",
    "gpt_train_xmixers_gpt_small_50b_init1_token_mixer_init2": "GPT-sincos-method4.2",
    "gpt_lpe_train_xmixers_gpt_small_lpe_50b_init2": "GPT-method5",
}
# hue_order = ["GPT-method3", "GPT-method4.2", "GPT-sincos-method3", "GPT-sincos-method4.2", "LLaMA-method3", "LLaMA-method4.1", "LLaMA-method4.2"]
hue_order = ["GPT-method3", "GPT-method3-scale", "GPT-method5", "GPT-method5-scale", "GPT-sincos-method4.2", "GPT-sincos-method4.2-scale", "LLaMA-method4.2", "LLaMA-method4.2-scale",]

title = "Language Model(124M) loss curve"
output_name = f"method6_{y_name}".replace("/", "_")
folder = "v1"

# medium
# 6 medium
names = ["train_xmixers_llama_medium_50b", "llama_train_xmixers_llama_medium_50b_init1_token_mixer_init2", 
         "train_xmixers_gpt_medium_lpe_50b", 
        #  "gpt_lpe_train_xmixers_gpt_medium_lpe_50b_init1", 
         "gpt_lpe_train_xmixers_gpt_medium_lpe_50b_init2_scale", 
         "gpt_lpe_train_xmixers_gpt_medium_lpe_50b_init1_scale",
        #  "gpt_train_xmixers_gpt_medium_50b_init1_token_mixer_init2",
         "gpt_train_xmixers_gpt_medium_50b_init1_token_mixer_init2_scale"]
y_name = 'val/loss'
y_name_label = "Validation Loss"
# y_name = 'train/loss'
# y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_xmixers_llama_medium_50b": "LLaMA-method2",
    "llama_train_xmixers_llama_medium_50b_init1_token_mixer_init2": "LLaMA-method4.2",
    "train_xmixers_gpt_medium_lpe_50b": "GPT-method2",
    # "gpt_lpe_train_xmixers_gpt_medium_lpe_50b_init1": "GPT-method3",
    "gpt_lpe_train_xmixers_gpt_medium_lpe_50b_init1_scale": "GPT-method3-scale",
    "gpt_lpe_train_xmixers_gpt_medium_lpe_50b_init2_scale": "GPT-method5-scale",
    # "gpt_train_xmixers_gpt_medium_50b_init1_token_mixer_init2": "GPT-sincos-method4.2",
    "gpt_train_xmixers_gpt_medium_50b_init1_token_mixer_init2_scale": "GPT-sincos-method4.2-scale",
}
hue_order = ["GPT-method2", "GPT-method3-scale", "GPT-method5-scale", "GPT-sincos-method4.2-scale", "LLaMA-method2", "LLaMA-method4.2",]

title = "Language Model(350M) loss curve"
output_name = f"method6_medium_{y_name}".replace("/", "_")
folder = "v1"

# large
names = ["train_xmixers_llama_large_50b", 
         "train_xmixers_gpt_large_lpe_50b", 
         "gpt_lpe_train_xmixers_gpt_large_lpe_50b_init2_scale",
         "gpt_train_xmixers_gpt_large_50b_init1_token_mixer_init2_scale",]
y_name = 'val/loss'
y_name_label = "Validation Loss"
y_name = 'train/loss'
y_name_label = "Training Loss"
x_name_label = "Number of iterations(k)"
name_map = {
    "train_xmixers_llama_large_50b": "LLaMA-method2",
    "train_xmixers_gpt_large_lpe_50b": "GPT-method2",
    "gpt_lpe_train_xmixers_gpt_large_lpe_50b_init2_scale": "GPT-method5-scale",
    "gpt_train_xmixers_gpt_large_50b_init1_token_mixer_init2_scale": "GPT-sincos-method4.2-scale",

}
hue_order = ["GPT-method2", "GPT-method5-scale", "GPT-sincos-method4.2-scale", "LLaMA-method2",]

title = "Language Model(774M) loss curve"
output_name = f"method6_large_{y_name}".replace("/", "_")
folder = "v1"


os.makedirs(folder, exist_ok=True)
        
# 读取历史记录的 CSV 文件
df = pd.read_csv("wandb_all_runs_history.csv")
# 根据 target_run_ids 列表进行过滤
df = df[df["name"].isin(names)]

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
# plt.legend(fontsize=12)
plt.legend(fontsize=10)
plt.title(label=title, fontsize=15)

plt.savefig(f"{folder}/{output_name}.pdf", bbox_inches='tight')
plt.savefig(f"{folder}/{output_name}.jpg", bbox_inches='tight')