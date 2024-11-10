import wandb
import pandas as pd
from tqdm import tqdm

# 初始化API并指定项目路径
api = wandb.Api()
project_path = "doraemonzzz/nanogpt"  # 替换为你的 <entity/project-name>

# 获取项目中所有运行
runs = api.runs(project_path)

# # 提取所有运行的日志数据
# summary_list = []   # 存储每个运行的 summary 数据
# config_list = []    # 存储每个运行的 config 数据
# name_list = []      # 存储每个运行的名称

# for run in runs:
#     summary_list.append(run.summary._json_dict)  # 获取 summary
#     config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})  # 获取 config
#     name_list.append(run.name)  # 获取运行名称

# # 创建 DataFrame
# summary_df = pd.DataFrame(summary_list)
# config_df = pd.DataFrame(config_list)
# runs_df = pd.concat([summary_df, config_df], axis=1)
# runs_df['run_name'] = name_list

# print(runs_df)

# # 保存为 CSV 文件
# # runs_df.to_csv("wandb_project_runs.csv", index=False)

# # 保存为 JSON 文件
# runs_df.to_json("wandb_project_runs.json", orient="records")


# 存储所有历史数据
all_history = []

# 提取每个运行的 history 并添加到列表
for run in tqdm(runs):
    history = run.history(keys=["iter", "val/loss", "train/loss"])
    history["run_id"] = run.id  # 添加运行 ID 列，方便区分
    history["name"] = run.name  # 添加运行 ID 列，方便区分
    all_history.append(history)

# 合并所有运行的历史数据
all_history_df = pd.concat(all_history, ignore_index=True)

# 保存为 CSV 文件
all_history_df.to_csv("wandb_all_runs_history.csv", index=False)
print("Saved all runs history to wandb_all_runs_history.csv")
