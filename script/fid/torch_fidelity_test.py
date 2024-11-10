import torch_fidelity


data_path1 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/val/"
data_path2 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/llamagen/"


metrics_dict = torch_fidelity.calculate_metrics(
    input1=data_path1, 
    input2=data_path2, 
    cuda=True, 
    fid=True, 
    verbose=False,
)

print(metrics_dict)