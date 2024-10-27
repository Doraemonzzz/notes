# python -m pytorch_fid /mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/val/ /mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/llamagen/
import torch
from fid_score import calculate_fid_given_paths

path1 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/val/"
path2 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/llamagen/"
# 0.24619678141195786 0.00445203216249807 0.2583887024936059 0.004604678618398146
# 2.1753721483866

path1 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/fid_debug/test1"
path2 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/fid_debug/test2"
# 0.29213035127752995 0.0017797296800605865 0.28205998160980084 0.017178276014185614

path = [path1, path2]
batch_size = 128
device = torch.cuda.current_device()
dims = 2048
num_workers = 4

fid = calculate_fid_given_paths(path, batch_size, device, dims, num_workers)
print(fid)

