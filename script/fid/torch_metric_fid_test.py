# credit to: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

# from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetric_fid import FrechetInceptionDistance
import pathlib
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

batch_size = 256
batch_size = 50
batch_size = 128
num_workers = 4


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img



IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

device = torch.cuda.current_device()
fid = FrechetInceptionDistance().to(device)
fid.eval()

data_path1 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/val/"
data_path2 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/llamagen/"
# tensor(0.2462, device='cuda:0', dtype=torch.float64) tensor(0.0045, device='cuda:0', dtype=torch.float64) tensor(0.2584, device='cuda:0', dtype=torch.float64) tensor(0.0046, device='cuda:0', dtype=torch.float64)
# tensor(2.1702, device='cuda:0')

# use torch-fid model
# tensor(0.2462, device='cuda:0', dtype=torch.float64) tensor(0.0045, device='cuda:0', dtype=torch.float64) tensor(0.2584, device='cuda:0', dtype=torch.float64) tensor(0.0046, device='cuda:0', dtype=torch.float64)
# tensor(2.1754, device='cuda:0')

# 2 images
# data_path1 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/fid_debug/test1"
# data_path2 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/fid_debug/test2"
# (fix code)
# tensor(0.2921, device='cuda:0', dtype=torch.float64) tensor(0.0025, device='cuda:0', dtype=torch.float64) tensor(0.2812, device='cuda:0', dtype=torch.float64) tensor(0.0174, device='cuda:0', dtype=torch.float64)
# tensor(154.3403, device='cuda:0')

# use torch-fid model
# tensor(0.3030, device='cuda:0', dtype=torch.float64) tensor(1.9601e-05, device='cuda:0', dtype=torch.float64) tensor(0.3030, device='cuda:0', dtype=torch.float64) tensor(0.0006, device='cuda:0', dtype=torch.float64)
# tensor(68.5429, device='cuda:0')

# real
path1 = pathlib.Path(data_path1)
files1 = sorted(
    [file for ext in IMAGE_EXTENSIONS for file in path1.glob("*.{}".format(ext))]
)

dataset1 = ImagePathDataset(files1, transforms=TF.ToTensor())
dataloader1 = torch.utils.data.DataLoader(
    dataset1,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

for batch in tqdm(dataloader1):
    # batch = (batch * 255).to(dtype=torch.uint8, device=device)
    # torch fid
    batch = batch.to(device=device)
    fid.update(batch, real=True)


path2 = pathlib.Path(data_path2)
files2 = sorted(
    [file for ext in IMAGE_EXTENSIONS for file in path2.glob("*.{}".format(ext))]
)

dataset2 = ImagePathDataset(files2, transforms=TF.ToTensor())
dataloader2 = torch.utils.data.DataLoader(
    dataset2,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

for batch in tqdm(dataloader2):
    # batch = (batch * 255).to(dtype=torch.uint8, device=device)
    # torch fid
    batch = batch.to(device=device)
    fid.update(batch, real=False)
    
print(fid.compute())

