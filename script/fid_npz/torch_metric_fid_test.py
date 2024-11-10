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
# batch_size = 50
# batch_size = 128
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

fid_statistics_file = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment/dit/reference/VIRTUAL_imagenet256_labeled.npz"


device = torch.cuda.current_device()
fid = FrechetInceptionDistance(fid_statistics_file=fid_statistics_file).to(device)
fid.eval()

data_path1 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/sample/val/"
data_path1 = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/elm/images_fid_ar/0250000-linearcfg-1.0-t1.0-g256-top1000"
# tensor(0.2395, device='cuda:0', dtype=torch.float64) tensor(0.0041, device='cuda:0', dtype=torch.float64) tensor(0.2462, device='cuda:0', dtype=torch.float64) tensor(0.0045, device='cuda:0', dtype=torch.float64)
# tensor(1.6757, device='cuda:0')

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
    fid.update(batch, real=False)


print(fid.compute())

