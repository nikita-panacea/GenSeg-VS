import os
import torch
import torch.utils.data as data_utils
import torchvision.datasets as dset
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from skimage import io
from models.wgan_gradient_penalty import WGAN_GP
from utils.config import parse_args1
from PIL import Image
import numpy as np


class ImageMaskDataset(Dataset):

    def __init__(self, root, transform=None, maskExt="jpg"):

        self.root_dir = root
        self.transform = transform
        self.imgs = os.listdir(os.path.join(root, "Images"))[:50]
        self.maskExt = maskExt

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mask_path = os.path.join(self.root_dir, "Masks", self.imgs[idx])
        # mask = io.imread(mask_path)
        mask = np.array(Image.open(mask_path).convert('L'))
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis = 2)
        else:
            mask = mask[:,:,0]
            mask = np.expand_dims(mask, axis = 2)

        img_path = os.path.join(self.root_dir, "Images", self.imgs[idx])
        # image = io.imread(img_path)
        image = np.array(Image.open(img_path))
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis = 2)
        # print(image.shape, mask.shape)
        
        if image.shape[2] == 1:
            image = image.repeat(3, axis=2)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
       
        return torch.concatenate((image, mask), axis=0)
    


image_size = (256, 256)
dataroot = "/data2/li/workspace/data/foot-ulcer/train"
train_dataset = ImageMaskDataset(root=dataroot,
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]), maskExt='png')
    

args = parse_args1()
model = WGAN_GP(args)

train_dataloader = data_utils.DataLoader(train_dataset, batch_size=2, shuffle=True)


model.train(train_dataloader)
for i in train_dataloader:
    print(i.shape)
    break