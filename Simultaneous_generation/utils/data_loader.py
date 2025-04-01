import os
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset
from utils.fashion_mnist import MNIST, FashionMNIST


class JSRTDataset(Dataset):

    def __init__(self, root, transform=None, maskExt="gif"):

        self.root_dir = root
        self.transform = transform
        self.imgs = os.listdir(os.path.join(root, "Images"))
        self.masks = os.listdir(os.path.join(root, "Masks"))
        self.maskExt = maskExt

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mask_path = os.path.join(self.root_dir, "Masks", self.masks[idx][:-3] + 'gif')
        mask = io.imread(mask_path)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis = 2)

        img_path = os.path.join(self.root_dir, "Images", self.imgs[idx])
        image = io.imread(img_path)
        if len(mask.shape) < 3:
            image = np.expand_dims(image, axis = 2)

        zer = np.zeros_like(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            zer = self.transform(zer)
        
        return torch.concatenate((image, mask, zer), axis=0), torch.tensor([1])
    

def get_data_loader(args):

    if args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = MNIST(root=args.dataroot, train=True, download=args.download, transform=trans)

    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)

    elif args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=args.dataroot, split='train', download=args.download, transform=trans)

    elif args.dataset == 'JSRT':
        image_size = args.image_size
        train_dataset = JSRTDataset(root=args.dataroot,
                                transform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5)),
                                ]), maskExt="png")
    

    # Check if everything is ok with loading datasets
    assert train_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader
