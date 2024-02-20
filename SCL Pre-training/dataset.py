import torch
import os
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Kadis(Dataset):
    def __init__(self, root):
        super().__init__()
        self.ref_path = os.path.join(root, 'ref_imgs')
        self.dist_path = os.path.join(root, 'dist_imgs')
        self.df = pd.read_csv(os.path.join(root, 'kadis.csv'))
        self.transform = transforms.Compose([
            transforms.RandomCrop(size=224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dist = self.df.iloc[idx, 0]
        ref = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]

        dist = Image.open(os.path.join(self.dist_path, dist))
        ref = Image.open(os.path.join(self.ref_path, ref))

        dist = self.transform(dist)
        ref = self.transform(ref)

        cls_label = torch.zeros(97)
        cls_label[int(label)] = 1

        return dist, ref, cls_label

