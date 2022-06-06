import os
import torch
from torch.utils import data
import scipy.io as scio
from PIL import Image
import imageio
import numpy as np
import re
import torch.nn as nn
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from scipy.ndimage import distance_transform_edt as eedt
from scipy.ndimage import binary_erosion
from torchvision.transforms import transforms
import albumentations as A
import random
import cv2


class create(data.Dataset):
    def __init__(self, data_root, label_root = '../ventricle/label/label' , is_train=True):
        self.is_train = is_train
        if self.is_train:
            start = 0.
            end = 0.8
        else:
            start = 0.
            end = 0.2

        data = [os.path.join(data_root, png)
                for png in os.listdir(data_root)]  # find png
        data = sorted(data, key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                           ord(x.split('/')[-1].split('_')[1]), int(re.split('[_.]', x.split('/')[-1])[2])))
        label = [os.path.join(label_root, png)
                 for png in os.listdir(label_root)]
        label = sorted(label, key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                             ord(x.split('/')[-1].split('_')[1]), int(x.split('/')[-1].split('_')[-1].split('.')[0])))
        length = len(data)
        self.data = data[round(start * length):round(end * length)]
        self.label = label[round(start * length):round(end * length)]

        self.normalize = T.Normalize(mean=[.5], std=[.5])

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((64, 64)),
        ])

        self.aug = A.OneOf([
            A.augmentations.geometric.rotate.Rotate(limit=60, interpolation=cv2.INTER_NEAREST, p=0.25),
            A.augmentations.transforms.HorizontalFlip(p=0.25),
            A.transforms.RandomBrightnessContrast(p=0.25),
            A.GaussNoise(p=0.25),
        ], p=0.5)

    def __getitem__(self, index):
        data_ori = imageio.imread(self.data[index])
        label = imageio.imread(self.label[index])

        data = self.trans(data_ori, self.is_train)

        return {'data': data, 'path': self.data[index], 'filename': self.data[index], 'data_ori': T.ToTensor()(data_ori), 'label': T.ToTensor()(label)}

    def trans(self, data, is_train=True):
        if is_train:
            defrom = self.aug(image=data)
            data = defrom['image']
        
        data = self.transforms(data)
        data[data>(80/255)] = 80/255
        if (data.max() != 0):
            data = data / data.max()

        return data

    def __len__(self):
        return len(self.data)
