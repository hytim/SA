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
from torchvision.transforms import InterpolationMode
import albumentations as A
import cv2
import random
import glob


class create(data.Dataset):
    def __init__(self, is_train=True, data_root='../ventricle/data/data_MR'):
        self.is_train = is_train
        if self.is_train:
            start = 0.
            end = 0.7
        else:
            start = 0.7
            end = 1.0

        data_f = [os.path.join(data_root, folder, '1')
                for folder in os.listdir(data_root)]  # find png

        length = len(data_f)
        data_f = data_f + data_f

        data_f = data_f[round(start * length):round(end * length)]
        self.data = []
        self.label = []
        self.edge = []
        for folder in data_f:
            self.data = self.data + glob.glob(os.path.join(folder, '*.png'))

        self.normalize = T.Normalize(mean=[.5], std=[.5])

        self.transforms_data = T.Compose([
            T.ToTensor(),
            T.Resize(size=[64,64]),
        ])

        self.aug = A.OneOf([
            A.GaussNoise(p=0.33),
            A.RandomBrightnessContrast(p=0.33),
            A.augmentations.transforms.HorizontalFlip(p=0.33),
        ], p=0.5)

    def __getitem__(self, index):
        data_ori = imageio.imread(self.data[index])
        data = self.trans(data_ori, self.is_train)
        data = self.normalize(data)
        return {'data': data, 'path': self.data[index], 'filename': self.data[index], 'data_ori': T.Resize(size=[384,384])(T.ToTensor()(data_ori))}

    def trans(self, data, is_train=True):
        if is_train:
            defrom = self.aug(image=data)
            data = defrom['image']
        data = self.transforms_data(data)

        return data

    def __len__(self):
        return len(self.data)