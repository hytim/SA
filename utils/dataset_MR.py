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
import glob


class create(data.Dataset):
    def __init__(self, args, fold_i, is_train=True, nfold=5, data_root='../ventricle/data/data_MR', label_root='../ventricle/label/label_MR', edge_root='../ventricle/label/edge_MR'):
        self.args = args
        self.is_train = is_train
        if self.is_train:
            start = (fold_i + 1) / nfold
            end = (fold_i + nfold) / nfold
        else:
            start = fold_i / nfold
            end = (fold_i + 1) / nfold

        data_f = [os.path.join(data_root, folder, '1')
                for folder in os.listdir(data_root)]  # find png
        label_f = [os.path.join(label_root, folder, '1')
                 for folder in os.listdir(label_root)]
        edge_f = [os.path.join(edge_root, folder, '1') for folder in os.listdir(edge_root)]

        length = len(data_f)
        data_f = data_f + data_f
        label_f = label_f + label_f
        edge_f = edge_f + edge_f

        data_f = data_f[round(start * length):round(end * length)]
        label_f = label_f[round(start * length):round(end * length)]
        edge_f = edge_f[round(start * length):round(end * length)]
        self.data = []
        self.label = []
        self.edge = []
        for folder in data_f:
            self.data = self.data + glob.glob(os.path.join(folder, '*.png'))
        for folder in label_f:
            self.label = self.label + glob.glob(os.path.join(folder, '*.png'))
        for folder in edge_f:
            self.edge = self.edge + glob.glob(os.path.join(folder, '*.png'))

        self.normalize = T.Normalize(mean=[.5], std=[.5])

        self.transforms_label = T.Compose([
            T.ToTensor(),
            T.Resize(size=[384,384], interpolation=InterpolationMode.NEAREST),
        ])
        self.transforms_data = T.Compose([
            T.ToTensor(),
            T.Resize(size=[384,384]),
        ])

        self.aug = A.OneOf([
            A.GaussNoise(p=0.33),
            A.RandomBrightnessContrast(p=0.33),
            A.augmentations.transforms.HorizontalFlip(p=0.33),
        ], p=0.5)

    def __getitem__(self, index):
        data = imageio.imread(self.data[index])
        label = imageio.imread(self.label[index])
        edge = imageio.imread(self.edge[index])

        data, label, dis_map, edge = self.trans(data, label, edge, self.is_train)
        data = self.normalize(data)

        if label.sum() == 0:
            category = 0
        else:
            category = 1

        return {'data': data, 'label': label, 'edge': edge, 'dis_map': dis_map, \
            'category': category, 'filename': self.data[index]}
    
    def label2dis(self, label: np.ndarray):
        if not label.any():
            return np.zeros_like(label)

        mask = label.astype(np.bool)
        bac2edge = eedt(~mask) * ~mask
        fore2edge = eedt(binary_erosion(mask)) * mask
        return fore2edge + bac2edge

    def trans(self, data, label, edge, is_train=True):
        if is_train:
            defrom = self.aug(image=data, mask=np.concatenate((np.expand_dims(label, 0), np.expand_dims(edge, 0)), 0).transpose(1,2,0))
            # defrom = self.aug(image=data)
            data = defrom['image']
            label = defrom['mask'][:,:,0]
            edge = defrom['mask'][:,:,1]

        dis_map = self.label2dis(np.array(label))   # distance of pixels to edge
        edge[dis_map<=1] = 255  # similar to dilate or weak edge
        dis_map = self.transforms_label(dis_map)
        dis_map = torch.sigmoid(dis_map)

        data = self.transforms_data(data)
        label = self.transforms_label(label)
        edge = self.transforms_label(edge)
        

        return data, label, dis_map, edge

    def __len__(self):
        return len(self.data)
