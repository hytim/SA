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
import cv2


class create(data.Dataset):
    def __init__(self, args, data_root, label_root, edge_root, fold_i=0, is_train=True, nfold=5):
        self.args = args
        self.is_train = is_train
        self.fold_i = args.fold_i
        if self.is_train:
            start = (self.fold_i + 1) / nfold
            end = (self.fold_i + nfold) / nfold
        else:
            start = self.fold_i / nfold
            end = (self.fold_i + 1) / nfold

        data = [os.path.join(data_root, png)
                for png in os.listdir(data_root)]  # find png
        label = [os.path.join(label_root, png)
                 for png in os.listdir(label_root)]
        edge = [os.path.join(edge_root, png) for png in os.listdir(edge_root)]

        data = sorted(data, key=lambda x : (int(x.split('/')[-1].split('_')[0]),
                                           ord(x.split('/')[-1].split('_')[1]), int(x.split('/')[-1].split('_')[-1].split('.')[0])))
        label = sorted(label, key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                             ord(x.split('/')[-1].split('_')[1]), int(x.split('/')[-1].split('_')[-1].split('.')[0])))
        edge = sorted(edge, key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                           ord(x.split('/')[-1].split('_')[1]), int(x.split('/')[-1].split('_')[-1].split('.')[0])))

        length = len(data)
        data = data + data
        label = label + label
        edge = edge + edge

        self.data = data[round(start * length):round(end * length)]
        self.label = label[round(start * length):round(end * length)]
        self.edge = edge[round(start * length):round(end * length)]

        self.normalize = T.Normalize(mean=[.5], std=[.5])

        self.transforms = T.Compose([
            T.ToTensor(),
        ])

        self.flip = T.RandomHorizontalFlip(p=0.5),

        self.aug = A.OneOf([
            A.augmentations.geometric.rotate.Rotate(limit=60, interpolation=cv2.INTER_NEAREST, p=0.25),
            A.augmentations.transforms.HorizontalFlip(p=0.25),
            A.transforms.RandomBrightnessContrast(p=0.25),
            A.GaussNoise(p=0.25),
        ], p=0.5)

    def __getitem__(self, index):
        data = imageio.imread(self.data[index])
        label = imageio.imread(self.label[index])
        edge = imageio.imread(self.edge[index])
        AngLabel = int(self.data[index].split('/')[-1].split('_')[-1].split('.')[0])

        data, label, dis_map, edge, AngLabel = self.trans(data, label, edge, AngLabel, self.is_train)

        if label.sum() == 0:
            category = 0
        else:
            category = 1

        return {'data': data, 'label': label, 'edge': edge, 'dis_map': dis_map, \
            'category': category, 'filename': self.data[index], 'AngLabel': AngLabel}
    
    def label2dis(self, label: np.ndarray):
        if not label.any():
            return np.zeros_like(label)

        mask = label.astype(np.bool)
        bac2edge = eedt(~mask) * ~mask
        fore2edge = eedt(binary_erosion(mask)) * mask
        return fore2edge + bac2edge

    def trans(self, data, label, edge, AngLabel, is_train=True):
        if is_train:
            defrom = self.aug(image=data, mask=np.concatenate((np.expand_dims(label, 0), np.expand_dims(edge, 0)), 0).transpose(1,2,0))
            data = defrom['image']
            label = defrom['mask'][:,:,0]
            edge = defrom['mask'][:,:,1]
        
        dis_map = self.label2dis(np.array(label))   # distance of pixels to edge
        edge[dis_map<=1] = 255 / 2  # similar to dilate or weak edge
        dis_map = self.transforms(dis_map)
        dis_map = torch.sigmoid(dis_map)

        data = T.ToTensor()(data)
        data[data>(80/255)] = 80/255
        if (data.max() != 0):
            data = data / data.max()
        label = T.ToTensor()(label)
        edge = T.ToTensor()(edge)

        return data, label, dis_map, edge, AngLabel

    def __len__(self):
        return len(self.data)
