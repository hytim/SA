import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt


class rotDataSet(Dataset):
    def __init__(self, data_dir='data', type='train'):
        self.data_dir = data_dir
        self.data_list = os.listdir(self.data_dir)
        self.data_list.sort()
        self.type = type
        if self.type == 'train':
            self.start = 0.
            self.end = 0.7
        else:
            self.start = 0.7
            self.end = 1.
        self.data_list = self.data_list[int(self.start*len(self.data_list)):int(self.end*len(self.data_list))]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.data_list[index])

        image = Image.open(path).convert('L')
        image = np.array(image) / 255
        

        angle = np.random.randint(-60, 61)

        # angle = 50
        image = ndimage.rotate(image, angle*1, order=0, reshape=False)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        angle = angle + 60
        # print(angle)
        
        return {'image': image, 'label': angle}


if __name__ == '__main__':  # debug

    dataset = rotDataSet()
    image = dataset[0]['image']
    angle = dataset[0]['angle']

    print(image.shape)
    print(angle)
    print(image.min())

    plt.figure(figsize=(6, 6))

    plt.imshow(image, cmap='gray')

    plt.savefig('test1')
