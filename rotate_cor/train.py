import sys
import torch
import os
import numpy as np
from torch import optim
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
# from loss import BCEFocalLoss
from model import resnet18, resnet34
from loss import smoothcrossentropy
import dataset
import argparse
import random
from utils.param_parser import args_parser


class Trainer(object):
    def __init__(self, args, train_set, val_set):
        super(Trainer, self).__init__()
        self.args = args
        self.train_set = train_set
        self.val_set = val_set
        self.num_class = 121
        self.model = resnet18(num_classes=self.num_class)

        # self.model.apply(weight_init)
        self.model = self.model.cuda()

        # self.Loss = nn.CrossEntropyLoss()
        self.Loss = smoothcrossentropy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.max_epoch)

    def train(self):
        best_score = 0.5
        count = 0
        patience = 100

        epoch_list = []
        loss_tra_list = []
        acc_tra_list = []
        acc_val_list = []

        for epoch in range(self.args.max_epoch):
            self.model.train()
            l = 0.
            positive = 0
            total = 0
            acc = 0.

            for i, (dictionary) in enumerate(self.train_set):
                data = Variable(dictionary['image'].cuda()).float()
                # data = torch.cat((data, data, data), dim=1).float()
                label = Variable(dictionary['label'].cuda())
                
                data = F.interpolate(data, size = [128,128])

                self.optimizer.zero_grad(set_to_none = True)
                pred = self.model(data)
                
                category = torch.argmax(pred, axis=1)
                # print(label, category)
                loss = self.Loss(pred, one_hot_encoder(label.unsqueeze(1), self.num_class))
                # loss = torch.abs((pred - label)).mean()
                loss.backward()
                self.optimizer.step()

                # pred = pred * 24
                positive += (category==label).sum()
                total += data.size(0)

            l += float(loss)
            acc = positive / total
            score = acc

            self.scheduler.step()

            print( 'Epoch[Tra] [%d/%d], Loss: %.4f, acc: %.4f' % ( epoch + 1, self.args.max_epoch, l, acc*100))
            # print( 'Epoch[Tra] [%d/%d], Loss: %.4f' % ( epoch + 1, self.args.max_epoch, l))

            if score > best_score:
                best_score = score
                best_net = self.model.state_dict()
                print('Best Accuracy : \033[1;35m%.4f\033[0m' % (best_score))
                torch.save(best_net, './model/' + 'model.ckpt')

def one_hot_encoder(label, num_class):
    one_hot = torch.zeros_like(label.repeat_interleave(num_class, dim=1))
    one_hot_label = one_hot.scatter_(1, label.long(), 1)   # label must be interger here
    one_hot_label = one_hot_label.float()

    return one_hot_label

def main():
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.DID)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    # for reproducibility
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    train_data = dataset.rotDataSet(data_dir=args.data_root)
    val_data = dataset.rotDataSet(data_dir=args.data_root,)
    train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True,)
    val_dataloader = DataLoader(val_data, args.batch_size)

    trainer = Trainer(args=args, train_set=train_dataloader, val_set=val_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
