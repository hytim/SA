import sys
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from model.stn_model import stn
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, args, train_set):
        super(Trainer, self).__init__()
        self.args = args
        self.train_set = train_set
        self.model = stn()

        self.loss = nn.L1Loss()
        self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.max_epoch)

    def inverse_affine(self, tran_matrix: torch.tensor, img: torch.tensor) -> torch.tensor:
        assert len(tran_matrix.size()) == 3, f'{len(tran_matrix.size())} != 3'
        rot_matrix = tran_matrix[:, :, :2]
        bias = tran_matrix[:, :, 2:3]
        # how to deal with inverse matrix not exist
        inv_rot = torch.linalg.inv(rot_matrix)
        # inv_tran = nn.parameter.Parameter(torch.cat((inv_rot, -inv_rot @ bias), dim=-1), requires_grad=False)
        inv_tran = torch.cat((inv_rot, -inv_rot @ bias), dim=-1)
        # inv_tran[:, :, :2] = inv_rot
        # inv_tran[:, :, 2:3] = -inv_rot @ bias
        grid = F.affine_grid(inv_tran, img.size())
        out = F.grid_sample(img, grid)
        return out

    def train(self):
        patience = self.args.patience
        min_loss1 = 1e9

        for epoch in range(self.args.max_epoch):
            self.model.train()
            l1 = 0.
            l2 = 0.
            l3 = 0.
            l4 = 0.

            for i, (dictionary) in enumerate(self.train_set):
                data = Variable(dictionary['data'].cuda()).float()

                self.optimizer.zero_grad(set_to_none = True)
                out_dict = self.model(data)
                align_img = out_dict['img']
                tran_matrix = out_dict['matrix']
                rot_matrix = tran_matrix[:, :, :2]
                offset_matrix = tran_matrix[:, :, 2:3]

                inv_img = self.inverse_affine(tran_matrix, align_img)   
                loss1 = self.loss(align_img, torch.flip(align_img, dims=[-1]))
                loss2 = self.loss(data, inv_img)
                loss4 = torch.abs(nn.ReLU(True)(torch.abs(offset_matrix) - 0.2 * torch.ones_like(offset_matrix))).mean()
                loss = 10 * loss1 + loss2 + 1e3 * loss4

                loss.backward()
                self.optimizer.step()

                l1 += float(loss1)
                l2 += float(loss2)
                # l3 += float(loss3)
                l4 += float(loss4)

            self.scheduler.step()
            print( 'Epoch[Tra] [%d/%d], Loss1: %.4f, Loss2: %.4f, Loss4: %.4f' % ( epoch + 1, self.args.max_epoch, l1, l2, l4))
            if(l1 < min_loss1):
                min_loss1 = l1
                net = self.model.state_dict()
                save_model_name = r'./models/' + 'stn.ckpt'
                torch.save(net, save_model_name)

if __name__ == '__main__':
    import os
    import utils.dataset_stn as dataset_stn
    import utils.dataset_MR_stn as dataset_MR_stn
    from torch.utils.data import DataLoader
    import random
    from utils.param_parser import args_parser

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
    if args.mode == 0:
        train_data = dataset_stn.create(data_root=args.data_root, is_train=True)
        val_data = dataset_stn.create(data_root=args.data_root, is_train=False)
    else:
        train_data = dataset_MR_stn.create(is_train=True)
        val_data = dataset_MR_stn.create(is_train=False)
    train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True,)
    val_dataloader = DataLoader(val_data, 1,)
    trainer = Trainer(args=args, train_set=train_dataloader)
    trainer.train()
