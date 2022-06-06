import os
import numpy as np
import torch
import utils.dataset_kfold as dataset_kfold
import utils.dataset_MR as dataset_MR
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import random

import train
import e2e_train
from utils.param_parser import args_parser


def main():
    args = args_parser()
    print(repr(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.DID)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    assert args.mode in [0, 1, 2], 'illegal train mode'
    if args.mode == 0:
        train_data = dataset_kfold.create(args=args, data_root=args.data_root,
                                    label_root=args.label_root, edge_root=args.edge_root, is_train=True)
        val_data = dataset_kfold.create(args=args, data_root=args.data_root,
                                    label_root=args.label_root, edge_root=args.edge_root, is_train=False)
    elif args.mode == 1:
        train_data = dataset_kfold.create(args=args, data_root=args.data_root,
                                    label_root=args.label_root, edge_root=args.edge_root, is_train=True, nfold=args.nfold, fold_i=args.fold_i)
        val_data = dataset_kfold.create(args=args, data_root=args.data_root,
                                    label_root=args.label_root, edge_root=args.edge_root, is_train=False, nfold=args.nfold, fold_i=args.fold_i)
    elif args.mode == 2:
        train_data = dataset_MR.create(args=args, is_train=True, nfold=args.nfold, fold_i=args.fold_i)
        val_data = dataset_MR.create(args=args, is_train=False, nfold=args.nfold, fold_i=args.fold_i)
    train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True,)
    val_dataloader = DataLoader(val_data, 1,)
    if args.use_stn:
        trainer = e2e_train.Trainer(args=args, train_set=train_dataloader, val_set=val_dataloader)
    else:
        trainer = train.Trainer(args=args, train_set=train_dataloader, val_set=val_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
