from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import U_Net, TB, ATT_U, NestedUNet, Symm, weight_init

class stnU(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.stn = stn()
        self.U = U_Net(feature_ch=args.in_ch, dilation=args.use_dilation, dsv=args.use_dsv,
                               res=args.use_res, use_sa=args.use_sa, use_ca=args.use_ca, use_edge=self.use_edge)
    
    def forward(self, x):
        d = stn(x)
        x = d['img']
        mat = d['matrix']
        x = self.U(x)
        return {'img': x, 'matrix': mat}

class stn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.localization = nn.Sequential(
            # 64 * 64 -> 58 * 58
            nn.Conv2d(1, 8, kernel_size=7),
            # -> 29 * 29
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            # -> 25 * 25
            nn.Conv2d(8, 10, kernel_size=5),
            # -> 12 * 12
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            # -> 10 * 10
            nn.Conv2d(10, 10, 3),
            # -> 5 * 5
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # self.fc_loc = nn.Sequential(
        #     nn.Linear(10 * 5 * 5, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 3 * 2)
        # )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 5 * 5, 64),
            nn.ReLU(True),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 5 * 5)
        theta = self.fc_loc(xs)
        # theta = theta.view(-1, 2, 3)
        # print(theta[:, 0:1])
        # theta[:, 0:1] = 10 * theta[:, 0:1]
        Cos = torch.cos(theta[:, 0:1])
        Sin = torch.sin(theta[:, 0:1])
        rot = torch.cat((torch.cat((Cos, -Sin), -1).unsqueeze(1), torch.cat((Sin, Cos), -1).unsqueeze(1)), 1)
        theta = torch.cat((rot, theta[:, 1:].unsqueeze(-1)), -1)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return {'img': x, 'matrix': theta}