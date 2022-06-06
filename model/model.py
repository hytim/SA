import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv1(att)
        att = self.sigmoid(att)
        return identity * x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * identity


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, dilation=False, res=False, use_sa=False, use_ca=False):
        super(conv_block, self).__init__()
        if not dilation:
            self.dil = 1
            self.pad = 1
        else:
            self.dil = 2
            self.pad = 2
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=self.pad, dilation=self.dil, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=self.pad, dilation=self.dil, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.downsample = None
        if res:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, 1, 1, True),
                nn.BatchNorm2d(ch_out)
            )
        self.SA = None
        if use_sa:
            self.SA = SpatialAttention()
        self.CA = None
        if use_ca:
            self.CA = ChannelAttention(in_planes=ch_in)

    def forward(self, x):
        if not self.CA is None:
            x = self.CA(x)
        identity = x
        x = self.conv(x)
        if not self.downsample is None:
            identity = self.downsample(identity)
            x = identity + x
        if not self.SA is None:
            x = self.SA(x)
        # if not self.CA is None:
        #     x = self.CA(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in, ch_out, use_interpolation=True):
        super(up_conv,self).__init__()
        upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True) if use_interpolation else \
        nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            upsample,
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DeepSupervision(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.dsv = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, 1),
           nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        out = self.dsv(x)
        return out

class edge_guidance_module1(nn.Module):
    def __init__(self, ch_in, ch_out=1):
        super().__init__()
        self.dsv4 = DeepSupervision(ch_in*8, 4, scale_factor=8)
        self.dsv3 = DeepSupervision(ch_in*4, 4, scale_factor=4)
        self.dsv2 = DeepSupervision(ch_in*2, 4, scale_factor=2)
        self.dsv1 = nn.Conv2d(ch_in, 4, 1)
        self.conv = conv_block(4*4, 4*4)
        self.final_edge = nn.Conv2d(4*4, ch_out, 1)    
    
    def forward(self, x1, x2, x3, x4):
        x = torch.cat((self.dsv4(x4), self.dsv3(x3), self.dsv2(x2), self.dsv1(x1)), dim=1)
        identity = x
        x = self.conv(x)
        x = self.final_edge(identity - x)
        return x

class edge_guidance_module(nn.Module):
    def __init__(self, ch_in, ch_out=1):
        super().__init__()
        self.up1 = up_conv(ch_in=ch_in*2, ch_out=ch_in, use_interpolation=False)
        self.up2 = up_conv(ch_in=ch_in*4, ch_out=ch_in*2, use_interpolation=False)

        self.block2 = conv_block(ch_in*4, ch_in*2)
        self.block1 = conv_block(ch_in*2, ch_in)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x1, x2, x3):
        x = self.block2(torch.cat((self.up2(x3), x2), dim=1))
        x = self.block1(torch.cat((x1, self.up1(x)), dim=1))
        identity = x
        x = self.conv(x)
        x = x - identity
        x = self.out(x)
        return x

class diff_edge(nn.Module):
    def __init__(self, ch_in, ch_out=1):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            # nn.ConvTranspose2d(ch_in, ch_in*2, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = x - identity
        x = self.out(x)
        return x

class weighted_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(weighted_block, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=False),   # 1*1
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, bias=False),  # 1*1
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, bias=False),  # 1*1
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.weight(x1)
        return x1 * x2
        
class weight_aggregation_block(nn.Module):
    def __init__(self, feature_ch=16, out_ch=32):
        super().__init__()
        # in_ch = [256, 128, 64, 32]
        self.conv_edge = nn.Conv2d(6 * feature_ch, out_ch, kernel_size=1, bias=False)
        self.weight1 = weighted_block(feature_ch * 8, out_ch)
        self.weight2 = weighted_block(feature_ch * 4, out_ch)
        self.weight3 = weighted_block(feature_ch * 2, out_ch)
        self.weight4 = weighted_block(feature_ch * 1, out_ch)
        self.out_conv = nn.Conv2d(out_ch * 2, 1, kernel_size=1, bias=False)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, egm, x1, x2, x3, x4):
        egm = self.conv_edge(egm)
        out1 = self.weight1(x1)
        out2 = self.weight2(x2)
        out3 = self.weight3(x3)
        out4 = self.weight4(x4)
        out = torch.cat((self.up(self.up(self.up(out1) + out2) + out3) + out4, egm), dim=1)
        out = self.out_conv(out)
        return out

class U_Net(nn.Module):
    def __init__(self, img_ch=1, feature_ch=32, output_ch=1, dilation=False, dsv=False, res=False, use_sa=False, use_ca=False, use_edge=False):
        super().__init__()
        self.dilation = dilation
        self.use_sa = use_sa
        self.use_ca = use_ca
        self.res = res
        self.dsv = dsv
        self.use_edge = use_edge
        interpolation = True
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(img_ch, feature_ch)
        self.Conv2 = conv_block(feature_ch, feature_ch*2, res=self.res, dilation=self.dilation)
        self.Conv3 = conv_block(feature_ch*2, feature_ch*4, res=self.res, dilation=self.dilation, use_sa=self.use_sa)
        self.Conv4 = conv_block(feature_ch*4, feature_ch*8, res=self.res, dilation=self.dilation, use_sa=self.use_sa)
        self.Conv5 = conv_block(feature_ch*8, feature_ch*16, res=self.res, dilation=self.dilation, use_sa=self.use_sa)

        self.Up4 = up_conv(ch_in=feature_ch*16, ch_out=feature_ch*8, use_interpolation=interpolation)
        self.Up_conv4 = conv_block(ch_in=feature_ch*16, ch_out=feature_ch*8, use_ca=self.use_ca)

        self.Up3 = up_conv(ch_in=feature_ch*8, ch_out=feature_ch*4, use_interpolation=interpolation)
        self.Up_conv3 = conv_block(ch_in=feature_ch*8, ch_out=feature_ch*4, use_ca=self.use_ca)

        self.Up2 = up_conv(ch_in=feature_ch*4, ch_out=feature_ch*2, use_interpolation=interpolation)
        self.Up_conv2 = conv_block(ch_in=feature_ch*4, ch_out=feature_ch*2, use_ca=self.use_ca)

        self.Up1 = up_conv(ch_in=feature_ch*2, ch_out=feature_ch, use_interpolation=interpolation)
        self.Up_conv1 = conv_block(ch_in=feature_ch*2, ch_out=feature_ch, use_ca=self.use_ca)

        self.final = nn.Conv2d(feature_ch, output_ch, kernel_size=1, stride=1, padding=0)

        if self.dsv:
            self.dsv4 = DeepSupervision(feature_ch*8, 4, scale_factor=8)
            self.dsv3 = DeepSupervision(feature_ch*4, 4, scale_factor=4)
            self.dsv2 = DeepSupervision(feature_ch*2, 4, scale_factor=2)
            self.dsv1 = nn.Conv2d(feature_ch, 4, 1)
            self.final = nn.Conv2d(4*4, output_ch, 1)
        
        if self.use_edge:
            self.egm = edge_guidance_module(ch_in=feature_ch)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        if self.use_edge:
            edge = self.egm(x1, x2, x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d4 = self.Up4(x5)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        if self.dsv:
            dsv4 = self.dsv4(d4)
            dsv3 = self.dsv3(d3)
            dsv2 = self.dsv2(d2)
            dsv1 = self.dsv1(d1)
            out = self.final(torch.cat([dsv4, dsv3, dsv2, dsv1], dim=1))
        else:
            out = self.final(d1)

        #classification

        if not self.use_edge:
            return torch.sigmoid(out)
        else:
            return torch.sigmoid(out), torch.sigmoid(edge)

class FCN(nn.Module):
    def __init__(self, img_ch=1,feature_ch=32, output_ch=1):
        super(FCN, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(img_ch, feature_ch)
        self.Conv2 = conv_block(feature_ch, feature_ch*2)
        self.Conv3 = conv_block(feature_ch*2, feature_ch*4)
        self.Conv4 = conv_block(feature_ch*4, feature_ch*8)
        self.Conv5 = conv_block(feature_ch*8, feature_ch*16)
        interpolation = True
        
        self.mpool = nn.Sequential(
            nn.Conv2d(feature_ch*16, feature_ch*8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(feature_ch*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_ch*8,feature_ch*4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(feature_ch*4),
            nn.ReLU(inplace=True)
        )
        
        self.Up5 = up_conv(ch_in=feature_ch*16, ch_out=feature_ch*8, use_interpolation=interpolation)
        self.Up4 = up_conv(ch_in=feature_ch*8, ch_out=feature_ch*4, use_interpolation=interpolation)
        self.Up3 = up_conv(ch_in=feature_ch*4, ch_out=feature_ch*2, use_interpolation=interpolation)
        self.Up2 = up_conv(ch_in=feature_ch*2, ch_out=feature_ch, use_interpolation=interpolation)
        self.up1 = up_conv(ch_in=feature_ch, ch_out=output_ch, use_interpolation=interpolation)
            

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)       
        x1 = self.Maxpool(x1)
        
        x2 = self.Conv2(x1)                
        x2 = self.Maxpool(x2)

        x3 = self.Conv3(x2)        
        x3 = self.Maxpool(x3)
        
        x4 = self.Conv4(x3)        
        x4 = self.Maxpool(x4)

        # x5 = self.Conv5(x4)   
        # x5 = self.Maxpool(x5)    
        
        # decoding        
        # d5 = self.Up5(x5)
        d4 = self.Up4(x4)
        d3 = self.Up3(d4)
        d2 = self.Up2(d3)
        d1 = self.up1(d2)

        #classification

        return torch.sigmoid(d1)


class ATT_U(nn.Module):
    def __init__(self, img_ch=1, feature_ch=32, output_ch=1):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(img_ch, feature_ch)
        self.Conv2 = conv_block(feature_ch, feature_ch*2)
        self.Conv3 = conv_block(feature_ch*2, feature_ch*4)
        self.Conv4 = conv_block(feature_ch*4, feature_ch*8)
        self.Conv5 = conv_block(feature_ch*8, feature_ch*16)
        interpolation = True

        self.Up5 = up_conv(ch_in=feature_ch*16, ch_out=feature_ch*8, use_interpolation=interpolation)
        self.Att5 = Attention_block(F_g=feature_ch*8, F_l=feature_ch*8, F_int=feature_ch*4)
        self.Up_conv5 = conv_block(ch_in=feature_ch*16, ch_out=feature_ch*8)

        self.Up4 = up_conv(ch_in=feature_ch*8, ch_out=feature_ch*4, use_interpolation=interpolation)
        self.Att4 = Attention_block(F_g=feature_ch*4, F_l=feature_ch*4, F_int=feature_ch*2)
        self.Up_conv4 = conv_block(ch_in=feature_ch*8, ch_out=feature_ch*4)

        self.Up3 = up_conv(ch_in=feature_ch*4, ch_out=feature_ch*2, use_interpolation=interpolation)
        self.Att3 = Attention_block(F_g=feature_ch*2, F_l=feature_ch*2, F_int=feature_ch)
        self.Up_conv3 = conv_block(ch_in=feature_ch*4, ch_out=feature_ch*2)

        self.Up2 = up_conv(ch_in=feature_ch*2, ch_out=feature_ch, use_interpolation=interpolation)
        self.Att2 = Attention_block(F_g=feature_ch, F_l=feature_ch, F_int=feature_ch//2)
        self.Up_conv2 = conv_block(ch_in=feature_ch*2, ch_out=feature_ch)

        self.Conv_1x1 = nn.Conv2d(feature_ch, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return torch.sigmoid(d1)

class TB(nn.Module):
    def __init__(self, img_ch=1, feature_ch=32, output_ch=1, dilation=False, dsv=False, res=False, use_sa=False, use_ca=False, use_edge=False):
        super().__init__()
        self.dilation = dilation
        self.use_sa = use_sa
        self.use_ca = use_ca
        self.use_edge = use_edge
        self.res = res
        self.dsv = dsv
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(img_ch, feature_ch)
        self.Conv2 = conv_block(feature_ch, feature_ch*2, res=self.res, dilation=self.dilation)
        # use SA only in high layer for that low layer only extract primary features
        self.Conv3 = conv_block(feature_ch*2, feature_ch*4, res=self.res, dilation=self.dilation, use_sa=self.use_sa)
        self.Conv4 = conv_block(feature_ch*4, feature_ch*8, res=self.res, dilation=self.dilation, use_sa=self.use_sa)
        self.Conv5 = conv_block(feature_ch*8, feature_ch*16, res=self.res, dilation=self.dilation, use_sa=self.use_sa)

        interpolation = True
        self.Up4 = up_conv(ch_in=feature_ch*16, ch_out=feature_ch*8, use_interpolation=interpolation)
        self.Up_conv4 = conv_block(ch_in=feature_ch*16, ch_out=feature_ch*8, use_ca=self.use_ca)

        self.Up3 = up_conv(ch_in=feature_ch*8, ch_out=feature_ch*4, use_interpolation=interpolation)
        self.Up_conv3 = conv_block(ch_in=feature_ch*8, ch_out=feature_ch*4, use_ca=self.use_ca)

        self.Up2 = up_conv(ch_in=feature_ch*4, ch_out=feature_ch*2, use_interpolation=interpolation)
        self.Up_conv2 = conv_block(ch_in=feature_ch*4, ch_out=feature_ch*2, use_ca=self.use_ca)

        self.Up1 = up_conv(ch_in=feature_ch*2, ch_out=feature_ch, use_interpolation=interpolation)
        self.Up_conv1 = conv_block(ch_in=feature_ch*2, ch_out=feature_ch, use_ca=self.use_ca)
        
        self.final = nn.Conv2d(feature_ch, output_ch, kernel_size=1, stride=1, padding=0)
        self.out_conv = conv_block(ch_in=2, ch_out=128)
        self.pred = nn.Conv2d(128, output_ch, kernel_size=1, stride=1, padding=0)
        
        if self.dsv:
        # if self.use_edge:
            self.dsv4 = DeepSupervision(feature_ch*8, 4, scale_factor=8)
            self.dsv3 = DeepSupervision(feature_ch*4, 4, scale_factor=4)
            self.dsv2 = DeepSupervision(feature_ch*2, 4, scale_factor=2)
            self.dsv1 = nn.Conv2d(feature_ch, 4, 1)
            self.final = nn.Conv2d(4*4, output_ch, 1)
    
        if self.use_edge:
            self.egm = edge_guidance_module1(ch_in=feature_ch)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d4 = self.Up4(x5)
        d4 = self.Up_conv4(d4)
        # d4 = self.CA4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)
        # d3 = self.CA3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)
        # d2 = self.CA2(d2)

        d1 = self.Up1(d2)
        d1 = self.Up_conv1(d1)
        # d1 = self.CA1(d1)

        # dsv
        if self.dsv:
            dsv4 = self.dsv4(d4)
            dsv3 = self.dsv3(d3)
            dsv2 = self.dsv2(d2)
            dsv1 = self.dsv1(d1)
            out = self.final(torch.cat([dsv4, dsv3, dsv2, dsv1], dim=1))
            # out = self.final(d1)
        else:
            out = self.final(d1)

        if self.use_edge:
            edge = self.egm(d1, d2, d3, d4)

        if not self.use_edge:
            return torch.sigmoid(out)
        else:
            return torch.sigmoid(out), torch.sigmoid(edge)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    from torchsummary import summary

    model = U_Net(feature_ch=32, numAngle=121, dilation=True, res=True, use_sa=True, use_ca=True, dsv=False, use_edge=False)
    out, alpha = model(torch.randn(2,1,128,128))
    summary(model, [(1, 128, 128)])
    pass