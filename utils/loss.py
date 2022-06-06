import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=0.5, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, input, target):
        pt = input
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt+1e-8) - \
               (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt+1e-8)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, per_image=True):
        super(DiceLoss, self).__init__()
        self.per_image = per_image
    
    def forward(self, output, label, smooth=1):
        
        dice = (2 * torch.sum(output * label, (1, 2, 3)) + smooth) / \
            (torch.sum(output**2, (1, 2, 3)) + torch.sum(label**2, (1, 2, 3)) + smooth)
        dice_loss = 1 - dice
        if self.per_image:
            return torch.mean(dice_loss)
        else:
            return torch.sum(dice_loss)

class EdgeLoss(nn.Module):
    def __init__(self, reduction='elementwise_mean', alpha=0.5):
        super(EdgeLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
    
    def forward(self, prediction_edge, edge, dis_map):
        loss = - self.alpha * edge * torch.log(prediction_edge + 1e-8) - \
            (1-self.alpha) * dis_map * (1 - edge) * torch.log(1 - prediction_edge + 1e-8)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class HematomaLoss(nn.Module):
    def __init__(self, he_threshold=40):
        super().__init__()
        self.he_th = he_threshold

    def forward(self, data, label, prediction):
        hematoma = ((label==1) & (data>((self.he_th/80-0.5)/0.5))).float()  # add window
        filter_label = label * hematoma
        filter_prediction = prediction * hematoma

        smooth = 1
        dice = (2 * torch.sum(filter_prediction * filter_label, (1, 2, 3)) + smooth) / \
            (torch.sum(filter_prediction**2, (1, 2, 3)) + torch.sum(filter_label**2, (1, 2, 3)) + smooth)
        loss = 1 - dice
        loss = torch.mean(loss)

        return loss

class smoothcrossentropy(nn.Module):
    def __init__(self, num_class = 121):
        super().__init__()
        self.weight = torch.tensor([0.05, 0.2, 0.5, 0.2, 0.05]).cuda().float()
        self.num_class = num_class
    
    def forward(self, pred, label):
        weight = torch.zeros(label.size(0), self.num_class + 4).cuda().float()
        for i in range(label.size(0)):
            weight[i, label[i] : label[i] + 5] = self.weight
        label = weight[:, 2 : self.num_class + 2]
        loss = - label * torch.log(pred + 1e-8)
        return loss.mean()

class AngCrossEntropy(nn.Module):
    def __init__(self, num_class = 121):
        super().__init__()
        self.num_class = num_class
    
    def forward(self, pred, label):
        label = one_hot_encoder(label.unsqueeze(1), self.num_class)
        loss = - label * torch.log(pred + 1e-8)
        return loss.mean()

def one_hot_encoder(label, num_class=121):
    one_hot = torch.zeros_like(label.repeat_interleave(num_class, dim=1)).cuda()
    one_hot_label = one_hot.scatter_(1, label.long(), 1)   # label must be interger here
    one_hot_label = one_hot_label.float()

    return one_hot_label

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    l = AngCrossEntropy(5)
    pred = torch.randn(2,5).cuda()
    mask = torch.randint(5,(2,1)).cuda()
    loss = l(pred, mask)
    pass