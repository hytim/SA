import sys
import time
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from model.model import U_Net, TB, ATT_U, weight_init
from model.stn_model import stn
from model.ca_net import CA_Net
from model.bio_net import BiONet
from utils.evaluation import *
from utils.hausdorff_distance import get_HausdorffDistance
from utils.mean_surface_distance import get_MSD
from utils.loss import BCEFocalLoss, DiceLoss, EdgeLoss, HematomaLoss
import os
import time
import warnings
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, args, train_set, val_set):
        super(Trainer, self).__init__()
        self.args = args
        self.train_set = train_set
        self.val_set = val_set
        self.use_edge = args.use_edge
        self.modelname = args.model
        self.sym_co = args.sym_co
        self.local_rank = args.local_rank
        if self.modelname == 'U':
            self.model = U_Net(feature_ch=args.in_ch, dilation=args.use_dilation, dsv=args.use_dsv,
                               res=args.use_res, use_sa=args.use_sa, use_ca=args.use_ca, use_edge=self.use_edge)
        elif self.modelname == 'TB' or self.modelname == 'Align':
            self.model = TB(feature_ch=args.in_ch, dilation=args.use_dilation, res=args.use_res,
                            use_sa=args.use_sa, use_ca=args.use_ca, dsv=args.use_dsv, use_edge=self.use_edge)
        elif self.modelname == 'ATT':
            self.model = ATT_U(feature_ch=args.in_ch)
        elif self.modelname == 'CA':
            self.model = CA_Net()
        elif self.modelname == 'BiO':
            self.model = BiONet()
        else:
            print('unknown model')
            sys.exit()

        self.model = self.model.cuda()
        if torch.cuda.device_count() >= 2:
            self.model = nn.DataParallel(self.model)

        self.FLoss = BCEFocalLoss(alpha=args.wce)
        self.HLoss = HematomaLoss()
        self.DLoss = DiceLoss()
        self.ELoss = EdgeLoss(alpha=args.wedge)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        assert args.lr_schedule in [0, 1, 2, 3], 'illegal lr_schedule'
        if args.lr_schedule == 0:
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.max_epoch)
        elif args.lr_schedule == 1:
            self.scheduler = lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.98)
        elif args.lr_schedule == 2:
            self.scheduler = lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[20, 40, 80], gamma=0.1)
        else:
            self.scheduler = None

    def train(self):
        print('start training of fold %d'%(self.args.fold_i))
        best_score = 0.1
        best_epoch = 0
        count = 0
        patience = self.args.patience
        pre_edge = None
        threshold = 0.15

        epoch_list = []
        dice_tra_list = []
        dice_val_list = []
        Floss_tra_list = []
        Floss_val_list = []
        Dloss_tra_list = []
        Dloss_val_list = []
        Fealoss_tra_list = []
        Fealoss_val_list = []
        Eloss_tra_list = []
        Eloss_val_list = []
        se_tra_list = []
        se_val_list = []
        He_se_tra_list = []
        He_se_val_list = []
        pc_tra_list = []
        pc_val_list = []
        sp_tra_list = []
        sp_val_list =[]
        hdd_val_list = []
        msd_val_list = []
        best_net = None
        today = str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday)
        model_save_root = os.path.join('./model/', today)
        result_save_root = os.path.join('./result/', today)


        for epoch in range(self.args.max_epoch):
            self.model.train()
            l1 = 0.
            l2 = 0.
            l3 = 0.
            l4 = 0.
            l5 = 0.
            acc = 0.
            SE = 0.
            SP = 0.
            PC = 0.
            DC = 0.
            He_SE = 0.
            length = 0

            start_time = time.time()
            for i, (data, label, edge, dis_map, category, _) in enumerate(self.train_set):
                loss = 0.
                data = Variable(data.cuda()).to(self.local_rank)
                GT = Variable(label.cuda()).to(self.local_rank)
                dis_map = Variable(dis_map.cuda()).to(self.local_rank)
                edge = Variable(edge.cuda()).to(self.local_rank)
                category = Variable(category.cuda()).float()

                self.optimizer.zero_grad(set_to_none=True)
                if self.args.use_sym:
                    data = torch.cat((data, torch.flip(data, dims=[-1])), dim=0)
                    GT = torch.cat((GT, torch.flip(GT, dims=[-1])), dim=0)
                    edge = torch.cat((edge, torch.flip(edge, dims=[-1])), dim=0)
                    dis_map = torch.cat((dis_map, torch.flip(dis_map, dims=[-1])), dim=0)
                if self.use_edge:
                    output, pre_edge = self.model(data)
                else:
                    output = self.model(data)                   
                
                l1 += float(loss1)
                l2 += float(loss2)
                l3 += float(loss3)

                loss += self.args.weight[0]*loss1 + self.args.weight[1]*loss2 + self.args.weight[2]*loss3

                if self.args.use_sym:
                    batch = int(data.size(0)/2)
                    if self.sym_co:
                        symm_coeff = 1 - torch.sum(abs(GT[:batch,:,:,:] - GT[batch:,:,:,:]), dim=(1,2,3)) / torch.sum(abs(GT[:batch,:,:,:] + GT[batch:,:,:,:]), dim=(1,2,3))
                        loss5 = torch.abs(symm_coeff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (output[:batch,:,:,:] - output[batch:,:,:,:])).mean()
                    else:
                        loss5 = torch.abs(output[:batch,:,:,:] - output[batch:,:,:,:]).mean()
                    l5 += float(loss5)
                    loss += self.args.weight[4]*loss5

                if pre_edge != None:
                    loss4 = self.ELoss(pre_edge, edge, dis_map)
                    l4 += float(loss4)
                    loss += self.args.weight[3]*loss4
                
                loss.backward()
                self.optimizer.step()

                if torch.sum(label) != 0:
                    acc += get_accuracy(output, GT)
                    SE += get_sensitivity(output, GT)
                    SP += get_specificity(output, GT)
                    PC += get_precision(output, GT)
                    DC += get_DC(output, GT)
                    He_SE += get_He_sensitivity(output, GT, data)
                    length += 1

            end_time = time.time()
            print((end_time-start_time)/60)
            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            DC = DC / length
            He_SE = He_SE / length

            if self.scheduler != None:
                self.scheduler.step()

            print(
                'Epoch [%d/%d], CELoss: %.4f, DLoss: %.4f, ELoss: %.4f, FeaLoss: %.4f \n[Training] Acc: %.4f, SE: %.4f, He_SE: %.4f, SP: %.4f, PC: %.4f, DC: %.4f' % (
                    epoch + 1, self.args.max_epoch, l1, l2, l4, l5,
                    acc, SE, He_SE, SP, PC, DC))

            epoch_list.append(epoch)
            Floss_tra_list.append(l1)
            Dloss_tra_list.append(l2)
            Fealoss_tra_list.append(l5)
            Eloss_tra_list.append(l4)
            dice_tra_list.append(DC.item())
            se_tra_list.append(SE.item())
            He_se_tra_list.append(He_SE.item())
            pc_tra_list.append(PC.item())
            sp_tra_list.append(SP.item())

            self.model.eval()
            l1 = 0.
            l2 = 0.
            l3 = 0.
            l4 = 0.
            acc = 0.
            SE = 0.
            SP = 0.
            PC = 0.
            DC = 0.
            He_SE = 0.
            HDD = 0.
            MSD = 0.
            length = 0

            with torch.no_grad():
                HDD_l, MSD_l, ACC_l, SE_l, SE_H_l, PC_l, SP_l, DSC_l = [], [], [], [], [], [], [], []
                pa_last = 0
                ch_cur = 'A'
                ch_last = 'A'
                for i, (data, label, edge, dis_map, category, filename) in enumerate(self.val_set):
                    data = Variable(data.cuda()).to(self.local_rank)
                    label = Variable(label.cuda()).to(self.local_rank)
                    edge = Variable(edge.cuda()).to(self.local_rank)
                    dis_map = Variable(dis_map.cuda()).to(self.local_rank)
                    if self.args.mode == 2:
                        pa_cur = filename[0].split('/')[-3]
                    else:
                        pa_cur = int(filename[0].split('/')[-1].split('_')[0])
                        ch_cur = filename[0].split('/')[-1].split('_')[1]
                    if pa_cur != pa_last or ch_cur != ch_last or (i == (self.val_set.__len__() - 1)):
                        if i == (self.val_set.__len__() - 1):
                            if self.use_edge:
                                output, pre_edge = self.model(data)
                            else:
                                output = self.model(data)
                            output = output > 0.5
                            label = label > 0.5

                            TP += torch.sum((output == 1) & (label == 1))
                            TN += torch.sum((output == 0) & (label == 0))
                            FN += torch.sum((output == 0) & (label == 1))
                            FP += torch.sum((output == 1) & (label == 0))
                            hematoma = ((label==1) & (data>(40/80)))
                            filter_label = label * hematoma
                            filter_prediction = output * hematoma
                            TP_H += torch.sum((filter_prediction == 1) & (filter_label == 1))
                            FN_H += torch.sum((filter_prediction == 0) & (filter_label == 1))

                            HDD += get_HausdorffDistance(output, label)
                            MSD += get_MSD(output, label, symmetric=True)
                            length += 1
                        if (pa_last != 0):
                            HDD = HDD / length
                            MSD = MSD /length
                            ACC = (TP + TN) / (TP + TN + FP + FN)
                            SE = TP / (TP + FN)
                            SE_H = TP_H / (TP_H + FN_H)
                            PC = TP / (TP + FP)
                            SP = TN / (TN + FP)
                            DSC = (2 * SE * PC) / (SE + PC)
                            HDD_l.append(HDD)
                            MSD_l.append(MSD)
                            ACC_l.append(ACC.item())
                            SE_l.append(SE.item())
                            if (filter_label.sum() / label.sum()) > threshold:
                                SE_H_l.append(SE_H.item())
                            PC_l.append(PC.item())
                            SP_l.append(SP.item())
                            DSC_l.append(DSC.item())

                        pa_last = pa_cur
                        ch_last = ch_cur
                        TP = TN = FP = FN = HDD = MSD = length = 0
                        TP_H = FN_H = 0

                    if self.use_edge:
                        output, pre_edge = self.model(data)
                    else:
                        output = self.model(data)

                    loss1 = self.FLoss(output, label.float()) 
                    loss2 = self.DLoss(output, label.float()) 
                    loss3 = self.HLoss(data, label.float(), output)

                    l1 += float(loss1)
                    l2 += float(loss2)
                    l3 += float(loss3)
                    if pre_edge != None:
                        loss4 = self.ELoss(pre_edge, edge, dis_map)
                        l4 += float(loss4)
                    
                    output = output > 0.5
                    label = label > 0.5

                    TP += torch.sum((output == 1) & (label == 1))
                    TN += torch.sum((output == 0) & (label == 0))
                    FN += torch.sum((output == 0) & (label == 1))
                    FP += torch.sum((output == 1) & (label == 0))
                    hematoma = ((label==1) & (data>((40/80-0.5)/0.5)))
                    filter_label = label * hematoma
                    filter_prediction = output * hematoma
                    TP_H += torch.sum((filter_prediction == 1) & (filter_label == 1))
                    FN_H += torch.sum((filter_prediction == 0) & (filter_label == 1))

                    HDD += get_HausdorffDistance(output, label)
                    MSD += get_MSD(output, label, symmetric=True)
                    length += 1

                score = np.mean(DSC_l)

                print(
                    'Epoch [%d/%d], CELoss: %.4f, DLoss: %.4f, HLoss: %.4f, ELoss: %.4f \n[val] ACC: %.4f, SE: %.4f, SE_H: %.4f, SP: %.4f, PC: %.4f, HD: %.4f, ASD: %.4f, DSC: %.4f' % (
                        epoch + 1, self.args.max_epoch, l1, l2, l3, l4,
                        np.mean(ACC_l), np.mean(SE_l), np.mean(
                            SE_H_l), np.mean(SP_l),
                        np.mean(PC_l), np.mean(HDD_l), np.mean(MSD_l), np.mean(DSC_l),))


                Floss_val_list.append(l1)
                Dloss_val_list.append(l2)
                Fealoss_val_list.append(l5)
                Eloss_val_list.append(l4)
                dice_val_list.append(np.mean(DSC_l))
                se_val_list.append(np.mean(SE_l))
                He_se_val_list.append(np.mean(SE_H_l))
                pc_val_list.append(np.mean(PC_l))
                sp_val_list.append(np.mean(SP_l))
                hdd_val_list.append(np.mean(HDD_l))
                msd_val_list.append(np.mean(MSD_l))
                
                if score.item() > best_score:
                    best_score = score.item()
                    best_net = self.model.state_dict()
                    best_epoch = epoch + 1
                    
                    if not os.path.exists(model_save_root):
                        os.mkdir(model_save_root)
                    save_model_name = self.generate_filename(best_score, False) + '.ckpt'
                    torch.save(best_net, os.path.join(model_save_root, save_model_name))
                    
                    print('Best DICE : \033[1;35m%.4f\033[0m' % (best_score))
                    tem_net = self.model.state_dict()
                    torch.save(tem_net, './model/tem_' + self.modelname + '_' + str(self.args.fold_i) + '.ckpt')
                    count = 0
                else:
                    count += 1
                    if count >= patience:
                        break

                output_excel = {'epoch': epoch_list, 'CEloss_tra': Floss_tra_list, 'Dloss_tra': Dloss_tra_list, 'Fealoss_tra': Fealoss_tra_list, 'Eloss_tra': Eloss_tra_list,
                                'dice_tra': dice_tra_list, 'dice_val': dice_val_list, 'se_tra': se_tra_list, 'se_val': se_val_list, 'sp_tra': sp_tra_list, 'sp_val': sp_val_list, 'pc_tra': pc_tra_list, 'pc_val': pc_val_list, 'He_se_tra': He_se_tra_list, 'He_se_val': He_se_val_list,
                                'hd_val': hdd_val_list, 'msd_val': msd_val_list}
                out_excel = pd.DataFrame(output_excel)
                param_excel = pd.DataFrame.from_dict(data=vars(self.args), orient='index')    # convert agrs to dict
                if not os.path.exists(result_save_root):
                    os.mkdir(result_save_root)
                save_xls_name = self.generate_filename(best_score, False) + '.xlsx'
                with pd.ExcelWriter(os.path.join(result_save_root, save_xls_name)) as xlsWriter:
                    out_excel.to_excel(xlsWriter, float_format='%.4f', index=False, sheet_name='metrics')
                    param_excel.to_excel(xlsWriter, float_format='%.4f', index=True, header=False, sheet_name='param')

                
                print('fold %d/%d: current Best DICE of %s: %.4f in %d' % (
                    self.args.fold_i, self.args.nfold, self.modelname, best_score, best_epoch))
                print(
                    '------------------------------------------------------------------------------------------------------------------------------------------')

        print(
            '\n fold %d: Best DICE in entire training of %s: \033[1;35m%.4f\033[0m \n' % (self.args.fold_i, self.modelname, best_score))
        
        former_model_name = self.generate_filename(best_score, False) + '.ckpt'
        latter_model_name = self.generate_filename(best_score, True) + '.ckpt'
        os.rename(os.path.join(model_save_root, former_model_name), os.path.join(model_save_root, latter_model_name))
        former_xls_name = self.generate_filename(best_score, False) + '.xlsx'
        latter_xls_name = self.generate_filename(best_score, True) + '.xlsx'
        os.rename(os.path.join(result_save_root, former_xls_name), os.path.join(result_save_root, latter_xls_name))

        final_net = self.model.state_dict()
        torch.save(final_net, './model/final_' + self.modelname + '.ckpt')

    def generate_filename(self, score: float, include_score: bool):
        filename = self.args.model + '_' + str(self.args.in_ch)
        if include_score:
            filename = str(score)[2:6] + '_' + self.args.model + '_' + str(self.args.in_ch)
        if self.args.use_res:
            filename += '_res'
        if self.args.use_dilation:
            filename += '_dil'
        if self.args.use_sa:
            filename += '_sa'
        if self.args.use_ca:
            filename += '_ca'
        if self.args.use_dsv:
            filename += '_dsv'
        if self.args.use_edge:
            filename += '_edge'
        if self.args.use_sym:
            filename += '_sym'
        if self.args.sym_co:
            filename += '_symco'
        if self.args.mode == 1:
            filename += ('_cro' + '_' + str(self.args.fold_i) + 'of' + str(self.args.nfold))
        if self.args.mode == 2:
            filename += ('_cro' + '_' + str(self.args.fold_i) + 'of' + str(self.args.nfold) + '_MR')
        return filename
