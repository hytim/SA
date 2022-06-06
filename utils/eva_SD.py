from email.policy import strict
import sys
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torchvision import transforms as T

from torch.utils.data import DataLoader
import utils.dataset_kfold as dataset_kfold
import utils.dataset_MR as dataset_MR
from model.model import U_Net, TB, ATT_U
from model.ca_net import CA_Net
from model.bio_net import BiONet
from model.stn_model import stn
from utils.evaluation import *
from utils.hausdorff_distance import get_HausdorffDistance
from utils.mean_surface_distance import get_MSD
import warnings
from collections import OrderedDict
from param_parser import args_parser
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def test():
    args = args_parser()
    add_window = True
    is_norm = False
    model_root = r'./models/e2e_rotFlip_3fold_CT/square_coTra/BiO'
    threshold = 0.05
    print(threshold)
    print(model_root)
    HDD_all_avg, MSD_all_avg, ACC_all_avg, SE_all_avg, SE_H_all_avg, PC_all_avg, SP_all_avg, DSC_all_avg = [], [], [], [], [], [], [], []
    HDD_all_avg_s, MSD_all_avg_s, ACC_all_avg_s, SE_all_avg_s, SE_H_all_avg_s, PC_all_avg_s, SP_all_avg_s, DSC_all_avg_s = [], [], [], [], [], [], [], []
    HDD_l1_avg, MSD_l1_avg, ACC_l1_avg, SE_l1_avg, SE_H_l1_avg, PC_l1_avg, SP_l1_avg, DSC_l1_avg = [], [], [], [], [], [], [], []
    HDD_l2_avg, MSD_l2_avg, ACC_l2_avg, SE_l2_avg, SE_H_l2_avg, PC_l2_avg, SP_l2_avg, DSC_l2_avg = [], [], [], [], [], [], [], []
    HDD_l_s1_avg, MSD_l_s1_avg, ACC_l_s1_avg, SE_l_s1_avg, SE_H_l_s1_avg, PC_l_s1_avg, SP_l_s1_avg, DSC_l_s1_avg = [], [], [], [], [], [], [], []
    HDD_l_s2_avg, MSD_l_s2_avg, ACC_l_s2_avg, SE_l_s2_avg, SE_H_l_s2_avg, PC_l_s2_avg, SP_l_s2_avg, DSC_l_s2_avg = [], [], [], [], [], [], [], []
    model_list = os.listdir(model_root)
    for model_name in model_list:
        print(model_name)
        HDD_l1, MSD_l1, ACC_l1, SE_l1, SE_H_l1, PC_l1, SP_l1, DSC_l1 = [], [], [], [], [], [], [], []
        HDD_l2, MSD_l2, ACC_l2, SE_l2, SE_H_l2, PC_l2, SP_l2, DSC_l2 = [], [], [], [], [], [], [], []
        HDD_l_s1, MSD_l_s1, ACC_l_s1, SE_l_s1, SE_H_l_s1, PC_l_s1, SP_l_s1, DSC_l_s1 = [], [], [], [], [], [], [], []
        HDD_l_s2, MSD_l_s2, ACC_l_s2, SE_l_s2, SE_H_l_s2, PC_l_s2, SP_l_s2, DSC_l_s2 = [], [], [], [], [], [], [], []
        args.in_ch = 32 if '_32' in model_name else 64
        args.use_res = 'res' in model_name
        args.use_dilation = 'dil' in model_name
        args.use_ca = 'ca' in model_name
        args.use_sa = 'sa' in model_name
        args.use_dsv = 'dsv' in model_name
        args.use_edge = 'edge' in model_name
        args.use_sym = 'sym' in model_name
        args.fold_i = int(''.join(c for c in model_name.split('of')[0] if c.isdigit())[-1])
        args.nfold = 3
        print(args.fold_i)
        is_MR = 'MR' in model_name
        if is_MR:
            args.in_ch = 8
            val_data = dataset_MR.create(args=args, is_train=False, nfold=args.nfold, fold_i=args.fold_i)
        else:
            val_data = dataset_kfold.create(args=args, data_root=args.data_root,
                            label_root=args.label_root, edge_root=args.edge_root, is_train=False, nfold=args.nfold, fold_i=args.fold_i)
        val_set = DataLoader(val_data, 1,)
        if 'ATT' in model_name:
            model = ATT_U(feature_ch=32)
        elif 'U' in model_name:
            model = U_Net(feature_ch=args.in_ch, dilation=args.use_dilation, dsv=args.use_dsv,
                               res=args.use_res, use_sa=args.use_sa, use_ca=args.use_ca, use_edge=args.use_edge)
        elif 'CA' in model_name:
            model = CA_Net()
        elif 'BiO' in model_name:
            model = BiONet()
        elif 'Align' in model_name or 'TB' in model_name:
            model = TB(feature_ch=args.in_ch, dilation=args.use_dilation, res=args.use_res,
                            use_sa=args.use_sa, use_ca=args.use_ca, dsv=args.use_dsv, use_edge=args.use_edge)
        else:
            print('illegal model')
        state_dict = torch.load(os.path.join(model_root, model_name))
        multi_gpu = False
        if multi_gpu:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        model = model.cuda().eval()

        with torch.no_grad():
            pa_last = 0
            ch_cur = 'A'
            ch_last = 'A'
            flag = False
            for i, (dictionary) in enumerate(val_set):
                data = Variable(dictionary['data'].cuda())
                label = Variable(dictionary['label'].cuda())
                edge = Variable(dictionary['edge'].cuda())
                filename = dictionary['filename']
                if is_MR:
                    pa_cur = filename[0].split('/')[-3]
                else:
                    pa_cur = int(filename[0].split('/')[-1].split('_')[0])
                    ch_cur = filename[0].split('/')[-1].split('_')[1]
                if pa_cur != pa_last or ch_cur != ch_last or (i == (val_set.__len__() - 1)):
                    if (i == (val_set.__len__() - 1)):
                        if args.use_edge:
                            output, pre_edge = model(data)
                        else:
                            output = model(data)

                        output = output > 0.5
                        label = label > 0.5

                        TP_single = torch.sum((output == 1) & (label == 1))
                        TN_single = torch.sum((output == 0) & (label == 0))
                        FN_single = torch.sum((output == 0) & (label == 1))
                        FP_single = torch.sum((output == 1) & (label == 0))

                        TP += TP_single
                        TN += TN_single
                        FN += FN_single
                        FP += FP_single
                        
                        if is_norm:
                            hematoma = ((label == 1) & (
                                data > ((40/80-0.5)/0.5)))
                        else:
                            hematoma = ((label == 1) & (
                                data > (40/80)))
                        filter_label = label * hematoma
                        filter_prediction = output * hematoma

                        TP_H_single = torch.sum((filter_prediction == 1) & (filter_label == 1))
                        FN_H_single = torch.sum((filter_prediction == 0) & (filter_label == 1))
                        TP_H += TP_H_single
                        FN_H += FN_H_single

                        HDD_single = get_HausdorffDistance(output, label)
                        MSD_single = get_MSD(output, label, symmetric=True)
                        HDD += HDD_single
                        MSD += MSD_single
                        length += 1

                        ACC_s = (TP_single + TN_single) / (TP_single + TN_single + FP_single + FN_single)
                        SE_s = TP_single / (TP_single + FN_single)
                        PC_s = TP_single / (TP_single + FP_single)
                        SP_s = TN_single / (TN_single + FP_single)
                        DSC_s = (2 * TP_single) / (TP_single + FN_single + TP_single + FP_single)
                        if (filter_label.sum() / label.sum()) > threshold:
                            flag = True
                            SE_H_s = TP_H_single / (TP_H_single + FN_H_single)

                            SE_H_l_s2.append(SE_H_s.item())
                            HDD_l_s2.append(HDD_single)
                            MSD_l_s2.append(MSD_single)
                            ACC_l_s2.append(ACC_s.item())
                            SE_l_s2.append(SE_s.item())
                            if not torch.isnan(PC_s):
                                PC_l_s2.append(PC_s.item())
                            SP_l_s2.append(SP_s.item())
                            DSC_l_s2.append(DSC_s.item())
                        else:
                            HDD_l_s1.append(HDD_single)
                            MSD_l_s1.append(MSD_single)
                            ACC_l_s1.append(ACC_s.item())
                            SE_l_s1.append(SE_s.item())
                            if not torch.isnan(PC_s):
                                PC_l_s1.append(PC_s.item())
                            SP_l_s1.append(SP_s.item())
                            DSC_l_s1.append(DSC_s.item())

                    if (pa_last != 0):
                        HDD = HDD / length
                        MSD = MSD /length
                        ACC = (TP + TN) / (TP + TN + FP + FN)
                        SE = TP / (TP + FN)
                        SE_H = TP_H / (TP_H + FN_H)
                        PC = TP / (TP + FP)
                        SP = TN / (TN + FP)
                        DSC = (2 * SE * PC) / (SE + PC)
                        if not flag:
                            SE_H_l1.append(SE_H.item())
                            HDD_l1.append(HDD)
                            MSD_l1.append(MSD)
                            ACC_l1.append(ACC.item())
                            SE_l1.append(SE.item())
                            PC_l1.append(PC.item())
                            SP_l1.append(SP.item())
                            DSC_l1.append(DSC.item())
                        else:
                            SE_H_l2.append(SE_H.item())
                            HDD_l2.append(HDD)
                            MSD_l2.append(MSD)
                            ACC_l2.append(ACC.item())
                            SE_l2.append(SE.item())
                            PC_l2.append(PC.item())
                            SP_l2.append(SP.item())
                            DSC_l2.append(DSC.item())
                            flag = False

                    pa_last = pa_cur
                    ch_last = ch_cur
                    TP = TN = FP = FN = HDD = MSD = length = 0
                    TP_H = FN_H = 0

                if (i == (val_set.__len__() - 1)):
                    continue
                if args.use_edge:
                    output, pre_edge = model(data)
                else:
                    output = model(data)

                output = output > 0.5
                label = label > 0.5

                TP_single = torch.sum((output == 1) & (label == 1))
                TN_single = torch.sum((output == 0) & (label == 0))
                FN_single = torch.sum((output == 0) & (label == 1))
                FP_single = torch.sum((output == 1) & (label == 0))

                TP += TP_single
                TN += TN_single
                FN += FN_single
                FP += FP_single
                
                if is_norm:
                    hematoma = ((label == 1) & (
                        data > ((40/80-0.5)/0.5)))
                else:
                    hematoma = ((label == 1) & (
                        data > (40/80)))
                filter_label = label * hematoma
                filter_prediction = output * hematoma

                TP_H_single = torch.sum((filter_prediction == 1) & (filter_label == 1))
                FN_H_single = torch.sum((filter_prediction == 0) & (filter_label == 1))
                TP_H += TP_H_single
                FN_H += FN_H_single

                HDD_single = get_HausdorffDistance(output, label)
                MSD_single = get_MSD(output, label, symmetric=True)
                HDD += HDD_single
                MSD += MSD_single
                length += 1

                ACC_s = (TP_single + TN_single) / (TP_single + TN_single + FP_single + FN_single)
                SE_s = TP_single / (TP_single + FN_single)
                PC_s = TP_single / (TP_single + FP_single)
                SP_s = TN_single / (TN_single + FP_single)
                DSC_s = (2 * TP_single) / (TP_single + FN_single + TP_single + FP_single)
                if (filter_label.sum() / label.sum()) > threshold:
                    flag = True
                    SE_H_s = TP_H_single / (TP_H_single + FN_H_single)

                    SE_H_l_s2.append(SE_H_s.item())
                    HDD_l_s2.append(HDD_single)
                    MSD_l_s2.append(MSD_single)
                    ACC_l_s2.append(ACC_s.item())
                    SE_l_s2.append(SE_s.item())
                    if not torch.isnan(PC_s):
                        PC_l_s2.append(PC_s.item())
                    SP_l_s2.append(SP_s.item())
                    DSC_l_s2.append(DSC_s.item())
                else:
                    HDD_l_s1.append(HDD_single)
                    MSD_l_s1.append(MSD_single)
                    ACC_l_s1.append(ACC_s.item())
                    SE_l_s1.append(SE_s.item())
                    if not torch.isnan(PC_s):
                        PC_l_s1.append(PC_s.item())
                    SP_l_s1.append(SP_s.item())
                    DSC_l_s1.append(DSC_s.item())

        DSC_l1_avg.append(np.mean(DSC_l1))
        DSC_l2_avg.append(np.mean(DSC_l2))
        DSC_l_s1_avg.append(np.mean(DSC_l_s1))
        DSC_l_s2_avg.append(np.mean(DSC_l_s2))
        DSC_all_avg.append(np.mean(DSC_l1+DSC_l2))
        DSC_all_avg_s.append(np.mean(DSC_l_s1+DSC_l_s2))
        SE_l1_avg.append(np.mean(SE_l1))
        SE_l2_avg.append(np.mean(SE_l2))
        SE_l_s1_avg.append(np.mean(SE_l_s1))
        SE_l_s2_avg.append(np.mean(SE_l_s2))
        SE_all_avg.append(np.mean(SE_l1+SE_l2))
        SE_all_avg_s.append(np.mean(SE_l_s1+SE_l_s2))
        PC_l1_avg.append(np.mean(PC_l1))
        PC_l2_avg.append(np.mean(PC_l2))
        PC_l_s1_avg.append(np.mean(PC_l_s1))
        PC_l_s2_avg.append(np.mean(PC_l_s2))
        PC_all_avg.append(np.mean(PC_l1+PC_l2))
        PC_all_avg_s.append(np.mean(PC_l_s1+PC_l_s2))
        SP_l1_avg.append(np.mean(SP_l1))
        SP_l2_avg.append(np.mean(SP_l2))
        SP_l_s1_avg.append(np.mean(SP_l_s1))
        SP_l_s2_avg.append(np.mean(SP_l_s2))
        SP_all_avg.append(np.mean(SP_l1+SP_l2))
        SP_all_avg_s.append(np.mean(SP_l_s1+SP_l_s2))
        HDD_l1_avg.append(np.mean(HDD_l1))
        HDD_l2_avg.append(np.mean(HDD_l2))
        HDD_l_s1_avg.append(np.mean(HDD_l_s1))
        HDD_l_s2_avg.append(np.mean(HDD_l_s2))
        HDD_all_avg.append(np.mean(HDD_l1+HDD_l2))
        HDD_all_avg_s.append(np.mean(HDD_l_s1+HDD_l_s2))
        MSD_l1_avg.append(np.mean(MSD_l1))
        MSD_l2_avg.append(np.mean(MSD_l2))
        MSD_l_s1_avg.append(np.mean(MSD_l_s1))
        MSD_l_s2_avg.append(np.mean(MSD_l_s2))
        MSD_all_avg.append(np.mean(MSD_l1+MSD_l2))
        MSD_all_avg_s.append(np.mean(MSD_l_s1+MSD_l_s2))
        SE_H_l1_avg.append(np.mean(SE_H_l1))
        SE_H_l2_avg.append(np.mean(SE_H_l2))
        SE_H_l_s1_avg.append(np.mean(SE_H_l_s1))
        SE_H_l_s2_avg.append(np.mean(SE_H_l_s2))
        SE_H_all_avg.append(np.mean(SE_H_l1+SE_H_l2))
        SE_H_all_avg_s.append(np.mean(SE_H_l_s1+SE_H_l_s2))
        
    # assert len(DSC_all_avg) == 5, f'{len(DSC_all_avg)}!=5'
    assert len(val_set) == len(DSC_l_s1+DSC_l_s2), f'{len(val_set)} != {len(DSC_l_s1+DSC_l_s2)}'
    print(DSC_all_avg)
    print(DSC_all_avg_s)

    print('pa_wise\nnoIVH: DC: %.4f\u00B1%.4f, SE: %.4f\u00B1%.4f, SP: %.4f\u00B1%.4f, HD: %.4f\u00B1%.4f, ASD: %.4f\u00B1%.4f, SE_H: %.4f\u00B1%.4f'%(
                np.mean(DSC_l1_avg), np.std(DSC_l1_avg), np.mean(SE_l1_avg), np.std(SE_l1_avg),np.mean(SP_l1_avg), np.std(SP_l1_avg), 
                np.mean(HDD_l1_avg), np.std(HDD_l1_avg), np.mean(MSD_l1_avg), np.std(MSD_l1_avg),np.mean(SE_H_l1_avg), np.std(SE_H_l1_avg), 
            ))
    print('IVH:   DC: %.4f\u00B1%.4f, SE: %.4f\u00B1%.4f, SP: %.4f\u00B1%.4f, HD: %.4f\u00B1%.4f, ASD: %.4f\u00B1%.4f, SE_H: %.4f\u00B1%.4f'%(
                np.mean(DSC_l2_avg), np.std(DSC_l2_avg), np.mean(SE_l2_avg), np.std(SE_l2_avg),np.mean(SP_l2_avg), np.std(SP_l2_avg), 
                np.mean(HDD_l2_avg), np.std(HDD_l2_avg), np.mean(MSD_l2_avg), np.std(MSD_l2_avg),np.mean(SE_H_l2_avg), np.std(SE_H_l2_avg),
            ))
    print('total: DC: %.4f\u00B1%.4f, SE: %.4f\u00B1%.4f, SP: %.4f\u00B1%.4f, HD: %.4f\u00B1%.4f, ASD: %.4f\u00B1%.4f, SE_H: %.4f\u00B1%.4f'%(
                np.mean(DSC_all_avg), np.std(DSC_all_avg), np.mean(SE_all_avg), np.std(SE_all_avg), np.mean(SP_all_avg), np.std(SP_all_avg), 
                np.mean(HDD_all_avg), np.std(HDD_all_avg), np.mean(MSD_all_avg), np.std(MSD_all_avg),np.mean(SE_H_l2_avg), np.std(SE_H_l2_avg),
            ))

    print('slice_wise\nnoIVH: DC: %.4f\u00B1%.4f, SE: %.4f\u00B1%.4f, SP: %.4f\u00B1%.4f, HD: %.4f\u00B1%.4f, ASD: %.4f\u00B1%.4f, SE_H: %.4f\u00B1%.4f'%(
                np.mean(DSC_l_s1_avg), np.std(DSC_l_s1_avg), np.mean(SE_l_s1_avg), np.std(SE_l_s1_avg),np.mean(SP_l_s1_avg), np.std(SP_l_s1_avg), 
                np.mean(HDD_l_s1_avg), np.std(HDD_l_s1_avg), np.mean(MSD_l_s1_avg), np.std(MSD_l_s1_avg), np.mean(SE_H_l_s1_avg), np.std(SE_H_l_s1_avg), 
            ))
    print('IVH:   DC: %.4f\u00B1%.4f, SE: %.4f\u00B1%.4f, SP: %.4f\u00B1%.4f, HD: %.4f\u00B1%.4f, ASD: %.4f\u00B1%.4f, SE_H: %.4f\u00B1%.4f'%(
                np.mean(DSC_l_s2_avg), np.std(DSC_l_s2_avg), np.mean(SE_l_s2_avg), np.std(SE_l_s2_avg),np.mean(SP_l_s2_avg), np.std(SP_l_s2_avg), 
                np.mean(HDD_l_s2_avg), np.std(HDD_l_s2_avg), np.mean(MSD_l_s2_avg), np.std(MSD_l_s2_avg),np.mean(SE_H_l_s2_avg), np.std(SE_H_l_s2_avg), 
            ))

    print('total: DC: %.4f\u00B1%.4f, SE: %.4f\u00B1%.4f, SP: %.4f\u00B1%.4f, HD: %.4f\u00B1%.4f, ASD: %.4f\u00B1%.4f, SE_H: %.4f\u00B1%.4f'%(
                np.mean(DSC_all_avg_s), np.std(DSC_all_avg_s), np.mean(SE_all_avg_s), np.std(SE_all_avg_s),np.mean(SP_all_avg_s), np.std(SP_all_avg_s), 
                np.mean(HDD_all_avg_s), np.std(HDD_all_avg_s), np.mean(MSD_all_avg_s), np.std(MSD_all_avg_s),np.mean(SE_H_l_s2_avg), np.std(SE_H_l_s2_avg), 
            ))

if __name__ == '__main__':
    test()
