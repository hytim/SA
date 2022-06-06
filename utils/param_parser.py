import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Ventricle Segmentation')

    parser.add_argument('--seed', dest='seed', type=int,
                        default=128, required=False, help='seed for producing random digits')

    parser.add_argument('--max_epoch', dest='max_epoch', type=int,
                        default=200, required=False, help='max epoch')

    parser.add_argument('--data_root', dest='data_root', type=str,
                        default='../ventricle/data/data', required=False, help='data root')

    parser.add_argument('--label_root', dest='label_root', type=str,
                        default='../ventricle/label/label', required=False, help='label root')

    parser.add_argument('--edge_root', dest='edge_root', type=str,
                        default='../ventricle/label/edge', required=False, help='edge root')

    parser.add_argument('--DID', dest='DID', nargs='+',
                        type=str, default='3', required=False, help='device ID')

    parser.add_argument('--model', dest='model', type=str,
                        default='U', required=False, help='model name')
    
    parser.add_argument('--in_ch', dest='in_ch', type=int,
                        default=32, required=False, help='first convolution channel')

    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--res', dest='use_res', action='store_true', default=False,
                        help='if --res in commond, use_res will be set to True; False if not use')
    
    parser.add_argument('--dil', dest='use_dilation', action='store_true', default=False,
                        help='whether to use dilation convolution')
    
    parser.add_argument('--sa', dest='use_sa', action='store_true', default=False,
                        help='whether to use spatial attention')

    parser.add_argument('--ca', dest='use_ca', action='store_true', default=False,
                        help='whether to use channel attention')
    
    parser.add_argument('--dsv', dest='use_dsv', action='store_true', default=False,
                        help='whether to use deep supervision')

    parser.add_argument('--edge', dest='use_edge', action='store_true', default=False,
                        help='if --edge in commond, use_edge will be set to True; False if not use')
    
    parser.add_argument('--stn', dest='use_stn', action='store_true', default=False,
                        help='if --stn in commond, use_stn will be set to True; False if not use')
    
    parser.add_argument('--sym', dest='use_sym', action='store_true', default=False,
                        help='if --sym in commond, use_sym will be set to True; False if not use')
    
    parser.add_argument('--sym_co', dest='sym_co', action='store_true', default=False,
                        help='if --sym_co in commond, sym_co will be set to True; False if not use')

    parser.add_argument('--no_edge', dest='use_edge', action='store_false',
                        help='if --no_edge in commond, use_edge will be set to False')

    parser.add_argument('--patience', dest='patience',
                        type=int, default=15, required=False, help='patience for early stop')

    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=8, required=False, help='Batch Size')

    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-4, required=False, help='learning rate')

    parser.add_argument('--lr_schedule', dest='lr_schedule', type=int, default=0,
                        required=False, help='0 for cos, 1 for exp, 2 for step, other is illegal')

    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4,
                        required=False, help='weight decay for optimizer')

    parser.add_argument('--train_mode', dest='mode', type=int, default=0,
                        required=False, help='0 for normal, 1 for cross val, other is illegal')

    parser.add_argument('--nfold', dest='nfold',
                        type=int, default=5, required=False, help='total fold for cross val')

    parser.add_argument('--fold_i', dest='fold_i',
                        type=int, default=0, required=False, help='i-th fold, 0 to nfold-1')

    parser.add_argument('--weight', dest='weight', nargs='+',
                        type=float, default=[1., 10., 0., 5., 0.1, 1], required=False, help='weights of loss')

    parser.add_argument('--wce', dest='wce', type=float, default=0.5,
                        required=False, help='weight for cross entropy')

    parser.add_argument('--wedge', dest='wedge', type=float,
                        default=0.8, required=False, help='weight for edge loss')
    
    parser.add_argument('--angle_th', dest='angle_th', type=int,
                        default=5, required=False, help='weight for edge loss')
    
    parser.add_argument('--co_tra', dest='co_tra', action='store_true', default=False,
                        help='if --co_tra in commond, co_tra will be set to True; False if not use')

    args = parser.parse_args()
    return args
