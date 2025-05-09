import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--isfuse', type=str, default='yes', help='Whether to test using converged multiple views')
    parser.add_argument('--isuse', type=str, default='no', help='Whether to use the fuse function')
    parser.add_argument('--parent_dir', type=str, default='data/HMDD v_4/predata', help='The parent_dir of os.path')
    parser.add_argument('--parent_dir_', type=str, default='data/HMDD v_4/10-fold-balance', help='The parent_dir_sub of os.path')

    parser.add_argument('--parent_dir1', type=str, default='data/miR2Disease/predata', help='The parent_dir of os.path')
    parser.add_argument('--parent_dir_1', type=str, default='data/miR2Disease/10-folds_balance', help='The parent_dir_sub of os.path')


    parser.add_argument('--pos_sum', type=int, default=10, help='The number of meta-path postive sample pairs')
    parser.add_argument('--lambda_2', type=float, default=0.4, help='The parameter in the total loss')

    # GConv
    parser.add_argument('--fm', type=int, default=64, help='Hidden  layer of miRNA in graph convolution')
    parser.add_argument('--fd', type=int, default=64, help='Hidden  layer of miRNA in graph convolution')


    # GConv-meta
    parser.add_argument('--meta_inchannels', type=int, default=64, help='input layer of meta-path graph convolution') # 64
    parser.add_argument('--meta_outchannels', type=int, default=64, help='output layer of meta-path graph convolution')# 64

    # construct-learning
    parser.add_argument('--cl_hidden', type=int, default=64, help='Hidden layer of feature project in the CL')# 64
    parser.add_argument('--temperature', type=float, default=0.7, help='The temperature in the CL loss function')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='The parameter in the CL loss function')


    # SLA语义层注意力
    parser.add_argument('--sla_hidden', type=int, default=64, help='Input layer of SLAattention') # 64
    parser.add_argument('--sla_dropout', type=float, default=0.5, help='Dropout of SLAattention') # 0.1


    # 模型参数设置
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--we_decay', type=float, default=1e-5, help='The weight decay')
    parser.add_argument('--epoch', type=int, default=110, help='The train epoch')

    return parser.parse_args()
