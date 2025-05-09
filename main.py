import torch
import pandas as pd
import os
import pickle
from torch.optim import Adam, RMSprop
from args import get_args
from utils import set_seed, construct_meta_pos, get_syn_sim, get_emb
from model import SMCLMDA
from train import train_SMCLMDA

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pkl(parent_dir, filename):
    file_path = os.path.join(parent_dir, filename)
    # 从文件加载字典
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def main():
    args = get_args()
    set_seed(args.seed)
    if args.isuse == 'yes':
        mi_final, di_final = get_syn_sim(78, 37)  # 采用非线性的方法进行融合各个数据源的相似度矩阵,保存在自己的目录下
    emb = get_emb() # 获取Node2Vec获取的miRNA、疾病的特征
    sim_set = load_pkl(args.parent_dir, 'sim_set.pkl') # 用于获取融合相似度矩阵或者多视图相似度矩阵
    meta_set = load_pkl(args.parent_dir, 'meta_set.pkl') # 用于元路径的相似度矩阵
    mdm = meta_set['meta']['mdm']
    dmd = meta_set['meta']['dmd']
    pos_miRNA_mask, pos_disease_mask = construct_meta_pos(mdm, dmd, args.pos_sum) # 获取正负样本对的掩码
    print('--------------------------------数据整理完毕，准备构建模型-------------------')
    model = SMCLMDA(args)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.we_decay)
    Acc = []
    Mcc = []
    F1 = []
    AUROC = []
    AUPR = []
    pair_pos_neg_fengceng = load_pkl(args.parent_dir_, f'pos_neg_pair_10_1.pkl')  # 加载正负样本用于十折训练和测试
    accuracy, mcc, f1, auc_, aupr_ = train_SMCLMDA(args, model, sim_set, meta_set, emb, pos_miRNA_mask, pos_disease_mask, optimizer, pair_pos_neg_fengceng, device)
    Acc.append(accuracy)
    Mcc.append(mcc)
    F1.append(f1)
    AUROC.append(auc_)
    AUPR.append(aupr_)
    print('---------------------------------打印指标----------------------------------')
    for acc in Acc:
        print(acc)
    print(f'avg ACC:{sum(Acc) / len(Acc)}')
    for mcc in Mcc:
        print(mcc)
    print(f'avg Mcc:{sum(Mcc) / len(Mcc)}')
    for f1 in F1:
        print(f1)
    print(f'avg F1:{sum(F1) / len(F1)}')
    for auc in AUROC:
        print(auc)
    print(f'avg AUROC:{sum(AUROC) / len(AUROC)}')
    for aupr in AUPR:
        print(aupr)
    print(f'avg AUPR:{sum(AUPR) / len(AUPR)}')
    print('----------------------------------- 结束 ----------------------------------')
if __name__ == "__main__":
    main()