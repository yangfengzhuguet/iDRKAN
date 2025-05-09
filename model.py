import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.init as init
import os
from sklearn.decomposition import PCA
# import umap
from sklearn.manifold import TSNE
from KAN_ import *
from sklearn.manifold import LocallyLinearEmbedding
from torch_geometric.nn import SAGEConv
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch_geometric.nn import GATConv
from torch_geometric.nn import TransformerConv

# 定义归一化层
class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
        """
        assert mode in ['PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        col_mean = x.mean(dim=0)
        ######对于图卷积神经网络(GCN)、图注意力网络(GAT)使用PN-SI或者PN-SCS，而且层数超过五层的时候效果比较好
        if self.mode == 'PN-SI':
            x = x - col_mean  # center
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual  # scale

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


# 定义语义层注意力SLA,用于融合不同原路径下的视图信息
class SLAttention(nn.Module):
    def __init__(self, hidden, dropout):
        super(SLAttention, self).__init__()
        self.fc = nn.Linear(hidden, hidden, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.act_tan = nn.Tanh()
        self.a = nn.Parameter(torch.empty(size=(1, hidden)), requires_grad=True)
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        #self.act_sof = torch.nn.functional.softmax()

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x # 如果没有dropout那么使用匿名函数直接返回原先的值

    def forward(self, list_): # 该list_各个视图的结果
        beta = []
        a = self.a
        for view_ in list_:
            view_ = self.dropout(view_)
            feat = self.act_tan(self.fc(view_)).mean(dim=0)
            beta.append(a.matmul(feat.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = torch.nn.functional.softmax(beta, dim=-1)

        fin_metra = 0
        for i in range(len(list_)):
            fin_metra += list_[i] * beta[i]
        return fin_metra

# 定义用于元路径的图卷积
class GConv_meta(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.8, bias=True):
        super(GConv_meta, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=True)
        self.drop = nn.Dropout(drop)
        self.acti_fun = nn.PReLU() # PRelu

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            self.bias.data.fill_(0.001)
        else:
            self.register_parameter('bias', None)
        for model in self.modules():
            self.weight_init(model)
    def weight_init(self, model):
        if isinstance(model, nn.Linear):
            nn.init.xavier_normal_(model.weight, gain=1.414)
            if model.bias is not None:
                model.bias.data.fill_(0.0)

    def forward(self, emb, meta):
        emb = F.dropout(emb, 0.3)
        emb_feat = self.fc(emb)
        out = torch.spmm(meta, emb_feat)
        if self.bias is not None:
            out += self.bias
        out = self.drop(out)
        out = self.acti_fun(out)
        return out

# 定义相似度图卷积
class GCN(nn.Module):
    def __init__(self, in_feat, hidden, out_feat, flag):
        super(GCN, self).__init__()

        if flag == 'miRNA':
            # GCN for miRNA
            self.mi_SNF_1 = GCNConv(in_feat, hidden)
            self.mi_SNF_2 = GCNConv(hidden, out_feat)
            self.mi_fc1 = nn.Linear(in_features=6,
                                    out_features=5 * 6)
            self.mi_fc2 = nn.Linear(in_features=5 * 6,
                                    out_features=6)
            self.globalAvgPool_x = nn.AvgPool2d((64, 728), (1, 1))
            self.sigmoidx = nn.Sigmoid()
            self.cnn_mi = nn.Conv2d(in_channels=6, out_channels=1,
                                    kernel_size=(1, 1), stride=1, bias=True)

            self.pca = PCA(n_components=64)
            self.sage1 = SAGEConv(64, 64)
        else:
            # GCN for disease
            self.di_SNF_1 = GCNConv(in_feat, hidden)
            self.di_SNF_2 = GCNConv(hidden, in_feat)
            # 图卷积后的各个视图堆叠成立方体进一步提取特征用于消融
            self.di_fc1 = nn.Linear(in_features=6,
                                    out_features=5 * 6)
            self.di_fc2 = nn.Linear(in_features=5 * 6,
                                    out_features=6)
            self.globalAvgPool_y = nn.AvgPool2d((64, 884), (1, 1))
            self.sigmoidy = nn.Sigmoid()
            self.cnn_di = nn.Conv2d(in_channels=6, out_channels=1,
                                    kernel_size=(1, 1), stride=1, bias=True)

            self.pca = PCA(n_components=64)
            self.sage1 = SAGEConv(64, 64)

    def forward(self, sim_set, flag_):
        if flag_ == 'miRNA':
            mi_infeats = torch.randn(728, 64)
            # miRNA
            x_m_g1 = torch.relu(self.mi_SNF_1(mi_infeats.to(device), sim_set['miRNA_mut']['mi_gua_edges'].to(device),
                                sim_set['miRNA_mut']['mi_gua'][sim_set['miRNA_mut']['mi_gua_edges'][0], sim_set['miRNA_mut']['mi_gua_edges'][1]]))
            x_m_g2 = torch.relu(self.mi_SNF_2(x_m_g1, sim_set['miRNA_mut']['mi_gua_edges'].to(device), sim_set['miRNA_mut']['mi_gua']
                                [sim_set['miRNA_mut']['mi_gua_edges'][0], sim_set['miRNA_mut']['mi_gua_edges'][1]]))
            # x_m_g_1 = torch.relu(self.sage1(x_m_g2, sim_set['miRNA_mut']['mi_gua_edges'].to(device)))
            # x_m_g_1 = torch.relu(self.sage1(mi_infeats.to(device), sim_set['miRNA_mut']['mi_gua_edges'].to(device)))
            # x_m_g_2 = torch.relu(self.sage1(x_m_g_1, sim_set['miRNA_mut']['mi_gua_edges'].to(device)))


            x_m_c1 = torch.relu(self.mi_SNF_1(mi_infeats.to(device), sim_set['miRNA_mut']['mi_cos_edges'].to(device),
                                sim_set['miRNA_mut']['mi_cos'][sim_set['miRNA_mut']['mi_cos_edges'][0], sim_set['miRNA_mut']['mi_cos_edges'][1]]))
            x_m_c2 = torch.relu(self.mi_SNF_2(x_m_c1, sim_set['miRNA_mut']['mi_cos_edges'].to(device), sim_set['miRNA_mut']['mi_cos']
                                [sim_set['miRNA_mut']['mi_cos_edges'][0], sim_set['miRNA_mut']['mi_cos_edges'][1]]))
            # x_m_c_1 = torch.relu(self.sage1(x_m_c2, sim_set['miRNA_mut']['mi_cos_edges'].to(device)))
            # x_m_c_1 = torch.relu(self.sage1(mi_infeats.to(device), sim_set['miRNA_mut']['mi_cos_edges'].to(device)))
            # x_m_c_2 = torch.relu(self.sage1(x_m_c_1, sim_set['miRNA_mut']['mi_cos_edges'].to(device)))

            x_m_f1 = torch.relu(self.mi_SNF_1(mi_infeats.to(device), sim_set['miRNA_mut']['mi_fun_edges'].to(device),
                                sim_set['miRNA_mut']['mi_fun'][sim_set['miRNA_mut']['mi_fun_edges'][0], sim_set['miRNA_mut']['mi_fun_edges'][1]]))
            x_m_f2 = torch.relu(self.mi_SNF_2(x_m_f1, sim_set['miRNA_mut']['mi_fun_edges'].to(device), sim_set['miRNA_mut']['mi_fun']
                                [sim_set['miRNA_mut']['mi_fun_edges'][0], sim_set['miRNA_mut']['mi_fun_edges'][1]]))
            # x_m_f_1 = torch.relu(self.sage1(x_m_f2, sim_set['miRNA_mut']['mi_fun_edges'].to(device)))
            # x_m_f_1 = torch.relu(self.sage1(mi_infeats.to(device), sim_set['miRNA_mut']['mi_fun_edges'].to(device)))
            # x_m_f_2 = torch.relu(self.sage1(x_m_f_1, sim_set['miRNA_mut']['mi_fun_edges'].to(device)))

            # 使用堆叠立方体特征进行提取
            # XM = torch.cat((x_m_g1, x_m_g2, x_m_c1, x_m_c2, x_m_f1, x_m_f2, x_m_g_1, x_m_g_2, x_m_c_1, x_m_c_2, x_m_f_1, x_m_f_2), 1)
            # XM = torch.cat(( x_m_g1,x_m_g2, x_m_g_1, x_m_c1,x_m_c2, x_m_c_1, x_m_f1, x_m_f2, x_m_f_1), 1)
            XM = torch.cat((x_m_g1,x_m_g2, x_m_c1,x_m_c2, x_m_f1, x_m_f2), 1)
            XM = XM.view(1, 6, 64, -1)
            # XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1)
            x_channel_attention = self.globalAvgPool_x(XM)
            x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), -1)
            x_channel_attention = self.mi_fc1(x_channel_attention)
            x_channel_attention = torch.relu(x_channel_attention)
            x_channel_attention = self.mi_fc2(x_channel_attention)
            x_channel_attention = self.sigmoidx(x_channel_attention)
            x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), x_channel_attention.size(1), 1, 1)
            XM_channel_attention = x_channel_attention * XM
            XM_channel_attention = torch.relu(XM_channel_attention)

            x = self.cnn_mi(XM_channel_attention)
            mi_emb = x.view(64, 728).t()
            # 主成分
            # XM = torch.cat((x_m_g1, x_m_g2, x_m_c1, x_m_c2, x_m_f1, x_m_f2), 1)
            # XM = self.pca.fit_transform(XM.cpu().detach().numpy())
            # XM = torch.Tensor.cpu(torch.from_numpy(XM)).to(device)
            # mi_emb = XM

            # mi_emb = (x_m_g1 + x_m_g2 + x_m_c1 + x_m_c2 + x_m_f1 + x_m_f2) / 6
            # mi_gcn_feat = (x_m_g1 + x_m_g2) / 2
            return mi_emb
        else:
            di_infeats = torch.randn(884, 64)
            # # disease
            y_d_g1 = torch.relu(self.di_SNF_1(di_infeats.to(device), sim_set['disease_mut']['di_gua_edges'].to(device),
                                sim_set['disease_mut']['di_gua'][sim_set['disease_mut']['di_gua_edges'][0], sim_set['disease_mut']['di_gua_edges'][1]]))
            y_d_g2 = torch.relu(self.di_SNF_2(y_d_g1, sim_set['disease_mut']['di_gua_edges'].to(device), sim_set['disease_mut']['di_gua']
                                [sim_set['disease_mut']['di_gua_edges'][0], sim_set['disease_mut']['di_gua_edges'][1]]))
            # y_d_g_1 = torch.relu(self.sage1(y_d_g2, sim_set['disease_mut']['di_gua_edges'].to(device)))
            # y_d_g_1 = torch.relu(self.sage1(di_infeats.to(device), sim_set['disease_mut']['di_gua_edges'].to(device)))
            # y_d_g_2 = torch.relu(self.sage1(y_d_g_1, sim_set['disease_mut']['di_gua_edges'].to(device)))

            y_d_c1 = torch.relu(self.di_SNF_1(di_infeats.to(device), sim_set['disease_mut']['di_cos_edges'].to(device),
                                sim_set['disease_mut']['di_cos'][sim_set['disease_mut']['di_cos_edges'][0], sim_set['disease_mut']['di_cos_edges'][1]]))
            y_d_c2 = torch.relu(self.di_SNF_2(y_d_c1, sim_set['disease_mut']['di_cos_edges'].to(device), sim_set['disease_mut']['di_cos']
                                [sim_set['disease_mut']['di_cos_edges'][0], sim_set['disease_mut']['di_cos_edges'][1]]))
            # y_d_c_1 = torch.relu(self.sage1(y_d_c2, sim_set['disease_mut']['di_cos_edges'].to(device)))
            # y_d_c_1 = torch.relu(self.sage1(di_infeats.to(device), sim_set['disease_mut']['di_cos_edges'].to(device)))
            # y_d_c_2 = torch.relu(self.sage1(y_d_c_1, sim_set['disease_mut']['di_cos_edges'].to(device)))

            y_d_d1 = torch.relu(self.di_SNF_1(di_infeats.to(device), sim_set['disease_mut']['di_sem_edges'].to(device),
                                sim_set['disease_mut']['di_sem'][sim_set['disease_mut']['di_sem_edges'][0], sim_set['disease_mut']['di_sem_edges'][1]]))
            y_d_d2 = torch.relu(self.di_SNF_2(y_d_d1, sim_set['disease_mut']['di_sem_edges'].to(device), sim_set['disease_mut']['di_sem']
                                [sim_set['disease_mut']['di_sem_edges'][0], sim_set['disease_mut']['di_sem_edges'][1]]))
            # y_d_d_1 = torch.relu(self.sage1(y_d_d2, sim_set['disease_mut']['di_sem_edges'].to(device)))
            # y_d_d_1 = torch.relu(self.sage1(di_infeats.to(device), sim_set['disease_mut']['di_sem_edges'].to(device)))
            # y_d_d_2 = torch.relu(self.sage1(y_d_d_1, sim_set['disease_mut']['di_sem_edges'].to(device)))

            # 使用堆叠立方体特征进行提取
            # XM = torch.cat((y_d_g1, y_d_g2, y_d_c1, y_d_c2, y_d_d1, y_d_d2, y_d_g_1, y_d_g_2, y_d_c_1, y_d_c_2, y_d_d_1, y_d_d_2), 1)
            # XM = torch.cat((y_d_g_1, y_d_g_2, y_d_c_1, y_d_c_2, y_d_d_1, y_d_d_2), 1)
            # XM = torch.cat((y_d_g1, y_d_g2, y_d_g_1, y_d_c1, y_d_c2, y_d_c_1, y_d_d1, y_d_d2, y_d_d_1), 1)
            XM = torch.cat((y_d_g1, y_d_g2, y_d_c1, y_d_c2, y_d_d1, y_d_d2), 1)
            XM = XM.view(1, 6, 64, -1)
            # XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fd, -1)
            x_channel_attention = self.globalAvgPool_y(XM)
            x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), -1)
            x_channel_attention = self.di_fc1(x_channel_attention)
            x_channel_attention = torch.relu(x_channel_attention)
            x_channel_attention = self.di_fc2(x_channel_attention)
            x_channel_attention = self.sigmoidy(x_channel_attention)
            x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), x_channel_attention.size(1), 1, 1)
            XM_channel_attention = x_channel_attention * XM
            XM_channel_attention = torch.relu(XM_channel_attention)

            x = self.cnn_di(XM_channel_attention)
            di_emb = x.view(64, 884).t()
            # 主成分
            # XM = torch.cat((y_d_g1, y_d_g2, y_d_c1, y_d_c2, y_d_d1, y_d_d2), 1)
            #
            # XM = self.pca.fit_transform(XM.cpu().detach().numpy())
            # XM = torch.Tensor.cpu(torch.from_numpy(XM)).to(device)
            # di_emb = XM


            # di_emb = (y_d_g1 + y_d_g2 + y_d_c1 + y_d_c2 + y_d_d1 + y_d_d2) / 6
            # di_gcn_feat = (y_d_g1 + y_d_g2) / 2
            return di_emb


# 定义对比学习
class contrast_learning(nn.Module):
    def __init__(self, hidden, temperature, lambda_1):
        super(contrast_learning, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden)
        )
        self.temperature = temperature
        self.lambda_1 = lambda_1
        # 对权重矩阵进行初始化
        for fc in self.project:
            if isinstance(fc, nn.Linear):
                nn.init.xavier_normal_(fc.weight, gain=1.414)

    # 计算两个视图之间的相似性用于后续的损失函数
    def similarity(self, meta_view, sim_view):
        meta_view_norm = torch.norm(meta_view, dim=-1, keepdim=True)
        sim_view_norm = torch.norm(sim_view, dim=-1, keepdim=True)
        view_dot_fenzi = torch.mm(meta_view, sim_view.t())
        view_dot_fenmu = torch.mm(meta_view_norm, sim_view_norm.t())
        sim_matrix = torch.exp(view_dot_fenzi / view_dot_fenmu / self.temperature)
        return sim_matrix

    def forward(self, meta_, sim_, posSamplePairs):
        # 将特征经过一层线性层进行投影
        meta_project = self.project(meta_)
        sim_project = self.project(sim_)
        view_sim = self.similarity(meta_project, sim_project)
        view_sim_T = view_sim.t()

        view_sim = view_sim / (torch.sum(view_sim, dim=1).view(-1, 1) + 1e-8)
        loss_meta = -torch.log(view_sim.mul(posSamplePairs.to(device)).sum(dim=-1)).mean()

        view_sim_T = view_sim_T / (torch.sum(view_sim_T, dim=1).view(-1, 1) + 1e-8)
        loss_sim = -torch.log(view_sim_T.mul(posSamplePairs.to(device)).sum(dim=-1)).mean()

        return self.lambda_1 * loss_meta + (1 - self.lambda_1) * loss_sim

# 定义元路径特征的提取
class meta_emb(nn.Module):
    def __init__(self, args):
        super(meta_emb, self).__init__()
        self.mdm = GConv_meta(args.meta_inchannels, args.meta_outchannels)
        self.mdmdm = GConv_meta(args.meta_inchannels, args.meta_outchannels)
        self.dmd = GConv_meta(args.meta_inchannels, args.meta_outchannels)
        self.dmdmd = GConv_meta(args.meta_inchannels, args.meta_outchannels)
        self.SLA = SLAttention(args.sla_hidden, args.sla_dropout)

    def forward(self, emb, meta):
        mdm = self.mdm(emb['miRNA'], meta['miRNA']['mdm'].to(device))
        mdmdm = self.mdmdm(emb['miRNA'], meta['miRNA']['mdmdm'].to(device))
        dmd = self.dmd(emb['disease'], meta['disease']['dmd'].to(device))
        dmdmd = self.dmdmd(emb['disease'], meta['disease']['dmdmd'].to(device))

        list_view_mi = [mdm, mdmdm]
        list_view_di = [dmd, dmdmd]
        meta_mi_emb = self.SLA(list_view_mi)
        meta_di_emb = self.SLA(list_view_di)
        # meta_mi_emb = (mdm + mdmdm) / 2
        # meta_di_emb = (dmd + dmdmd) / 2
        return meta_mi_emb, meta_di_emb

class KAN(nn.Module):
    def __init__(self):
        super(KAN, self).__init__()
        self.kanlayer1 = KANLinear(64, 32)
        self.kanlayer2 = KANLinear(32, 16)
        self.kanlayer3 = KANLinear(16, 1)
    def forward(self, mi_emb, di_emb, mi_index, di_index):
        # mi_feat = mi_emb[mi_index]
        # di_feat = di_emb[di_index]
        pair_feat1 = mi_emb * di_emb
        pair_feat2 = self.kanlayer1(pair_feat1)
        pair_feat3 = self.kanlayer2(pair_feat2)
        pair_feat4 = self.kanlayer3(pair_feat3)
        # mi_1 = self.kanlayer1(mi_emb)
        # mi_2 = self.kanlayer2(mi_1)
        # mi_3 = self.kanlayer3(mi_2)
        #
        # di_1 = self.kanlayer1(di_emb)
        # di_2 = self.kanlayer2(di_1)
        # di_3 = self.kanlayer3(di_2)
        #
        # association_score = torch.matmul(mi_3, di_3.t())

        return torch.sigmoid(pair_feat4), pair_feat3

# MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.acti_func = torch.sigmoid
        self.linear1 = nn.Linear(64, 32, bias=True)
        self.linear2 = nn.Linear(32, 16, bias=True)
        self.linear3 = nn.Linear(16, 1, bias=True)
    def forward(self, mi_emb, di_emb, m, d):
        pair_feat1 = mi_emb * di_emb
        pair_feat2 = self.linear1(pair_feat1)
        pair_feat3 = self.linear2(pair_feat2)
        pair_feat4 = self.linear3(pair_feat3)
        return torch.sigmoid(pair_feat4), pair_feat3

# 定义模型guet
class SMCLMDA(nn.Module):
    def __init__(self, args):
        super(SMCLMDA, self).__init__()
        self.args = args
        self.meta_emb = meta_emb(args)
        # 构建对比学习模块
        self.CL_mi = contrast_learning(args.cl_hidden, args.temperature, args.lambda_1)
        self.CL_di = contrast_learning(args.cl_hidden, args.temperature, args.lambda_1)
        self.LayerNorm = torch.nn.LayerNorm(64)
        self.linear = nn.Linear(64, 64)
        # define GCN for miRNA and dsease
        self.gcn_miRNA = GCN(64, 64, 64, 'miRNA')
        self.gcn_disease = GCN(64, 64, 64, 'disease')
        self.kan = KAN()
        self.mlp = MLP()
    def forward(self, sim_set, meta_set, emb, pos_miRNA, pos_disease, miRNA_index, disease_index):
        mi_emb = self.gcn_miRNA(sim_set, 'miRNA')
        di_emb = self.gcn_disease(sim_set, 'disease')


        meta_mi_emb, meta_di_emb = self.meta_emb(emb, meta_set) # 获取元路径的特征
        # 进行对比学习
        loss_cl = self.CL_mi(meta_mi_emb, mi_emb, pos_miRNA) + self.CL_di(meta_di_emb, di_emb, pos_disease)

        mi_emb = mi_emb[miRNA_index]
        meta_mi_emb = meta_mi_emb[miRNA_index]
        di_emb = di_emb[disease_index]
        meta_di_emb = meta_di_emb[disease_index]

        mi_emb_1 = 0.5 * mi_emb + 0.5 * meta_mi_emb
        di_emb_1 = 0.5 * di_emb + 0.5 * meta_di_emb


        mi_emb_ = self.LayerNorm(mi_emb_1)
        di_emb_ = self.LayerNorm(di_emb_1)

        train_score, p = self.kan(mi_emb_, di_emb_, miRNA_index, disease_index)
        return train_score.view(-1), loss_cl, p
        # return train_score, loss_cl, p




