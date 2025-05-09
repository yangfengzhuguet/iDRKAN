import numpy as np
import pandas as pd
import torch.nn
from sklearn.manifold import TSNE
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, matthews_corrcoef, precision_recall_curve
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc
import tqdm
import time
from matplotlib.colors import ListedColormap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,precision_recall_curve,auc,precision_recall_fscore_support,matthews_corrcoef,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score



def show_auc(pre_score, label, flag):
    y_true = label.flatten().detach().cpu().numpy()
    y_score = pre_score.flatten().detach().cpu().numpy()
    # y_true = label
    # y_score = pre_score
    fpr,tpr,rocth = roc_curve(y_true,y_score)
    auroc = auc(fpr,tpr)
    precision,recall,prth = precision_recall_curve(y_true,y_score)
    aupr = auc(recall,precision)
    # if flag == 'test':
    #     # 绘制roc曲线并保存
    #     plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.4f})')
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc='lower right')
    #
    #     # 保存曲线到指定目录
    #     # output_dir = 'result/HMDD v3-2-wuguangdui/10-fold_'
    #     output_dir = 'result/HMDD V4/fenceng'
    #     os.makedirs(output_dir, exist_ok=True)
    #     # plt.savefig(os.path.join(output_dir, '10-fold_9_test_auc.png'))
    #     plt.savefig(os.path.join(output_dir, 'fenceng.png'))
    return auroc, aupr, fpr, tpr, precision, recall

def train_SMCLMDA(arges, model, sim_set, meta_set, emb, pos_miRNA, pos_disease,optimizer, pair_pos_neg_fengceng, device):
    model = model.to(device)
    train_miRNA_index_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,0], dtype=torch.long).to(device) # HMDD v3.2中分层抽样训练集中miRNA的索引
    train_disease_index_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,1],dtype=torch.long).to(device) # HMDD v3.2中分层抽样训练集中disease的索引
    train_label_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,2]).to(device).float() # HMDD v3.2中分层抽样训练集的标签

    test_miRNA_index_fc = torch.tensor(pair_pos_neg_fengceng['test'][:, 0],dtype=torch.long).to(device)  # HMDD v3.2中分层抽样训练集中miRNA的索引
    test_disease_index_fc = torch.tensor(pair_pos_neg_fengceng['test'][:,1],dtype=torch.long).to(device)  # HMDD v3.2中分层抽样训练集中disease的索引
    test_label_fc = torch.tensor(pair_pos_neg_fengceng['test'][:,2]).to(device).float()  # HMDD v3.2中分层抽样训练集的标签

    loss_min = float('inf')
    best_auc = 0
    best_aupr = 0
    tsne = TSNE(n_components=2, random_state=42)
    m = 1
    n = 1
    # HMDD_v4_label = pd.read_csv('data/HMDD v_4/hmdd4_MDA_.csv',sep=',', header=None)
    # HMDD_v4_label = HMDD_v4_label.values
    # HMDD_v4_label = torch.from_numpy(HMDD_v4_label).float()
    # HMDD_v4_label = HMDD_v4_label.to(device)
    #
    # miR2Disease_label = pd.read_csv('data/miR2Disease/miR2Disease_MDA01.csv',sep=',', header=None)
    # miR2Disease_label = miR2Disease_label.values
    # miR2Disease_label = torch.from_numpy(miR2Disease_label).float()
    # miR2Disease_label = miR2Disease_label.to(device)

    print('######################### 开始进入模型的训练 #############################')
    for epoch_ in tqdm.tqdm(range(arges.epoch), desc='Training Epochs'):
        time_start = time.time()
        model.train()
        train_score, loss_cl_train, p = model(sim_set, meta_set, emb, pos_miRNA, pos_disease, train_miRNA_index_fc, train_disease_index_fc)
        loss1 = torch.nn.BCELoss()
        # loss1 = torch.nn.BCEWithLogitsLoss()
        loss_train = loss1(train_score, train_label_fc)
        loss = arges.lambda_2 * loss_cl_train + (1 - arges.lambda_2) * loss_train
        auc_, aupr, f, t, p1, r = show_auc(train_score, train_label_fc, 'train')
        if loss_train < loss_min:
            loss_min = loss_train
        if auc_ > best_auc:
            best_auc = auc_
        if aupr > best_aupr:
            best_aupr = aupr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_end = time.time()
        time_epoch = time_end - time_start
        # if epoch_ in [0, 29, 109]:
        #     plt.rcParams.update({'font.size': 20})
        #     fig, ax = plt.subplots(figsize=(3, 2))
        #     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        #     y = train_label_fc.flatten().detach().cpu().numpy()
        #     features = p
        #     features = features.detach().cpu().numpy()
        #     low_feature = tsne.fit_transform(features)
        #     plt.figure(figsize=(12, 10), dpi=600)
        #     cmap_custom = ListedColormap(['blue', 'red'])
        #     scatter = plt.scatter(
        #         low_feature[:, 0], low_feature[:, 1],
        #         c=y, cmap=cmap_custom, alpha=0.8,
        #         edgecolors='w', linewidths=0.5
        #     )
        #
        #     # 获取图例句柄
        #     handles, _ = scatter.legend_elements()
        #
        #     # 创建自定义图例，只显示色块
        #     # plt.legend(handles, [''] * len(handles), title="Classes",
        #     #            loc='upper right', bbox_to_anchor=(1.15, 1))
        #
        #     plt.title(f'epoch:{epoch_ + 1}', fontsize=20, y=-0.12)
        #     # plt.xlabel('t-SNE Dimension 1', fontsize=20)
        #     # plt.ylabel('t-SNE Dimension 2', fontsize=20)
        #
        #     plt.savefig(f"hmdd4_{epoch_ + 1}.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
        #     plt.close()

        print('-------Time when the epoch runs：{} seconds ----------'.format(time_epoch))
        print('-------The train: epoch{}, Loss:{}, AUC:{}, AUPR{}-------------'.format(epoch_, loss, auc_, aupr))
    print('The loss_min:{}, best auc{}, best aupr{}'.format(loss_min, best_auc, best_aupr))
    print('######################### 模型训练结束,开始进入测试 #############################')
    model.eval()
    with torch.no_grad():
        test_score, loss_cl_test, s = model(sim_set, meta_set, emb, pos_miRNA, pos_disease, test_miRNA_index_fc, test_disease_index_fc)
        loss2 = torch.nn.BCELoss()
        loss_test = loss2(test_score, test_label_fc)
        loss_ = arges.lambda_2 * loss_cl_test + (1 - arges.lambda_2) * loss_test
        auc_, aupr_, fp, tp, pre, rec = show_auc(test_score, test_label_fc, 'test')
        # np.save(f'data/miR2Disease/MLP/fpr_1.npy', fp)
        # np.save(f'data/miR2Disease/MLP/tpr_1.npy', tp)
        # np.save(f'data/miR2Disease/MLP/p_1.npy', pre)
        # np.save(f'data/miR2Disease/MLP/r_1.npy', rec)
        print('-------The test: Loss:{}, AUC:{}, AUPR{}-----------'.format(loss_, auc_, aupr_))

        # 设置图像分辨率和字体
        plt.rcParams['figure.dpi'] = 600
        font1 = {"family": "Arial", "weight": "book", "size": 9}

        # 示例数据
        y_true = np.array(test_label_fc.detach().cpu())  # 因果性标签
        y_true = np.where(y_true == 1, True, False)  # 将1设为True，2设为False
        y_scores = np.array(test_score.detach().cpu())   # 预测分数
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc_ = auc(fpr, tpr)
        roc_auc_ = round(roc_auc_, 3)
        # np.save('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_fp10.npy', fpr)
        # np.save('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_tp10.npy', tpr)


        # 计算Precision-Recall曲线
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        aupr = auc(recall, precision)
        aupr = round(aupr, 3)
        # 计算不同阈值下的性能指标，并找到最佳阈值
        best_threshold = 0.0
        best_f1 = 0.0
        best_metrics = {}

        # 保存每个阈值下的灵敏度和特异性
        sensitivities = []
        specificities = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            accuracy = (y_pred == y_true).mean()
            mcc = matthews_corrcoef(y_true, y_pred)
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            specificity = tn / (tn + fp)

            sensitivities.append(recall)
            specificities.append(specificity)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                    "specificity": specificity
                }

        # 绘制ROC曲线
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"AUROC={roc_auc_}")
        plt.xlabel('False Positive Rate', font1)
        plt.ylabel('True Positive Rate', font1)
        plt.title('ROC Curve', font1)
        plt.legend(prop=font1)

        # 绘制Precision-Recall曲线
        plt.figure(2)
        plt.plot(recall, precision, label=f"AUPR={aupr}", color='purple')
        plt.xlabel('Recall', font1)
        plt.ylabel('Precision', font1)
        plt.title('Precision-Recall Curve', font1)
        plt.legend(prop=font1)

        # 显示最佳阈值下的性能指标
        best_metrics_str = (f"Best Threshold: {best_threshold:.4f}\n"
                            f"Accuracy: {best_metrics['accuracy']:.4f}\n"
                            f"Precision: {best_metrics['precision']:.4f}\n"
                            f"Recall: {best_metrics['recall']:.4f}\n"
                            f"Specificity: {best_metrics['specificity']:.4f}\n"
                            f"MCC: {best_metrics['mcc']:.4f}\n"
                            f"F1 Score: {best_metrics['f1']:.4f}")
        plt.text(0.6, 0.2, best_metrics_str, bbox=dict(facecolor='white', alpha=0.5), fontsize=9)

        # 显示并保存图像
        # plt.savefig("./Result_causal_for_ROC_10_fold_mean.tiff", dpi=600)
        # plt.show()
        # plt.close()
        # 打印最佳阈值和性能指标
        print(f"Best Threshold: {best_threshold}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"MCC: {best_metrics['mcc']:.4f}")
        print(f"F1 Score: {best_metrics['f1']:.4f}")
        print(f"AUROC: {roc_auc_:.4f}")
        print(f"AUPR: {aupr:.4f}")
        model.train()
    return best_metrics['accuracy'], best_metrics['mcc'], best_metrics['f1'], auc_, aupr_

