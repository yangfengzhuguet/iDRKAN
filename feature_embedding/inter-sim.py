# -*- coding: utf-8 -*-
import csv
import torch
import random
import numpy as np
import pandas as pd

#   将关联矩阵中的每个数据转化为tensor后读出
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return md_data



"disease gip sim"
dd_g_matrix = read_csv('../data/HMDD v2/disease_sim/disGIPSim.csv')
"disease consine sim"
dd_c_matrix = read_csv('../data/HMDD v2/disease_sim/disCosSim.csv')
"disease DAG sim"
dd_dag_matrix = read_csv('../data/HMDD v2/disease_sim/disSSim.csv')



### intergate embedding
"disease sim"
disSim=np.zeros((len(dd_g_matrix), len(dd_g_matrix)))
for i in range(disSim.shape[0]):
    for j in range(disSim.shape[1]):
        # if dd_dag_matrix[i][j]!=0:
        #     disSim[i][j] = dd_dag_matrix[i][j]
        # elif dd_c_matrix[i][j]!=0:
        #     disSim[i][j] = (dd_g_matrix[i][j] + dd_c_matrix[i][j]) / 2
        # else:
        #     disSim[i][j]= dd_g_matrix[i][j]

        if dd_c_matrix[i][j] !=0 and dd_dag_matrix[i][j] !=0:
            disSim[i][j] = (dd_g_matrix[i][j] + dd_c_matrix[i][j]+ dd_dag_matrix[i][j]) / 3
        elif dd_c_matrix[i][j] ==0 and dd_dag_matrix[i][j] !=0:
            disSim[i][j] = (dd_g_matrix[i][j] + dd_dag_matrix[i][j]) / 2
        elif dd_c_matrix[i][j] !=0 and dd_dag_matrix[i][j] ==0:
            disSim[i][j] = (dd_g_matrix[i][j] + dd_c_matrix[i][j]) / 2
        else:
            disSim[i][j] = dd_g_matrix[i][j]

# 保存结果
result = pd.DataFrame(disSim)
result.to_csv('embedding-hmdd2/disSim_all.csv', header=False, index=False)



#########################################   circRNA Sim ##################################################
"miRNA gip sim"
cc_g_matrix = read_csv('../data/HMDD v2/miRNA_sim/miGIPSim.csv')
"miRNA consine sim"
cc_c_matrix = read_csv('../data/HMDD v2/miRNA_sim/miCosSim.csv')
"miRNA fun sim"
cc_sem_matrix = read_csv('../data/HMDD v2/miRNA_sim/miFunSim.csv')   #   circRNASS


"miRNA  sim"
circSim = np.zeros((len(cc_g_matrix), len(cc_g_matrix)))
for ii in range(circSim.shape[0]):
    for jj in range(circSim.shape[1]):
        if cc_c_matrix[ii][jj] != 0 and cc_sem_matrix[ii][jj] !=0:
            circSim[ii][jj] = (cc_g_matrix[ii][jj] + cc_c_matrix[ii][jj] + cc_sem_matrix[ii][jj]) / 3
        elif cc_c_matrix[ii][jj] ==0 and cc_sem_matrix !=0:
            circSim[ii][jj] = (cc_g_matrix[ii][jj] + cc_sem_matrix[ii][jj]) / 2
        elif cc_c_matrix[ii][jj] !=0 and cc_sem_matrix[ii][jj] ==0:
            circSim[ii][jj] = (cc_g_matrix[ii][jj] + cc_c_matrix[ii][jj]) / 2
        else:
            circSim[ii][jj] = cc_g_matrix[ii][jj]
# 保存结果
result = pd.DataFrame(circSim)
result.to_csv('embedding-hmdd2/miSim_all.csv', header=False, index=False)