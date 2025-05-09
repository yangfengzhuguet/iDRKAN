import numpy as np
import pandas as pd
import math
import csv



MDA_adj = pd.read_csv('../data/HMDD v2/association_matrix_.csv', header=None)

MDA_adj=np.array(MDA_adj)
# miRNA和diease的数量
nc=np.array(MDA_adj).shape[0]   #535
nd=np.array(MDA_adj).shape[1]   #302



with open("embedding-hmdd2/adj_edgelist.txt", "w") as csvfile:

    for i in range(nc):
        for j in range(nd):
            # if MDA_adj[i][j]==1:
            if MDA_adj[i][j]==1 or MDA_adj[i][j]==-1:
                # csvfile.writerow([i+nd, j])
                csvfile.writelines([str(i+nd), " ", str(j), '\n'])
