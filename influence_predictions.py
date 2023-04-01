import random
import os
from tqdm import tqdm
import numpy as np 
import glob
import pandas as pd
import torch
import scipy.sparse as sp 
import torch.nn as nn
import os

random.seed(1)
torch.manual_seed(1) 


class GNN_skip_small(nn.Module):
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_hidden_3, dropout):
        super(GNN_skip_small, self).__init__()
        self.fc1 = nn.Linear(2*n_feat, n_hidden_1)
        self.fc2 = nn.Linear(2*n_hidden_1, n_hidden_2)
        self.fc4 = nn.Linear(n_feat+n_hidden_1+n_hidden_2, 1)#+n_hidden_3
        
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)

        
    def forward(self, adj,x_in,idx):
        lst = list()

        # 1st message passing layer
        lst.append(x_in)
        
        x = self.relu(self.fc1( torch.cat( (x_in, torch.mm(adj, x_in)),1 ) ) )
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        # 2nd message passing layer
        x = self.relu(self.fc2( torch.cat( (x, torch.mm(adj, x)),1) ))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        # output layer
        x = torch.cat(lst, dim=1)
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1 , x.size(1)).to(x_in.device)
        x = out.scatter_add_(0, idx, x)
        
        #print(out.size())
        x = self.relu(self.fc4(x))

        return x

def gnn_eval(model,A,tmp,feat_d,device):
    idx = torch.LongTensor(np.array([0]*A.shape[0])).to(device)
    feature = np.zeros([A.shape[0],feat_d])
    feature[tmp,:] = 1
    
    output = model( A,torch.FloatTensor(feature).to(device),idx).squeeze()
    return output.cpu().detach().numpy().item()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    
    feat_d = 50
    dropout = 0.4
    hidden=64
    model = GNN_skip_small(feat_d,hidden,int(hidden/2),int(hidden/4),dropout).to(device)
    checkpoint = torch.load('models/model_g.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    
    x = pd.read_csv("data/eval_estimations.csv")
    
    
    for g in x.graph.unique():
        
        row = x[(x.graph==g)]
        
        if "l" in g:
            path = "data/sim_graphs/"+g+".txt"
        else:
            if g=="CR":
                path = "data/real/crime_ic.inf"
            elif g=="GR":
                path = "data/real/gr_ic.inf"
            elif g=="HT":
                path = "data/real/ht_ic.inf"
         
            
        G = pd.read_csv(path,header=None,sep=" ")
        nodes = set(G[0].unique()).union(set(G[1].unique()))
        # make the sparse transpose adjacency matrix 
        adj = sp.coo_matrix((G[2], (G[1], G[0])), shape=(len(nodes), len(nodes)))
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        
        for i,ro in tqdm(row.iterrows()):
            seeds = [int(no) for no in ro["node"].split(",")]
            if len(seeds)>10:
                continue
            x.loc[i,"preds"] = gnn_eval(model,adj,seeds,feat_d,device)
       
    
    x.to_csv("data/predictions.csv",index=False)    
    
    x["diff"] = abs(x["preds"] -x["infl"])
    
    print(x.loc[:,["graph","infl","diff"]].groupby(["graph"]).mean())


if __name__ == "__main__":
    main()
