
import random
import os
#import igraph as ig
from tqdm import tqdm
import numpy as np 
import time
import glob
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp 
import torch.nn as nn
import networkx as nx

from diffuse import IC
import math



    
class GNN_skip_small(nn.Module):
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_hidden_3, dropout):
        super(GNN_skip_small, self).__init__()
        self.fc1 = nn.Linear(2*n_feat, n_hidden_1)
        self.fc2 = nn.Linear(2*n_hidden_1, n_hidden_2)
        #self.fc3 = nn.Linear(2*n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_feat+n_hidden_1+n_hidden_2, 1)#+n_hidden_3
        
        
        #self.device=device
        self.n_feat=n_feat
        self.n_hidden_1 = n_hidden_1
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)
        #self.bn3 = nn.BatchNorm1d(n_hidden_3)

        
    def forward(self, adj, x_in, mask, pos, idx, step):
        lst = list()
        #x_in[new_seed,:] = 1
        
        
        # 1st message passing layer
        lst.append(x_in)
        
        x = self.relu(self.fc1( torch.cat( (x_in, torch.mm(adj, x_in)), 1 ) ) )
        x = self.bn1(x)
        x = self.dropout(x)
        
        x[mask,:] = 0 ###---- update
        
        
        h1 = x.mean(1)
        lst.append(x)

        # 2nd message passing layer
        x = self.relu(self.fc2( torch.cat( (x, torch.mm(adj, x)),1) ))
        x = self.bn2(x)
        x = self.dropout(x)
        
        x[mask,:] = 0 ###---- update
        
        
        h2 = x.mean(1)
        lst.append(x)

        # output layer
        x = torch.cat(lst, dim=1)
        
        #------------------
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1 , x.size(1)).to(x_in.device)
        h_g = out.scatter_add_(0, idx, x)
        
        #print(out.size())
        out = self.relu(self.fc4(h_g))
        
        #------------ update pos and adj
        
        infset = torch.sign(h1).unsqueeze(1)#+h2
        
        if step%update==0:
            infset_ = torch.where(infset>0)[0].unsqueeze(0)
        else:
            infset_ = []
            
        infset[infset<=0]=2
        infset = infset-1
        
        ovr = torch.mm(pos,infset)
        ovr[seed_set] = -math.inf 
        #new_seed = torch.argmin(ovr) 
        new_seed = torch.argmax(ovr) 
        
        
        return out.item(), infset_, new_seed
    
 
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

random.seed(1)
torch.manual_seed(1) 
def main():
    
    device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    feat_d = 50
    dropout = 0.4
    hidden = 64
    model = GNN_skip_small(feat_d,hidden,int(hidden/2),int(hidden/4),dropout).to(device)
    checkpoint = torch.load('models/model_best246.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    fw = open("results/pseudo.txt","a")
    fw.write("--------------------------\n")

    update = 10
    seed_set_size = 100

    for g in ["data/real/crime_ic.inf","data/real/gr_ic.inf","data/real/ht_ic.inf","data/real/enron_ic.inf","data/real/facebook_ic.inf","data/real/youtube_ic.inf"]:
        print(g)
        start_game = time.time()
        
        indices = torch.load(g.replace("_ic.inf","_indices.pt")) 
        values = torch.load(g.replace("_ic.inf","_values.pt")) 
        shape = torch.load(g.replace("_ic.inf","_shape.pt")) 
        
        adj = torch.sparse.FloatTensor(indices, values, shape).to(device)


        indices = torch.load(g.replace("_ic.inf","_indices_p.pt")) 
        values = torch.load(g.replace("_ic.inf","_values_p.pt")) 
        
        pos = torch.sparse.FloatTensor(indices, values, shape).to(device)
        
        idx = torch.LongTensor(np.array([0]*adj.shape[0])).to(device)
        seeds = torch.FloatTensor(torch.zeros([adj.shape[0],feat_d])).to(device)  

        seed_set = []
        
        m1 = torch.ones(adj.shape[0]).unsqueeze(1).to(device)
        new_seed = torch.argmax(torch.sparse.mm(pos,m1 ) ).cpu().item()
        seed_set.append(new_seed)
        seeds[new_seed,:] = 1
        mask = torch.LongTensor([]).to(device)
        
        
        for step in range(1,seed_set_size):
            
            with torch.no_grad():
                # assign chosen seed back to 0, as we only give 1 seed as input
                influence, infset , new_seed =  model(adj, seeds, mask, pos, idx, step)
            
                
            new_seed = new_seed.cpu().item()
            seed_set.append(new_seed)
            seeds[new_seed,:]=1
            
            if len(infset)>1:
                # remove influenced nodes and reinitialize seed set
                mask = torch.cat((mask, infset ), 1)
            
        tim = time.time()-start_game      
        print(tim)
        
        G = pd.read_csv(g,header=None,sep=" ")
        G.columns = ["source","target","weight"]
        #------ store everything
        sigma2, std2 = IC(G,seed_set[0:20])
        sigma, std  = IC(G,seed_set)
        
        fw.write(g+","+str(len(set(seed_set)))+","+str(tim)+","+" , "+str(round(sigma))+" , "+str(round(sigma2))+" , "+str(round(std))+" , "+str(round(std2))+"\n") 
        fw.flush()
    fw.close()

if __name__ == "__main__":
    main()          
        
        