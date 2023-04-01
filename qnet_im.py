import random
import os
from tqdm import tqdm
import numpy as np 
import glob
import pandas as pd
import torch
import scipy.sparse as sp 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp 
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

import os
import time

from diffuse import IC


    
class GNN_skip_small(nn.Module):
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_hidden_3, dropout,feat_d=50):
        super(GNN_skip_small, self).__init__()
        self.fc1 = nn.Linear(2*n_feat, n_hidden_1)
        self.fc2 = nn.Linear(2*n_hidden_1, n_hidden_2)
        self.fc4 = nn.Linear(n_feat+n_hidden_1+n_hidden_2, 1)#+n_hidden_3
        self.feat_d = feat_d
        
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
        h_g = out.scatter_add_(0, idx, x)
        
        #print(out.size())
        out = self.relu(self.fc4(h_g))
        return out, x[:,self.feat_d:] 
    
 

class DeepQNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, device,lr=0.01 ,dropout=0.2 ): 
        super(DeepQNetwork,self).__init__()
        self.input_dims=n_input
        
        self.fc1_dims=n_hidden

        self.fc1 = nn.Linear(self.input_dims , self.fc1_dims)
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims ,1 )
      
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        #self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = device 
        self.to(self.device)

    def forward(self, x): #, idx = []):
        #forward(adj_torch,  model, seed_set)
        #lst = list()
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        return x   
    
    
def gnn_eval(model,A, tmp, feature ,idx ,device):
    feature[tmp,:] = 1
    output, node_rep = model(A, feature,idx)
    return output , node_rep


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


random.seed(1)
torch.manual_seed(1) 

def main():
    device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    ovrlap_thres = 1
    
    feat_d = 50
    dropout = 0.2
    hidden = 64
    model = GNN_skip_small(feat_d,hidden,int(hidden/2),int(hidden/4),dropout).to(device)
    checkpoint = torch.load('models/model_g.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    
    n_input = 3
    hidden_q = 32
    qval =  DeepQNetwork(n_input, hidden_q, device,  0.01) 
    checkpoint = torch.load('models/model_q.pth.tar')
    qval.load_state_dict(checkpoint['state_dict'])
    qval.eval()
    
    
    fw = open("ql_results.txt","a")
    
    
    repeat = 1000
    for g in tqdm(["CR","GR","HT","EN", "FB","YT"]):
        
        print(g)
        if "l" in g:
            path = "data/sim_graphs/train/"+g+".txt"
        else:
            if g=="CR":
                path = "data/real/crime_ic.inf"
            elif g=="GR":
                path = "data/real/gr_ic.inf"
            elif g=="FB":
                path = "data/real/facebook_ic.inf"
            elif g=="HT":
                path = "data/real/ht_ic.inf"
            elif g=="EN":
                path = "data/real/enron_ic.inf"
            elif g=="YT":
                path = "data/real/youtube_ic.inf"
    
        start = time.time()
        
        # Remove nodes based on degree
        G = pd.read_csv(path,header=None,sep=" ")
        nodes = set(G[0].unique()).union(set(G[1].unique()))
        adj = sp.coo_matrix((G[2], (G[1], G[0]) ), shape=(len(nodes), len(nodes)))
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        G.columns = ["source","target","weight"]
    
        outdegree = G.groupby("source").agg('target').count().reset_index()
        if g!="YT":
            deg_thres = np.histogram(outdegree.target,20)#,30) #np.histogram(outdegree.target)
            deg_thres  = deg_thres[1][1] #-deg_thres[1][0])/2  #deg_thres[1][1]
        else:
            deg_thres = np.histogram(outdegree.target,30)#,30) #np.histogram(outdegree.target)
            deg_thres  = deg_thres[1][1] #-deg_thres[1][0])/2  #deg_thres[1][1]
            
        nodes = list(outdegree.source[outdegree.target>deg_thres].values)
        idx = torch.LongTensor(np.array([0]*adj.shape[0])).to(device)
        
        feature = torch.FloatTensor(np.zeros([adj.shape[0],feat_d])).to(device)
    
        influence = []
        inf_set = []
        candidates = []
        #---- run IE to every candidate
        with torch.no_grad():
            for i,u in enumerate(nodes):
    
                o, h_n =  gnn_eval(model, adj,[u],feature.clone(),idx,device) 
                influence.append(o.item()) # name and influence 
    
                pos_b = h_n[:,:hidden].mean(1)+h_n[:,hidden:].mean(1)
    
                pos_b = pos_b.cpu().detach().numpy()
                inf_set.append(set(np.where(pos_b>0)[0]) )
    

        candidates = nodes
        action_idx = np.argmax(influence)
        seed = candidates[action_idx]
    
        seed_set_inf = inf_set[action_idx]
        
        state = torch.FloatTensor(  [influence[action_idx] ] ).to(device) 
    
    
        #------ remove the seed from the data
        del candidates[action_idx]
    
        influence = np.delete(influence, action_idx, 0)
        del inf_set[action_idx]
        
        
        seed_set = [seed]
        
        candi_inf = torch.FloatTensor(influence).unsqueeze(1).to(device) /adj.shape[0]
    
    
        for i in range(1,100):
            #print(i)
            inf_set = [infs - seed_set_inf for infs in inf_set]#/inf_set_lengths[j]
            inf_ovr =  torch.FloatTensor([len(inf_set[j]) for j in range(len( inf_set)) ]).unsqueeze(1).to(device)
            
            state = state/adj.shape[0]
            with torch.no_grad():
                
                #print(state_act.shape)
                state_act = torch.cat([state.repeat(candi_inf.size(0),1) ,candi_inf,inf_ovr], dim=1)#.to(device) 
    
                actions = qval.forward(state_act )
    
            action_idx = torch.argmax(actions).item()
    
            seed =  candidates[action_idx]
    
            del candidates[action_idx]
            del inf_set[action_idx] 
            candi_inf  =  torch.cat([ candi_inf[:action_idx,:],candi_inf[action_idx+1:,:]],dim =0 )
            
            seed_set.append(seed)
    
            state, h_n = gnn_eval(model,adj,seed_set,feature.clone(),idx,device) 
    
    
            new_seed_set_inf = h_n[:,:hidden].mean(1)+h_n[:,hidden:].mean(1)>ovrlap_thres 
            new_seed_set_inf = set(new_seed_set_inf.nonzero().squeeze().cpu().numpy())
    
            #for seen in new_seed_set_inf - seed_set_inf: #(new_seed_set_inf - seed_set_inf).nonzero()
            #    try:
                    
            #        seen_idx = candidates.index(seen)
            #        del candidates[seen_idx]
            #        del inf_set[seen_idx]
            #        candi_inf  =  torch.cat([ candi_inf[:seen_idx,:],candi_inf[seen_idx+1:,:]],dim =0 )
                    
                    #infset = np.delete(infset, seen_idx, 0)
            #    except:
                    
            #        pass
                
            seed_set_inf = new_seed_set_inf
    
        
        xt = time.time()-start
        print("Done, now evaluating..") 
        x_ic = IC(G,seed_set,repeat)
        x_ic20 = IC(G,seed_set[:20],repeat)
        
        fw.write(g.replace("\n","")+',"'+",".join([str(i) for i in seed_set])+'",'+str(x_ic20)+","+str(xt)+","+str(x_ic)+"\n") 
        fw.flush()
    fw.close()
    

if __name__ == "__main__":
    main()