import glob
import networkx as nx

import scipy.sparse as sp
import torch
import numpy as np
import random 

import torch.nn as nn
import os
import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing

import time

from dql_target import Agent 



class GNN_skip_small(nn.Module):
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_hidden_3, dropout,feat_d = 50):
        super(GNN_skip_small, self).__init__()
        self.fc1 = nn.Linear(2*n_feat, n_hidden_1)
        self.fc2 = nn.Linear(2*n_hidden_1, n_hidden_2)
        #self.fc3 = nn.Linear(2*n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_feat+n_hidden_1+n_hidden_2, 1)#+n_hidden_3
        self.feat_d = feat_d
        #self.device=device
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)
        #self.bn3 = nn.BatchNorm1d(n_hidden_3)

        
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

        return out,  x[:,self.feat_d:]

    
def gnn_eval(model,A, tmp, feature ,idx ,device):
    feature[tmp,:] = 1
    output, node_rep = model(A, feature,idx)
    return output.cpu().detach().numpy().item() ,  node_rep


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



random.seed(1)
torch.manual_seed(1) 

def main():

    os.chdir("data")
    
    device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #features = True
    v = str(15)
    
    batch_size = 128
    
    
    MaxIter = 500 # maximum number of training episodes
    #input_dims = 64
    n_step_replay = 2
    
    seed_set_size = 100
    
    epsilon = 0.4
    
    max_mem_size = 10000
    target_update = 50
    hidden_q = 32
    state_dim = 1
    action_dim = 2
    dropout = 0.2
    
    feat_d = 50
    hidden = 64
    model = GNN_skip_small(feat_d,hidden,int(hidden/2),int(hidden/4),dropout).to(device)
    checkpoint = torch.load('../models/model_g.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    
    
    ovrlap_thres = 1
    
    
    log = open("dql_log_"+v+".txt","w")

    graph_dic = {}
    graph_nodes = {}
    graph_qs = {}
    # use max_n for padding with zeros in the new state embedding
    max_n = 0
    
    
    # load all data and run the initial representations
    for g in tqdm(glob.glob("dql_graphs/*")[:3]):#00]:
        print(g)
        G = pd.read_csv(g,header=None,sep=" ")
        
        nodes = set(G[0].unique()).union(set(G[1].unique()))
        adj = sp.coo_matrix((G[2], (G[1], G[0]) ), shape=(len(nodes), len(nodes)))
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        G.columns = ["source","target","weight"]

        outdegree = G.groupby("source").agg('target').count().reset_index()
        deg_thres = np.histogram(outdegree.target)
        
        deg_thres  = deg_thres[1][0]+1
        nodes = list(outdegree.source[outdegree.target>deg_thres].values)
        
        if len(nodes)> max_n:
            max_n = len(nodes)
            
        graph_nodes[g] = nodes
        graph_dic[g] = adj
     
        
        influence = []
        graph_pos = {}
        graph_infset = {}
        
        
        #---- run celf in the beginning
        with torch.no_grad():
            idx = torch.LongTensor(np.array([0]*adj.shape[0])).to(device)

            feature = torch.FloatTensor(np.zeros([adj.shape[0],feat_d])).to(device)
            
            for u in nodes:
                
                o, h_n =  gnn_eval(model, adj,[u],feature.clone(),idx,device) 
                
                influence.append(o) # name and influence 
                
                #graph_emb[u]  = h_g.squeeze(0).cpu().detach().numpy() # graph embedding
                
                pos_b = h_n[:,:hidden].mean(1)+h_n[:,hidden:].mean(1) # influence rep
                graph_pos[u] = pos_b.cpu().detach().numpy()
                
                infset = set(np.where(pos_b.cpu().detach().numpy()>0)[0]) # influence set
                
                graph_infset[u] = infset
                
        candidates = nodes
        action_idx = np.argmax(influence)
        seed = candidates[action_idx]
        
        influence = np.array(influence).reshape(-1, 1)
        
        graph_pos = np.array(list(graph_pos.values()))
        
        
        graph_qs[g] = [seed,  candidates, influence, graph_pos ,list(graph_infset.values())]

        
    max_n = round(max_n)+1
    
   
    agent = Agent(gamma =0.99, epsilon = epsilon, lr = 0.01, state_dim = state_dim, action_dim= action_dim, hidden_dims = hidden_q, batch_size = batch_size, 
                max_n = max_n,max_mem_size=max_mem_size,  device= device, version= v, eps_end = 0.01)
    
    
    print("Finished Loading, starting training")
    bigi = 0
    
    max_score = 0
    eps_history = []
    while(bigi<MaxIter):
        scores = []
        influence_l = []
        counter = 0
        
        for g in graph_dic:
            start_game = time.time()
            
            counter+=1
            adj = graph_dic[g].clone()
            nodes = graph_nodes[g].copy()
            
            idx = torch.LongTensor(np.array([0]*adj.shape[0])).to(device)
    
            feature = torch.FloatTensor(np.zeros([adj.shape[0],feat_d])).to(device)
        
            # run the game and play until the seed set size is reached
            actions_l = []
            
            states_l = []
            new_states_l = []
            
            # choose the first seed with the best 
            steps = 0
            Q = graph_qs[g]
            #[seed,  candidates, influence, graph_pos ,list(graph_infset.values())]

            seed = Q[0].copy()
            candidates = Q[1].copy()
            influence = Q[2].copy()
            
            inf_pos = Q[3].copy()
            
            inf_set = Q[4].copy()
            
            action_idx = candidates.index(seed)
            #--------- current seed set embeddings
            previous_influence = influence[action_idx]/adj.shape[0]
            
            seed_set_pos = inf_pos[action_idx] 
            seed_set_inf = inf_set[action_idx]
            

            #------ remove the seed from the data
            del candidates[action_idx]
            
            influence = np.delete(influence, action_idx, 0)
            inf_pos = np.delete(inf_pos, action_idx, 0)
            
            del inf_set[action_idx] 
            #del inf_set_lengths[action_idx] 
            
            
            seed_set = [seed]
            
            rew_l = []
            
            candidate_original_infl = torch.FloatTensor(influence ) 
            
           
            inf_set = [infs - seed_set_inf for infs in inf_set]#/inf_set_lengths[j]
            inf_ovr =  torch.FloatTensor([len(inf_set[j]) for j in range(len( inf_set)) ]).unsqueeze(1)
            
            state = torch.FloatTensor( previous_influence  )#.unsqueeze(1) 

            for steps in range(1,seed_set_size):
                
                x = torch.cat([state.repeat(candidate_original_infl.size(0),1) ,
                                                               candidate_original_infl,inf_ovr], dim=1)
                if random.uniform(0,1) > agent.epsilon:
                    
                    #----- take the representations of nodes and the graph
                    
                    #----- use the concatenation as input to forward
                    actions = agent.Q_eval.forward( torch.cat([state.repeat(candidate_original_infl.size(0),1) ,
                                                               candidate_original_infl,inf_ovr], dim=1).to(device) )
                      
                    action_idx = torch.argmax(actions).item()
                    
                    action =  candidates[action_idx]
                    
                else:
                    # take the representations of the state 
                    action_idx = np.random.choice(len(candidates))
                    action =  candidates[action_idx]
                # Store previous state, next state, action embedding and immediate reward
                states_l.append(state)
                
                # action is input of the Q network for the chosen seed
                
                actions_l.append( torch.cat([candidate_original_infl[action_idx],inf_ovr[action_idx]] ,dim=0))
                                            #candidate_deg[action_idx],infl_corr[action_idx]
                
                # remove seed 
                seed_set.append(action)
                
                del candidates[action_idx]
                #inf_emb = np.delete(inf_emb, action_idx, 0)
                influence = np.delete(influence, action_idx, 0)
            
                del inf_set[action_idx] 
                #del inf_set_lengths[action_idx] 
                
                
                #-----------------------------------------------------------------------
                # influence, graph embedding and positional influence of the new seed set
                with torch.no_grad():   
                    current_influence, h_n = gnn_eval(model,adj,seed_set,feature.clone(),idx,device) 
                
                #seed_set_pos= seed_set_pos.cpu().detach().numpy()
                
                seed_set_pos = h_n[:,:hidden].mean(1)+h_n[:,hidden:].mean(1)
                new_seed_set_inf = seed_set_pos>ovrlap_thres 

                new_seed_set_inf =set(new_seed_set_inf.nonzero().squeeze().cpu().numpy())
               
                inf_set = [infs - new_seed_set_inf for infs in inf_set]
                inf_ovr =  torch.FloatTensor([len(inf_set[j]) for j in range(len( inf_set)) ]).unsqueeze(1)
                    
                seed_set_inf = new_seed_set_inf    
                
                # the marginal gain is the new influence minus the previous one
                reward = current_influence - previous_influence 
                
                previous_influence = current_influence
                
                # store reward and action
                rew_l.append(reward)
              
                #------ new state using the matrices after the removal of seed
                state = torch.FloatTensor(  [ previous_influence /adj.shape[0] ])
                
                candidate_original_infl = torch.FloatTensor( influence )
                
                new_st =  torch.cat([state.repeat(candidate_original_infl.size(0),1),
                                     candidate_original_infl,
                                     inf_ovr  ], dim=1) #.numpy()
                
                
                #----- Pad with zeros such that all new states have the same dimensions    
                new_st = F.pad(input=new_st, pad=(0, 0, 0, max_n - new_st.shape[0]  ), mode='constant', value=0)
                
                new_states_l.append(new_st)
                #-------------------------------------------------------------------------------------
                # a state is defined from a tuple of the adjecancy and x
                if steps >= n_step_replay:
                    # N-step reward i.e. store  adj, action, rew t+n, ,state_ t+n
                    agent.store_transition( states_l[-n_step_replay], new_states_l[-1], actions_l[-n_step_replay],sum(rew_l[-n_step_replay:])) 
                    agent.learn()   
                   
                if(steps%target_update==0):
                    #target_net.load_state_dict(policy_net.state_dict())
                    agent.Q_target.load_state_dict(agent.Q_eval.state_dict())
                    
                    
            #print("game time :"+ str(time.time()-start_game)       )
            influence_l.append(current_influence)
            scores.append( sum(rew_l) )  #
            eps_history.append(agent.epsilon)
            
            tim = time.time()-start_game
            log.write(str(counter)+",  sc: "+"{:1.3f}".format(scores[-1][0])+"   size: "+"{:1.3f}".format(adj.shape[0])+ ",   tim: "+"{:1.3f}".format(tim)+" , "+str(bigi)+"\n")
            log.flush()
    
        agent.epsilon = 0.99* agent.epsilon
        score = np.mean(scores)
        
        print("turn %d " %bigi )
        
        
        if(score>max_score):
            max_score = score
            agent.save()
            
        bigi+=1  
    
    agent.save()
   


if __name__ == "__main__":
    main()