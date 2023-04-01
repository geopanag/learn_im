import logging
import torch
import numpy as np
import pickle
from pathlib import Path
import pandas as pd
import copy
#import seaborn as sns
#import matplotlib
#import matplotlib.pyplot as plt
import time

from main.deepis import DeepIS, DiffusionPropagate, Identity
from main.models.MLP import MLPTransform
from main.utils import to_nparray, to_torch, sp2adj_lists
from main.training import train_model, get_predictions_new_seeds, PIteration, FeatureCons
from main.earlystopping import stopping_args
from main.utils import load_dataset, load_latest_ckpt
# from im.influspread import IS

#plt.style.use('seaborn')
me_op = lambda x, y: np.mean(np.abs(x - y))
te_op = lambda x, y: np.abs(np.sum(x) - np.sum(y)) 

# key parameters
dataset = 'cora_ml' # 'cora_ml', 'citeseer', 'ms_academic', 'pubmed'
model_name = 'deepis' # 'deepis', 

graph = load_dataset(dataset)

influ_mat_list = copy.copy(graph.influ_mat_list)
graph.influ_mat_list = graph.influ_mat_list[:50]
graph.influ_mat_list.shape, influ_mat_list.shape


# training parameters
ndim = 5
propagate_model = DiffusionPropagate(graph.prob_matrix, niter=2)
fea_constructor = FeatureCons(model_name, ndim=ndim)
fea_constructor.prob_matrix = graph.prob_matrix
device = 'cuda' # 'cpu', 'cuda'
args_dict = {
    'learning_rate': 0.0001,
    'λ': 0,
    'γ': 0,
    'ckpt_dir': Path('./checkpoints'),
    'idx_split_args': {'ntraining': 1500, 'nstopping': 500, 'nval': 10, 'seed': 2413340114},  
    'test': False,
    'device': device,
    'print_interval': 1,
    'batch_size': None,
    
}
if model_name == 'deepis':
    gnn_model = MLPTransform(input_dim=ndim, hiddenunits=[64, 64], num_classes=1)
else:
    pass
model = DeepIS(gnn_model=gnn_model, propagate=propagate_model)


dataset = 'cora_ml'
graph = load_dataset(dataset)
influ_mat_list = copy.copy(graph.influ_mat_list)

print("training")
model, result = train_model(model_name + '_' + dataset, model, fea_constructor, graph, **args_dict)


def gnn_eval(model, A, seed_idx ,device, fea_constructor):
    #feature,idx
    seed_vec = np.zeros( [A.shape[0],1] )
    seed_vec[seed_idx,:] = 1
    
    final_preds = get_predictions_new_seeds(model,fea_constructor,seed_vec,np.arange(len(seed_vec)),A,seed_idx)

    return sum(final_preds)#output.cpu().detach().numpy().item()


def celf_(model,adj,chosen_nodes,device,seed_size,fea_constructor):
    start = time.time()
    #idx = torch.LongTensor(np.array([0]*adj.shape[0])).to(device)

    #feature = torch.FloatTensor(np.zeros([adj.shape[0],feat_d])).to(device)
    #feature.clone(),idx,
    with torch.no_grad():
        marg_gain = [gnn_eval(model,adj, [node], device, fea_constructor ) for node in chosen_nodes]
        
    
    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(   zip(chosen_nodes  ,marg_gain),   key=lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    S, spread = [Q[0][0]], Q[0][1]
    
    Q = Q[1:]
    
    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------
    for _ in range(seed_size-1):    

        #check, node_lookup = False, 0
        check= False
        while not check:
            
            # Count the number of times the spread is computed
            #node_lookup += 1
            
            # Recalculate spread of top node
            current = Q[0][0]
            with torch.no_grad():
                # Evaluate the spread function and store the marginal gain in the list
                new_s = gnn_eval(model,adj,S+[current], device, fea_constructor)
                
                
                Q[0] = (current, new_s - spread)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse = True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        
        if len(S)==100:
            x100 = time.time()-start
            
        #SPREAD.append(spread)
        #LOOKUPS.append(node_lookup)
        #timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]

    
    x200 = time.time()-start    
    return S,x100,x200




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



import os
import scipy.sparse as sp

from diffuse import IC

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

seed_size = 200

#fw = open("deepis_celf_1.csv","a")
#fw.write("graph,nodes,infl20,time20,infl100,time100\n")

fw_ = open("deepis_celf.csv","a")
fw_.write("graph,nodes,time100,time200,infl20,infl50,infl100,infl200\n")


for g in ["EN","FB","CR","GR","HT","YT" ]: 

    print(g)
    if "l" in g:
        path = "data/sim_graphs/train/"+g+".txt"
    else:
        if g=="CR":
            path = "data/real/crime_ic.inf"
        elif g=="GR":
            path = "data/real/gr_ic.inf"
        elif g=="GNU":
            path = "data/real/gnutella31_ic.inf"
        elif g=="FB":
            path = "data/real/facebook_ic.inf"
        elif g=="HT":
            path = "data/real/ht_ic.inf"
        elif g=="EN":
            path = "data/real/enron_ic.inf"
        elif g=="YT":
            path = "data/real/youtube_ic.inf"
        elif g=="EP":
            path = "data/real/epinions_ic.inf"

    print(g)
    
    st = time.time()
    
    G = pd.read_csv(path,header=None,sep=" ")
    nodes = set(G[0].unique()).union(set(G[1].unique()))
    adj = sp.coo_matrix((G[2], (G[1], G[0]) ), shape=(len(nodes), len(nodes)))
    #adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    model.propagate = DiffusionPropagate(adj, niter=2)
    G.columns = ["source","target","weight"]
    outdegree = G.groupby("source").agg('target').count().reset_index()

    if g!="YT":
        deg_thres = np.histogram(outdegree.target,20)#,30) #np.histogram(outdegree.target)
        deg_thres  = deg_thres[1][1] #-deg_thres[1][0])/2  #deg_thres[1][1]
    else:
        deg_thres = np.histogram(outdegree.target,30)#,30) #np.histogram(outdegree.target)
        deg_thres  = deg_thres[1][1] #-deg_thres[1][0])/2  #deg_thres[1][1]

    fea_constructor.prob_matrix = adj #graph.prob_matrix
    
    nodes = outdegree.source[outdegree.target>deg_thres].values

        
    S, x100,x200 = celf_(model,adj, nodes, device, seed_size, fea_constructor)
    print("evaluating")
    S = [int(i) for i in S]
    x_ic200,_ = IC(G,S[:200])
    x_ic100,_  = IC(G,S[:100])
    x_ic20,_  = IC(G,S[:20])
    x_ic50,_  = IC(G,S[:50])
    fw_.write(g.replace("\n","")+',"'+",".join([str(i) for i in S])+'",'+
             str(x100)+","+str(x200)+","+str(x_ic20)+","+str(x_ic50)+","+str(x_ic100)+","+str(x_ic200)+"\n")     
    fw_.flush()
    
fw_close()    
    
    