import os
import igraph as ig
from tqdm import tqdm
import numpy as np 
import time
import glob
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp 
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random




class GNN_skip_small(nn.Module):
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_hidden_3, dropout):
        super(GNN_skip_small, self).__init__()
        self.fc1 = nn.Linear(2*n_feat, n_hidden_1)
        self.fc2 = nn.Linear(2*n_hidden_1, n_hidden_2)
        #self.fc3 = nn.Linear(2*n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_feat+n_hidden_1+n_hidden_2, 1)#+n_hidden_3
        
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
        x_n = torch.cat(lst, dim=1)
        
        idx = idx.unsqueeze(1).repeat(1, x_n.size(1))
        out = torch.zeros(torch.max(idx)+1 , x_n.size(1)).to(x_in.device)
        x = out.scatter_add_(0, idx, x_n)
        
        #print(out.size())
        x = self.relu(self.fc4(x))

        return x
    



def gnn_eval(model,A,tmp,feat_d,device):
    idx = torch.LongTensor(np.array([0]*A.shape[0])).to(device)
    feature = np.zeros([A.shape[0],feat_d])
    feature[tmp,:]=1
    output = model(A, torch.FloatTensor(feature).to(device),idx)
    o = output.squeeze().cpu().detach().numpy()
    
    return o

    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




random.seed(1)
torch.manual_seed(1) 
   
def main():  

    os.chdir("data")
    v = "1"

    #----- Params
    learn = 0.01
    early_stop = 20    

    dropout = 0.4

    hidden = 64
    batch_size  = 64
    
    epochs = 100
    feat_d = 50

    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


    labels = pd.read_csv("influence_train_set.csv")
    
    gs = labels.graph.unique()
    
    #----- read the graphs
    graph_dic = {}
    gs = labels.graph.unique()
    for g in gs:
        G = pd.read_csv("train/"+g+".txt",header=None,sep=" ")
        nodes = set(G[0].unique()).union(set(G[1].unique()))
        graph_dic[g] = sp.coo_matrix((G[2], (G[1], G[0])), shape=(len(nodes), len(nodes))).toarray()
        #G = nx.read_weighted_edgelist("sim_graphs/"+g+".txt", create_using=nx.DiGraph,nodetype= int)#
        #graph_dic[g] = nx.adjacency_matrix(G).toarray().T 
    random.shuffle(gs)
    sam = round(len(gs)/5)      

    # test
    msk = [i for i in range(0,sam)]
    test_graphs = [gs[i] for i in msk]

    train_graphs = [gs[i] for i in range(len(gs)) if i not in msk ] 
    #print(train_graphs)
    val_graphs = [train_graphs[i] for i in msk] 
    train_graphs = [train_graphs[i] for i in range(len(train_graphs)) if i not in msk] 

    traind = labels[labels.graph.isin(train_graphs)]
    testd = labels[labels.graph.isin(test_graphs)]
    vald = labels[labels.graph.isin(val_graphs)]  

    model = GNN_skip_small(feat_d,hidden,int(hidden/2),int(hidden/4),dropout).to(device)   
    
    best_val_acc= 1e8
    val_among_epochs = []
    train_among_epochs = []

    error_log = open("errors/train_error"+str(v)+".txt","w")

    optimizer = optim.Adam(model.parameters(), lr=learn)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    bigstart = time.time()
    print("Training....")
    #------------------- Train
    for epoch in range(epochs):    
        start = time.time()

        model.train()
        train_loss = []
        val_loss = []

        #------------- train for one epoch
        for i in range(0,len(traind),batch_size):

            adj_batch = list()
            feature_batch = list()
            y_batch = list()
            idx_batch = list()

            #------------ create batch
            for u in range(i,min(len(traind),i+batch_size) ):
                row  = traind.iloc[u]

                # take graph
                adj_mat = graph_dic[row["graph"]]
                features = torch.zeros(adj_mat.shape[0],feat_d)

                for st in row["node"].split(","):
                    features[int(st),:] = 1 

                adj_batch.append(adj_mat)
                y_batch.append(row["infl"])
                idx_batch.extend([u-i]*adj_mat.shape[0])
                feature_batch.append(features)

            adj_batch = sp.block_diag(adj_batch)#.toarray()).to(device)        
            adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)

            feature_batch = torch.cat(feature_batch, dim=0).to(device) 
            y_batch = torch.FloatTensor(y_batch).to(device)

            #for batch in range(min(batch_size,N_train - i)):
            idx_batch_ = torch.LongTensor(idx_batch).to(device)

            optimizer.zero_grad()
            output = model(adj_batch, feature_batch,idx_batch_).squeeze()

            loss_train = F.mse_loss(output, y_batch)

            loss_train.backward(retain_graph=True)
            optimizer.step()

            train_loss.append(loss_train.data.item()/output.size()[0])


        #---------- validation
        model.eval()

        for i in range(0,len(vald),batch_size):

            adj_batch = list()
            feature_batch = list()
            y_batch = list()
            idx_batch = list()

            # create batch
            for u in range(i,min(len(vald),i+batch_size) ):
                #row  = vald.iloc[i+u]
                row  = vald.iloc[u]
                # take graph
                adj_mat =graph_dic[row["graph"]]# nx.adjacency_matrix(G_train[i + u])

                features = torch.zeros(adj_mat.shape[0],feat_d)

                for st in row["node"].split(","):
                    features[int(st),:] = 1 

                adj_batch.append(adj_mat)
                y_batch.append(row["infl"])
                idx_batch.extend([u-i]*adj_mat.shape[0])
                feature_batch.append(features)

            adj_batch = torch.FloatTensor(sp.block_diag(adj_batch).toarray()).to(device)        

            feature_batch = torch.cat(feature_batch, dim=0).to(device) 
            y_batch = torch.tensor(y_batch).to(device)

            #for batch in range(min(batch_size,N_train - i)):
            idx_batch_ = torch.LongTensor(idx_batch).to(device)
            output = model(adj_batch, feature_batch,idx_batch_).squeeze()

            vloss = F.mse_loss(output, y_batch)

            vloss = int(np.mean(vloss.detach().cpu().numpy())) 
            val_loss.append(vloss)

        # Print results
        if epoch%5==0:
            print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(np.mean(train_loss)),"val_loss=", "{:.5f}".format(np.mean(val_loss)), "time=", "{:.5f}".format(time.time() - start) )

        error_log.write(str(epoch)+","+str(np.mean(train_loss))+","+str(np.mean(val_loss))+"\n")

        train_among_epochs.append(np.mean(train_loss))
        val_among_epochs.append(np.mean(val_loss))

            #--------- Remember best accuracy and save checkpoint
        if np.mean(val_loss) < best_val_acc:
            best_val_acc =  np.mean(val_loss)
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, '../models/model_best'+str(v)+'.pth.tar')
        if( epoch>early_stop):
            if(len(set([round(val_e) for val_e in val_among_epochs[-10:]])) == 1):#
                print("break")
                break

        scheduler.step(np.mean(val_loss))
    
    print("Training finished")    
    print(time.time()-bigstart)
    print("Now testing")    
    #---------------- Testing
    checkpoint = torch.load('../models/model_best'+str(v)+'.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()


    test_error = []


    fw = open("errors/test_error"+str(v)+".txt","w")
    fw.write("graph,nodes,size,infl,real\n")

    for i in range(0,len(testd)):
        row  = testd.iloc[i]
        adj_mat =graph_dic[row["graph"]]
        features = torch.zeros(adj_mat.shape[0],feat_d)
        for st in row["node"].split(","):
            features[int(st),:] = 1 

        y = row["infl"]
        idx_ =  torch.LongTensor(np.array([0]*adj_mat.shape[0])).to(device)
        adj_mat = torch.FloatTensor(adj_mat).to(device)
        features = torch.FloatTensor(features).to(device)

        output = model(adj_mat, features,idx_).squeeze()

        o = output.cpu().detach().numpy()
        error = abs(o-y)

        test_error.append(error)

        fw.write(row["graph"]+","+row["node"].replace(",","-")+","+str(len(row["node"].split(",")))+","+str(o)+","+str(y)+"\n")

    fw.close()


if __name__ == "__main__":
    main()