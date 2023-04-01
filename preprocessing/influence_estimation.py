import os
import igraph as ig
from tqdm import tqdm
import numpy as np 
import time
import glob
import pandas as pd
import networkx as nx
from diffuse import IC#,dmpest
import random





random.seed(1)


t = time.time() 

labels = open("../influence_labels.csv","a")


seed_size= 10

for g in tqdm(glob.glob("*")):
    print(g)
    G = pd.read_csv(g,header=None,sep=" ")
    G.columns = ["source","target","weight"]
    nodes = set(G["target"].unique()).union(set(G["source"].unique()))
    
    
    Q = [] 
    S = []

    nid = 0
    mg = 1
    iteration = 2

    for u in tqdm(G.nodes()):
        temp_l = []
        temp_l.append(u)
        temp_l.append( IC(G,[u]) ) #
        temp_l.append(0) #iteration
        Q.append(temp_l)

    Q = sorted (Q, key=lambda x:x[1],reverse=True)

    S = [Q[0][0]]
    infl_spread = Q[0][1]
    labels.write(g.replace(".txt","")+',"'+','.join([str(tm) for tm in S])+'",'+str(infl_spread)+"\n")
    labels.flush()
    Q = Q[1:]
    while len(S) < seed_size :
        
        u = Q[0]
        # check if the node is already sorted
        if (u[iteration] != len(S)):
            #----- Update this node
            #------- Keep only the number of nodes influenceed to rank the candidate seed        
            u[mg] = IC(G,S+[u[nid]],100)-infl_spread  
            u[iteration] = len(S)
            Q = sorted(Q, key=lambda x:x[1],reverse=True)

        else:
            print(len(S))
            #----- Store the new seed
            infl_spread = u[mg]+infl_spread#simulate_spreading(subg,S+[u[nid]],sim)
            
            S.append(u[nid])
            labels.write(g.replace(".txt","")+',"'+','.join([str(tm) for tm in S])+'",'+str(infl_spread)+"\n")

            #----- Delete uid from Q
            Q = Q[1:]#[l for l in Q if l[0] != u[nid]]##
            
labels.close()

