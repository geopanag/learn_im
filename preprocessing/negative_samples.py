import os
import igraph as ig
from tqdm import tqdm
import numpy as np 
import time
import glob
import pandas as pd
import random
import networkx as nx

from diffuse import IC





random.seed(1)
t = time.time()

x = pd.read_csv("influence_labels.csv",header=None)
x.columns = ["graph","node","infl"]
x["len"] = x.node.apply(lambda x: len(x.split(",")))

gs = x.graph.unique()
neg_samples = 30



labels = open("data/influence_train_set.csv","a")

for g in gs:
    
    #if "l" not in g or g=="lg4":
    #    continue
    print(g)
    #if g in done: 
    #    continue
    tmp = x[x.graph==g]
    
    G = pd.read_csv("sim_graphs/"+g,header=None,sep=" ")
    G.columns = ["source","target","weight"]
    nodes = set(G["target"].unique()).union(set(G["source"].unique()))
   
    #--- Find the best seed set
    for i,row in tmp.iterrows():
        
        seeds = row["node"].split(",")
        #--- Draw randomly other seed sets of the same length
        for k in range(neg_samples):
            #for j in range(row["len"]):
            neg_seeds =set(random.sample(nodes,row["len"]))
            counter = 0
            while neg_seeds==seeds:
                counter+=1
                neg_seeds =set(random.sample(nodes,row["len"]))
                if counter>10:
                    print("stuck")
                    
            sigma = IC(G,list(neg_seeds))
            #---- store everything
            labels.write(row["graph"]+',"'+",".join([str(ng) for ng in neg_seeds])+'",'+str(sigma)+"\n")
          # do it only for the first node 
        labels.write(row["graph"]+',"'+row["node"]+'",'+str(row["infl"])+"\n")
           
labels.close()
#log.write("labeling time "+str(time.time()-t))
#log.close()


