from priorityQueue import PriorityQueue as PQ # priority queue
import os
import networkx as nx
import pandas as pd
import numpy as np
import json

def IC(G,S,mc=1000):
    """
    From https://hautahi.com/im_greedycelf
    Input:  graph pandas df, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):
        
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        while new_active:
            temp = G.loc[G['source'].isin(new_active)]
            # For each newly active node, find its neighbors that become activated
            targets = temp['target'].tolist()
            ic = temp['weight'].tolist()
            # Determine the neighbors that become infected
            coins  = np.random.uniform(0,1,len(targets))
            choice = [ic[c]>coins[c] for c in range(len(coins))]
            #sum(choice)

            new_ones = np.extract(choice, targets)

            # Create a list of nodes that weren't previously activated
            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active
           
        spread.append(len(A))
        
    return np.mean(spread)#, np.std(spread)



def degreeDiscountIC2(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (without priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    Note: the routine runs twice slower than using PQ. Implemented to verify results
    '''
    d = dict()
    dd = dict() # degree discount
    t = dict() # number of selected neighbors
    S = [] # selected set of nodes
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        u, ddv = max(dd.iteritems(), key=lambda (k,v): v)
        dd.pop(u)
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
    return S
    
def degreeDiscountStar(G,k,p=.01):
    
    S = []
    scores = PQ()
    d = dict()
    t = dict()
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])
        t[u] = 0
        score = -((1-p)**t[u])*(1+(d[u]-t[u])*p)
        scores.add_task(u, )
    for iteration in range(k):
        u, priority = scores.pop_item()
        print iteration, -priority
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']
                score = -((1-p)**t[u])*(1+(d[u]-t[u])*p)
                scores.add_task(v, score)
    return S
            

def degreeDiscountIC(G, k,d):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S
    #d = dict() # degree of each vertex
    
    
    # initialize degree discount
    for u in G.nodes():
        #sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += 1 # increase number of selected neighbors
                priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*G[u][v]['weight'] #*EP[(u,v)] # discount of degree
                dd.add_task(v, -priority)
    return S



if __name__ == "__main__":
    """
    Based on https://github.com/nd7141/influence-maximization/blob/master/IC/degreeDiscount.py
    """
    import time

    model = "Categories"

    #if model == "MultiValency":
    #    ep_model = "range"
    #elif model == "Random":
    #    ep_model = "random"
    #elif model == "Categories":
    #    ep_model = "degree"

    
    seed_set_size = 100
    #,"real/gnutella31_ic.inf", 
    for g in ["real/crime_ic.inf","real/gr_ic.inf","real/ht_ic.inf","real/enron_ic.inf","real/facebook_ic.inf","real/youtube_ic.inf"]:  
        start = time.time()  
        G = nx.read_edgelist(g,  nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        print 'Read graph G'
        print time.time() - start
        d = dict(G.degree())
        #print(G[779][780]["weight"])
        
        seeds_filename = "../results/dd/"+g.replace("real/","fseeds_n_")
        spread_filename = "../results/dd/"+g.replace("real/","fspread_n_")
        
        seeds_file = open(seeds_filename, "a+")
        spread_file = open(spread_filename, "a+")
        
        time_file = open(time_filename, "a+")
        theta = 1.0/5
        
        
        # open file for writing output
        seeds_file = open(seeds_filename, "a+")
        time_file = open(time_filename, "a+")
        
        S = degreeDiscountIC(G, seed_set_size,d)
        
        time2complete = time.time() - start
        print('evaluation')
        print >>seeds_file, json.dumps(S)
        print >> time_file, (time2complete)
        seeds_file.flush()
        time_file.flush()
        
        G = pd.read_csv(g,header=None,sep=" ")
        G.columns = ["source","target","weight"]
        #------ store everything
        sigma  = round(IC(G,S[:20],10000))
        sigma2 = round(IC(G,S[:50],10000))
        sigma3 = round(IC(G,S[:100],10000))
        sigma4 = round(IC(G,S[:200],10000))
        
        print >> spread_file, (sigma,sigma2,sigma3,sigma4)
        spread_file.flush()
        
        #print 'Total time for length = %s: %s sec' %(length, time.time() - time2length)
        #print '----------------------------------------------'


    seeds_file.close()
    #dbox_seeds_file.close()
    time_file.close()