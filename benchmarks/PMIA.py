''' Implementation of PMIA algorithm [1].

[1] -- Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks.
'''

from __future__ import division
import networkx as nx
import math, time
from copy import deepcopy
from runIAC import avgIAC
import multiprocessing, json
from runIAC import avgIAC, runIAC
import os
import pandas as pd
import numpy as np

def updateAP(ap, S, PMIIAv, PMIIA_MIPv, Ep):
    ''' Assumption: PMIIAv is a directed tree, which is a subgraph of general G.
    PMIIA_MIPv -- dictionary of MIP from nodes in PMIIA
    PMIIAv is rooted at v.
    '''
    # going from leaves to root
    sorted_MIPs = sorted(PMIIA_MIPv.iteritems(), key = lambda (_, MIP): len(MIP), reverse = True)
    for u, _ in sorted_MIPs:
        if u in S:
            ap[(u, PMIIAv)] = 1
        elif not PMIIAv.in_edges([u]):
            ap[(u, PMIIAv)] = 0
        else:
            in_edges = PMIIAv.in_edges([u], data=True)
            prod = 1
            for w, _, edata in in_edges:
                # p = (1 - (1 - Ep[(w, u)])**edata["weight"])
                p = Ep[(w,u)]
                prod *= 1 - ap[(w, PMIIAv)]*p
            ap[(u, PMIIAv)] = 1 - prod

def updateAlpha(alpha, v, S, PMIIAv, PMIIA_MIPv, Ep, ap):
    # going from root to leaves
    sorted_MIPs =  sorted(PMIIA_MIPv.iteritems(), key = lambda (_, MIP): len(MIP))
    for u, mip in sorted_MIPs:
        if u == v:
            alpha[(PMIIAv, u)] = 1
        else:
            out_edges = PMIIAv.out_edges([u])
            assert len(out_edges) == 1, "node u=%s must have exactly one neighbor, got %s instead" %(u, len(out_edges))
            w = out_edges[0][1]
            if w in S:
                alpha[(PMIIAv, u)] = 0
            else:
                in_edges = PMIIAv.in_edges([w], data=True)
                prod = 1
                for up, _, edata in in_edges:
                    if up != u:
                        # pp_upw = 1 - (1 - Ep[(up, w)])**edata["weight"]
                        pp_upw = Ep[(up, w)]
                        prod *= (1 - ap[(up, PMIIAv)]*pp_upw)
                # alpha[(PMIIAv, u)] = alpha[(PMIIAv, w)]*(1 - (1 - Ep[(u,w)])**PMIIAv[u][w]["weight"])*prod
                alpha[(PMIIAv, u)] = alpha[(PMIIAv, w)]*(Ep[(u,w)])*prod

def computePMIOA(G, u, theta, S, Ep):
    '''
     Compute PMIOA -- subgraph of G that's rooted at u.
     Uses Dijkstra's algorithm until length of path doesn't exceed -log(theta)
     or no more nodes can be reached.
    '''
    # initialize PMIOA
    PMIOA = nx.DiGraph()
    PMIOA.add_node(u)
    PMIOA_MIP = {u: [u]} # MIP(u,v) for v in PMIOA

    crossing_edges = set([out_edge for out_edge in G.out_edges([u]) if out_edge[1] not in S + [u]])
    edge_weights = dict()
    dist = {u: 0} # shortest paths from the root u

    # grow PMIOA
    while crossing_edges:
        # Dijkstra's greedy criteria
        min_dist = float("Inf")
        sorted_crossing_edges = sorted(crossing_edges) # to break ties consistently
        for edge in sorted_crossing_edges:
            if edge not in edge_weights:
                # edge_weights[edge] = -math.log(1 - (1 - Ep[edge])**G[edge[0]][edge[1]]["weight"])
                edge_weights[edge] = -math.log(Ep[edge])
            edge_weight = edge_weights[edge]
            if dist[edge[0]] + edge_weight < min_dist:
                min_dist = dist[edge[0]] + edge_weight
                min_edge = edge
        # check stopping criteria
        if min_dist < -math.log(theta):
            dist[min_edge[1]] = min_dist
            # PMIOA.add_edge(min_edge[0], min_edge[1], {"weight": G[min_edge[0]][min_edge[1]]["weight"]})
            PMIOA.add_edge(min_edge[0], min_edge[1])
            PMIOA_MIP[min_edge[1]] = PMIOA_MIP[min_edge[0]] + [min_edge[1]]
            # update crossing edges
            crossing_edges.difference_update(G.in_edges(min_edge[1]))
            crossing_edges.update([out_edge for out_edge in G.out_edges(min_edge[1])
                                   if (out_edge[1] not in PMIOA) and (out_edge[1] not in S)])
        else:
            break
    return PMIOA, PMIOA_MIP

def updateIS(IS, S, u, PMIOA, PMIIA):
    for v in PMIOA[u]:
        for si in S:
            # if seed node is effective and it's blocked by u
            # then it becomes ineffective
            if (si in PMIIA[v]) and (si not in IS[v]) and (u in PMIIA[v][si]):
                    IS[v].append(si)

def computePMIIA(G, ISv, v, theta, S, Ep):

    # initialize PMIIA
    PMIIA = nx.DiGraph()
    PMIIA.add_node(v)
    PMIIA_MIP = {v: [v]} # MIP(u,v) for u in PMIIA

    crossing_edges = set([in_edge for in_edge in G.in_edges([v]) if in_edge[0] not in ISv + [v]])
    edge_weights = dict()
    dist = {v: 0} # shortest paths from the root u

    # grow PMIIA
    while crossing_edges:
        # Dijkstra's greedy criteria
        min_dist = float("Inf")
        sorted_crossing_edges = sorted(crossing_edges) # to break ties consistently
        for edge in sorted_crossing_edges:
            if edge not in edge_weights:
                # edge_weights[edge] = -math.log(1 - (1 - Ep[edge])**G[edge[0]][edge[1]]["weight"])
                edge_weights[edge] = -math.log(Ep[edge])
            edge_weight = edge_weights[edge]
            if dist[edge[1]] + edge_weight < min_dist:
                min_dist = dist[edge[1]] + edge_weight
                min_edge = edge
        # check stopping criteria
        # print min_edge, ':', min_dist, '-->', -math.log(theta)
        if min_dist < -math.log(theta):
            dist[min_edge[0]] = min_dist
            # PMIIA.add_edge(min_edge[0], min_edge[1], {"weight": G[min_edge[0]][min_edge[1]]["weight"]})
            PMIIA.add_edge(min_edge[0], min_edge[1])
            PMIIA_MIP[min_edge[0]] = PMIIA_MIP[min_edge[1]] + [min_edge[0]]
            # update crossing edges
            crossing_edges.difference_update(G.out_edges(min_edge[0]))
            if min_edge[0] not in S:
                crossing_edges.update([in_edge for in_edge in G.in_edges(min_edge[0])
                                       if (in_edge[0] not in PMIIA) and (in_edge[0] not in ISv)])
        else:
            break
    return PMIIA, PMIIA_MIP

def PMIA(G, k, theta, Ep):
    start = time.time()
    # initialization
    S = []
    IncInf = dict(zip(G.nodes(), [0]*len(G)))
    PMIIA = dict() # node to tree
    PMIOA = dict()
    PMIIA_MIP = dict() # node to MIPs (dict)
    PMIOA_MIP = dict()
    ap = dict()
    alpha = dict()
    IS = dict()
    for v in G:
        IS[v] = []
        PMIIA[v], PMIIA_MIP[v] = computePMIIA(G, IS[v], v, theta, S, Ep)
        for u in PMIIA[v]:
            ap[(u, PMIIA[v])] = 0 # ap of u node in PMIIA[v]
        updateAlpha(alpha, v, S, PMIIA[v], PMIIA_MIP[v], Ep, ap)
        for u in PMIIA[v]:
            IncInf[u] += alpha[(PMIIA[v], u)]*(1 - ap[(u, PMIIA[v])])
    print 'Finished initialization'
    print time.time() - start

    # main loop
    for i in range(k):
        u, _ = max(IncInf.iteritems(), key = lambda (dk, dv): dv)
        # print i+1, "node:", u, "-->", IncInf[u]
        IncInf.pop(u) # exclude node u for next iterations
        PMIOA[u], PMIOA_MIP[u] = computePMIOA(G, u, theta, S, Ep)
        for v in PMIOA[u]:
            for w in PMIIA[v]:
                if w not in S + [u]:
                    IncInf[w] -= alpha[(PMIIA[v],w)]*(1 - ap[(w, PMIIA[v])])

        updateIS(IS, S, u, PMIOA_MIP, PMIIA_MIP)

        S.append(u)

        for v in PMIOA[u]:
            if v != u:
                PMIIA[v], PMIIA_MIP[v] = computePMIIA(G, IS[v], v, theta, S, Ep)
                updateAP(ap, S, PMIIA[v], PMIIA_MIP[v], Ep)
                updateAlpha(alpha, v, S, PMIIA[v], PMIIA_MIP[v], Ep, ap)
                # add new incremental influence
                for w in PMIIA[v]:
                    if w not in S:
                        IncInf[w] += alpha[(PMIIA[v], w)]*(1 - ap[(w, PMIIA[v])])

    return S


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



def getCoverage((G, S, Ep)):
    return len(runIAC(G, S, Ep))

if __name__ == "__main__":
    """
    Based on https://github.com/nd7141/influence-maximization/blob/master/IC/
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


        Ep = dict()
        with open(g) as f:
            for line in f:
                data = line.split()
                Ep[(int(data[0]), int(data[1]))] = float(data[2])

        
        #DROPBOX_FOLDER = "/home/sergey/Dropbox/Influence Maximization"
        seeds_filename = "../results/pmia/"+g.replace("real/","fseeds_n_")#SEEDS_FOLDER + "/%s_%s_%s_%s.txt" %(SEEDS_FOLDER, ALGO_NAME, dataset, model)
        spread_filename = "../results/pmia/"+g.replace("real/","fspread_n_")#SEEDS_FOLDER + "/%s_%s_%s_%s.txt" %(SEEDS_FOLDER, 
        time_filename = "../results/pmia/"+g.replace("real/","ftime_n_")#TIME_FOLDER + "/%s_%s_%s_%s.txt" %(TIME_FOLDER, ALGO_NAME, dataset, model)
        seeds_file = open(seeds_filename, "a+")
        spread_file = open(spread_filename, "a+")
        
        time_file = open(time_filename, "a+")
        theta = 1.0/5
        
        pool = None
        I = 1000
        l2c = [[0, 0]]
        # open file for writing output
        seeds_file = open(seeds_filename, "a+")
        time_file = open(time_filename, "a+")
        #dbox_seeds_file = open("%/%", DROPBOX_FOLDER, seeds_filename, "a+")
        #dbox_time_file = open("%/%", DROPBOX_FOLDER, time_filename, "a+")
        
        S = PMIA(G, seed_set_size, theta, Ep)
        
        #time2S = time.time()
            
        #time2complete = time.time() - time2S
        #print >>time_file, (time2complete)
        #print >>dbox_time_file, (time2complete)
        time2complete = time.time() - start
        print('evaluation')
        print >>seeds_file, json.dumps(S)
        print >> time_file, (time2complete)
        seeds_file.flush()
        time_file.flush()
        
        G = pd.read_csv(g,header=None,sep=" ")
        G.columns = ["source","target","weight"]
        #------ store everything
        sigma  = IC(G,S[:20],1000)
        sigma2 = IC(G,S[:50],1000)
        sigma3 = IC(G,S[:100],1000)
        sigma4 = IC(G,S[:200],1000)#
        
        print >> spread_file, (sigma,sigma2,sigma3,sigma4)
        spread_file.flush()
        
        #print 'Total time for length = %s: %s sec' %(length, time.time() - time2length)
        #print '----------------------------------------------'


    seeds_file.close()
    #dbox_seeds_file.close()
    time_file.close()
    