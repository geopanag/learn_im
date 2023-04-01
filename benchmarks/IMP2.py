from invgraph import Graph
from graph import pGraph
import random
import multiprocessing as mp
import time
import getopt
import sys
import math
import heapq
import ISE

import pandas as pd 
import numpy as np

class Worker(mp.Process):
    def __init__(self, inQ, outQ):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0

    def run(self):

        while True:
            theta = self.inQ.get()
            # print(theta)
            while self.count < theta:
                v = random.randint(1, node_num)
                rr = generate_rr(v)
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []


def create_worker(num):
    """
        create processes
        :param num: process number
        :param task_num: the number of tasks assigned to each worker
    """
    global worker
    for i in range(num):
        # print(i)
        worker.append(Worker(mp.Queue(), mp.Queue()))
        worker[i].start()


def finish_worker():
    for w in worker:
        w.terminate()


def sampling(epsoid, l):
    global graph, seed_size, worker
    R = []
    LB = 1
    n = node_num
    k = seed_size
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = 2
    create_worker(worker_num)
    for i in range(1, int(math.log2(n-1))+1):
        s = time.time()
        x = n/(math.pow(2, i))
        lambda_p = ((2+2*epsoid_p/3)*(logcnk(n, k) + l*math.log(n) + math.log(math.log2(n)))*n)/pow(epsoid_p, 2)
        theta = lambda_p/x
        # print(theta-len(R))
        for ii in range(worker_num):
            worker[ii].inQ.put((theta-len(R))/worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
        # finish_worker()
        # worker = []
        end = time.time()
        print('time to find rr', end - s)
        start = time.time()
        Si, f = node_selection(R, k)
        print(f)
        end = time.time()
        print('node selection time', time.time() - start)
        # print(F(R, Si))
        # f = F(R,Si)
        if n*f >= (1+epsoid_p)*x:
            LB = n*f/(1+epsoid_p)
            break
    # finish_worker()
    alpha = math.sqrt(l*math.log(n) + math.log(2))
    beta = math.sqrt((1-1/math.e)*(logcnk(n, k)+l*math.log(n)+math.log(2)))
    lambda_aster = 2*n*pow(((1-1/math.e)*alpha + beta), 2)*pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = theta - length_r
    # print(diff)
    _start = time.time()
    if diff > 0:
        # print('j')
        for ii in range(worker_num):
            worker[ii].inQ.put(diff/ worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
    '''
    
    while length_r <= theta:
        v = random.randint(1, n)
        rr = generate_rr(v)
        R.append(rr)
        length_r += 1
    '''
    _end = time.time()
    # print(_end - _start)
    finish_worker()
    return R


def generate_rr(v):
    global model
    if model == 'IC':
        return generate_rr_ic(v)
    elif model == 'LT':
        return generate_rr_lt(v)



def node_selection(R, k):
    Sk = set()
    rr_degree = [0 for ii in range(node_num+1)]
    node_rr_set = dict()
    # node_rr_set_copy = dict()
    matched_count = 0
    for j in range(0, len(R)):
        rr = R[j]
        for rr_node in rr:
            # print(rr_node)
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
                # node_rr_set_copy[rr_node] = list()
            node_rr_set[rr_node].append(j)
            # node_rr_set_copy[rr_node].append(j)
    for i in range(k):
        max_point = rr_degree.index(max(rr_degree))
        Sk.add(max_point)
        matched_count += len(node_rr_set[max_point])
        index_set = []
        for node_rr in node_rr_set[max_point]:
            index_set.append(node_rr)
        for jj in index_set:
            rr = R[jj]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(jj)
    return Sk, matched_count/len(R)



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
        
    return np.mean(spread), np.std(spread)


'''
def node_selection(R, k):
    # use CELF to accelerate
    Sk = set()
    node_rr_set = dict()
    rr_degree = [0 for ii in range(node_num + 1)]
    matched_count = 0
    for i, rr in enumerate(R):
        for v in rr:
            if v in node_rr_set:
                node_rr_set[v].add(i)
                rr_degree[v] += 1
            else:
                node_rr_set[v] = {i}
    max_heap = list()
    for key, value in node_rr_set.items():
        max_heap.append([-len(value), key, 0])
    heapq.heapify(max_heap)
    i = 0
    covered_set = set()
    while i < k:
        val = heapq.heappop(max_heap)
        if val[2] != i:
            node_rr_set[val[1]] -= covered_set
            val[0] = -len(node_rr_set[val[1]])
            val[2] = i
            heapq.heappush(max_heap, val)
        else:
            Sk.add(val[1])
            i += 1
            covered_set |= node_rr_set[val[1]]
    return Sk, len(covered_set) / len(R)
'''

def generate_rr_ic(node):
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in activity_nodes:
                    if random.random() < weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def generate_rr_lt(node):
    # calculate reverse reachable set using LT model
    # activity_set = list()
    activity_nodes = list()
    # activity_set.append(node)
    activity_nodes.append(node)
    activity_set = node

    while activity_set != -1:
        new_activity_set = -1

        neighbors = graph.get_neighbors(activity_set)
        if len(neighbors) == 0:
            break
        candidate = random.sample(neighbors, 1)[0][0]
        # print(candidate)
        if candidate not in activity_nodes:
            activity_nodes.append(candidate)
            # new_activity_set.append(candidate)
            new_activity_set = candidate
        activity_set = new_activity_set
    return activity_nodes


def imm(epsoid, l,k):
    n = node_num
    
    l = l*(1+ math.log(2)/math.log(n))
    R = sampling(epsoid, l)
    Sk, z = node_selection(R, k)
    return Sk


def logcnk(n, k):
    res = 0
    for i in range(n-k+1, n+1):
        res += math.log(i)
    for i in range(1, k+1):
        res -= math.log(i)
    return res


def read_file(network):
    """
    read network file into a graph and read seed file into a list
    :param network: the file path of network
    """
    global node_num, edge_num, graph, seeds, pgraph
    data_lines = open(network, 'r').readlines()
    node_num = int(data_lines[0].split()[0])
    edge_num = int(data_lines[0].split()[1])

    for data_line in data_lines[1:]:
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), float(weight))
        pgraph.add_edge(int(start), int(end), float(weight))


if __name__ == "__main__":
    """
    Based on https://github.com/snowgy/Influence_Maximization
  
        define global variables:
        node_num: total number of nodes in the network
        edge_num: total number of edges in the network
        graph: represents the network
        seeds: the list of seeds
    """
    """
    command line parameters
    usage: python3 IMP.py -i <graph file path> -k <the number of seeds> -m <IC or LT> -t <termination time> 
    """
    node_num = 0
    edge_num = 0
    model = 'IC'
    termination = 10
    
    
    fw = open("../imm_ic_result.txt","a")
    fw.write("------------------\n")
    for seed_size in [20,100]:#20,50,

        # start = time.time()
        #opts, args = getopt.getopt(sys.argv[1:], 'i:k:m:t:')

        #
        epsoid = 0.5
        l = 1
        for network_path in ["test_data/crime_ic.txt","test_data/gr_ic.txt","test_data/ht_ic.txt","test_data/enron_ic.txt","test_data/facebook_ic.txt","test_data/youtube_ic.txt"]:
            print(network_path)
            
            graph = Graph()
            pgraph = pGraph()
            start = time.time()    
            
            read_file(network_path)
            worker = []
            
            seeds = imm(epsoid, l,seed_size)

            end = time.time()
            tim = end - start
            print("done")
            print(tim)
            #fold = network_path.split("test_data/")[1].replace(".txt","")#.replace("_ic.txt","") 
            #gra = pd.read_csv("../imm/"+fold+"/graph_ic.inf",header=None,sep=" ")
            gra = pd.read_csv(network_path,header=None,sep=" ",skiprows=1)
            gra.columns = ["source","target","weight"]
            #gra["weight"] = 0.01
            
            seeds = [i for i in seeds]
            #for no in []:
            
            #if network_path in ["test_data/facebook.txt","test_data/youtube.txt"]:
            #    s1,st=  IC(gra,seeds,1000) #,1000)
            #else:
            s1,st=  IC(gra,seeds)
            print(s1)
            fw.write(str(seed_size)+","+str(round(tim))+","+str(round(s1))+"\n")
            fw.flush()

        fw.close()
  

