
def dmpest(adj,seed_set,T=10):
    """
    Dynamic message passing for influence estimation 
    (algorithm 1) in https://arxiv.org/pdf/1912.12749.pdf 
    adj: adjacency of the graph, used for computing p i->j in a vectorized manner
    """

    p0 = np.zeros(adj.shape[0])
    p0[seed_set]=1

    pi = p0

    p = np.array([list(pi),]*adj.shape[0]).T
    p_= p

    # compute iteratively p j-> i hat using eq 13 
    x = np.nonzero(adj.T)
    G_ = [(i,j) for i,j in zip(x[0],x[1])]

    for t in range(T):
        q = 1- p.T*adj
        q_ = np.prod(q,axis=1)
        qq = (1-p0)*q_

        #for e in G.edges():
        for i,j in G_:#.edges():
            #print(i,j)
            # p l-> j probability of j getting influenced by its neighbors before i
            if(q[j,i]!=0):
                p_[j,i] = 1 -qq[j]/q[j,i]
        p = p_


    # finally compute pi through eq 14    

    # prob of i getting influenced
    q = 1-adj*p.T
    pi = 1 -(1-p0)*np.prod(q,axis=1)

    return sum(pi)   

